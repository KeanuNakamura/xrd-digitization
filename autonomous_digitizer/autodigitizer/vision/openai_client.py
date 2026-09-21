"""Robust OpenAI Responses API client with caching and structured outputs."""

from __future__ import annotations

import base64
import copy
import io
import json
import logging
import time
from pathlib import Path
from typing import Any, Optional, Type, TypeVar

import numpy as np
from PIL import Image
from pydantic import BaseModel, ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from autodigitizer.config import Config

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class OpenAIClientError(RuntimeError):
    pass


class OpenAIVisionClient:
    """Thin wrapper around the OpenAI Responses API for vision + JSON schema."""

    def __init__(self, config: Config):
        self.config = config
        self.api_calls = 0
        self.cache_hits = 0
        self._client = None

    def _get_client(self):
        if self._client is None:
            self.config.require_api_key()
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self.config.openai_api_key,
                timeout=self.config.request_timeout_s,
            )
        return self._client

    def _cache_path(self, key: str) -> Path:
        d = self.config.cache_dir / key
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _read_cache(self, key: str, name: str) -> Optional[dict]:
        if not self.config.use_cache:
            return None
        path = self._cache_path(key) / name
        if path.exists():
            self.cache_hits += 1
            logger.info("Cache hit: %s/%s", key, name)
            return json.loads(path.read_text())
        return None

    def _write_cache(self, key: str, name: str, data: dict) -> None:
        if not self.config.use_cache:
            return
        path = self._cache_path(key) / name
        path.write_text(json.dumps(data, indent=2))

    @staticmethod
    def encode_image(image: np.ndarray | Path | str, max_side: int = 2048) -> str:
        if isinstance(image, (str, Path)):
            with Image.open(image) as im:
                im = im.convert("RGB")
                arr = np.asarray(im)
        else:
            arr = image
        h, w = arr.shape[:2]
        scale = min(1.0, max_side / max(h, w))
        if scale < 1.0:
            nh, nw = int(h * scale), int(w * scale)
            im = Image.fromarray(arr).resize((nw, nh), Image.Resampling.LANCZOS)
        else:
            im = Image.fromarray(arr)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def analyze_structured(
        self,
        *,
        prompt: str,
        schema_model: Type[T],
        images: list[np.ndarray | Path | str],
        cache_key: str,
        cache_name: str,
        system: str | None = None,
    ) -> T:
        cached = self._read_cache(cache_key, cache_name)
        if cached is not None:
            try:
                return schema_model.model_validate(cached)
            except ValidationError:
                logger.warning("Cached payload failed validation; refetching")

        raw = self._call_responses(
            prompt=prompt,
            schema_model=schema_model,
            images=images,
            system=system,
        )
        self._write_cache(cache_key, cache_name, raw)
        try:
            return schema_model.model_validate(raw)
        except ValidationError as exc:
            logger.warning("Structured validation failed: %s; attempting repair", exc)
            repair = self._call_responses(
                prompt=(
                    "Return corrected JSON matching the required schema. "
                    f"Validation errors: {exc}\n\nPrevious JSON:\n{json.dumps(raw)}"
                ),
                schema_model=schema_model,
                images=images[:1],
                system=system,
            )
            self._write_cache(cache_key, cache_name, repair)
            return schema_model.model_validate(repair)

    def _call_responses(
        self,
        *,
        prompt: str,
        schema_model: Type[BaseModel],
        images: list[np.ndarray | Path | str],
        system: str | None,
    ) -> dict[str, Any]:
        return self._call_with_retries(
            prompt=prompt,
            schema_model=schema_model,
            images=images,
            system=system,
        )

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        retry=retry_if_exception_type((OpenAIClientError, TimeoutError, ConnectionError)),
    )
    def _call_with_retries(
        self,
        *,
        prompt: str,
        schema_model: Type[BaseModel],
        images: list[np.ndarray | Path | str],
        system: str | None,
    ) -> dict[str, Any]:
        client = self._get_client()
        # Ensure prompts mention JSON for json_object fallback compatibility.
        user_text = prompt
        if "json" not in user_text.lower():
            user_text = user_text.rstrip() + "\n\nRespond with JSON only."

        content: list[dict[str, Any]] = [{"type": "input_text", "text": user_text}]
        for img in images:
            b64 = self.encode_image(img)
            content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{b64}",
                }
            )

        schema = to_openai_strict_schema(schema_model.model_json_schema())
        body_input: list[dict[str, Any]] = []
        if system:
            sys_text = system
            if "json" not in sys_text.lower():
                sys_text = sys_text.rstrip() + " Always respond with valid JSON."
            body_input.append(
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": sys_text}],
                }
            )
        body_input.append({"role": "user", "content": content})

        self.api_calls += 1
        t0 = time.time()
        try:
            response = client.responses.create(
                model=self.config.model,
                input=body_input,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": schema_model.__name__,
                        "schema": schema,
                        "strict": True,
                    }
                },
            )
        except Exception as exc:
            logger.error("OpenAI strict schema request failed: %s", exc)
            # Fallback: json_object mode (requires 'json' in input — already ensured)
            try:
                response = client.responses.create(
                    model=self.config.model,
                    input=body_input,
                    text={"format": {"type": "json_object"}},
                )
            except Exception as exc2:
                raise OpenAIClientError(str(exc2)) from exc2

        elapsed = time.time() - t0
        logger.info("OpenAI call #%d completed in %.1fs", self.api_calls, elapsed)
        text = _extract_output_text(response)
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise OpenAIClientError(f"Non-JSON response: {text[:500]}") from exc


def _extract_output_text(response: Any) -> str:
    if hasattr(response, "output_text") and response.output_text:
        return response.output_text
    chunks: list[str] = []
    for item in getattr(response, "output", []) or []:
        for part in getattr(item, "content", []) or []:
            if getattr(part, "type", None) in ("output_text", "text"):
                chunks.append(getattr(part, "text", ""))
    if not chunks and isinstance(response, dict):
        return json.dumps(response)
    if not chunks:
        raise OpenAIClientError("Empty response from OpenAI")
    return "".join(chunks)


def to_openai_strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Convert a Pydantic JSON schema into OpenAI strict structured-output form.

    OpenAI requires for every object:
    - additionalProperties: false
    - required: array listing EVERY key in properties
    """
    root = copy.deepcopy(schema)
    # Drop Pydantic root metadata that confuses some validators
    root.pop("title", None)
    root.pop("$schema", None)
    root.pop("description", None)

    defs_key = "$defs" if "$defs" in root else ("definitions" if "definitions" in root else None)
    if defs_key:
        root[defs_key] = {
            name: _strictify_object(defn) for name, defn in root[defs_key].items()
        }

    return _strictify_object(root)


def _strictify_object(node: Any) -> Any:
    if isinstance(node, list):
        return [_strictify_object(x) for x in node]

    if not isinstance(node, dict):
        return node

    node = dict(node)

    # Recurse into nested schema containers first
    for key in ("properties", "$defs", "definitions"):
        if key in node and isinstance(node[key], dict):
            node[key] = {k: _strictify_object(v) for k, v in node[key].items()}

    for key in ("items", "contains", "not"):
        if key in node:
            node[key] = _strictify_object(node[key])

    for key in ("anyOf", "oneOf", "allOf", "prefixItems"):
        if key in node and isinstance(node[key], list):
            node[key] = [_strictify_object(x) for x in node[key]]

    # Free-form objects (dict[str, Any]) are illegal in strict mode.
    if node.get("type") == "object" and node.get("additionalProperties") not in (None, False):
        # Collapse to empty closed object; callers should avoid free-form dicts.
        node["properties"] = node.get("properties") or {}
        node["additionalProperties"] = False

    if "properties" in node and isinstance(node["properties"], dict):
        node["type"] = "object"
        node["additionalProperties"] = False
        node["required"] = list(node["properties"].keys())
        node["properties"] = {
            k: _strictify_object(v) for k, v in node["properties"].items()
        }

    # Enum / const defs without properties
    if node.get("type") == "object" and "properties" not in node and "$ref" not in node:
        node["properties"] = {}
        node["additionalProperties"] = False
        node["required"] = []

    for bad in ("title", "default", "examples", "description"):
        node.pop(bad, None)

    return node
