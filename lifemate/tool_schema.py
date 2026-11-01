"""Adapt executor JSON Schemas to provider-compatible function parameters.

The official SDK always validates the original catalog schema before execution.
This module only adapts the model-facing copy: OpenAI function parameters must
have an object root and reject root-level composition keywords even in
non-strict mode.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


class ToolSchemaCompatibilityError(ValueError):
    """A tool schema cannot be represented safely for function calling."""


FORBIDDEN_ROOT_KEYWORDS = frozenset({"oneOf", "anyOf", "allOf", "enum", "const", "not"})

_SCALAR_KEYS = frozenset(
    {
        "type",
        "title",
        "description",
        "enum",
        "const",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "pattern",
        "minItems",
        "maxItems",
        "uniqueItems",
        "minProperties",
        "maxProperties",
        "format",
        "default",
    }
)
_STRUCTURAL_KEYS = frozenset(
    {
        "properties",
        "required",
        "additionalProperties",
        "items",
        "oneOf",
        "anyOf",
        "allOf",
        "not",
    }
)
_META_KEYS = frozenset({"$ref", "$defs", "definitions", "nullable", "x-one-of", "x-ref", "x-match"})
_KNOWN_KEYS = _SCALAR_KEYS | _STRUCTURAL_KEYS | _META_KEYS


def adapt_function_parameters(raw: Any) -> dict[str, Any]:
    """Return an object-root function schema without forbidden root keywords.

    Root object unions are projected into one property set. Cross-field union
    requirements are retained in the description and remain authoritative in
    the original executor schema used by ``env.validate``.
    """

    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ToolSchemaCompatibilityError("function parameters must be a JSON Schema object")
    document = _plain_mapping(raw)
    normalized = _normalize_node(document, document=document, path="$", ref_stack=())
    return _flatten_root_object(normalized, path="$")


def _normalize_node(
    raw: Mapping[str, Any],
    *,
    document: Mapping[str, Any],
    path: str,
    ref_stack: tuple[str, ...],
) -> dict[str, Any]:
    unknown = set(raw) - _KNOWN_KEYS
    if unknown:
        raise ToolSchemaCompatibilityError(f"{path}: unsupported schema keywords: {sorted(unknown)}")

    reference = raw.get("$ref")
    if reference is not None:
        if not isinstance(reference, str) or not reference.startswith("#/"):
            raise ToolSchemaCompatibilityError(f"{path}: only local $ref values are supported")
        if reference in ref_stack:
            raise ToolSchemaCompatibilityError(f"{path}: cyclic $ref is not supported: {reference}")
        target = _resolve_pointer(document, reference, path)
        resolved = _normalize_node(
            target,
            document=document,
            path=f"{path}->$ref({reference})",
            ref_stack=(*ref_stack, reference),
        )
        siblings = {key: value for key, value in raw.items() if key != "$ref"}
        if not siblings:
            return resolved
        normalized_siblings = _normalize_node(
            siblings,
            document=document,
            path=path,
            ref_stack=ref_stack,
        )
        return {"allOf": [resolved, normalized_siblings]}

    out: dict[str, Any] = {}
    for key in _SCALAR_KEYS:
        if key in raw:
            out[key] = _plain_value(raw[key])

    properties = raw.get("properties")
    if properties is not None:
        if not isinstance(properties, Mapping):
            raise ToolSchemaCompatibilityError(f"{path}.properties must be an object")
        out["properties"] = {
            str(name): _normalize_node(
                value,
                document=document,
                path=f"{path}.properties.{name}",
                ref_stack=ref_stack,
            )
            for name, value in properties.items()
            if _require_mapping(value, f"{path}.properties.{name}")
        }

    required = raw.get("required")
    if required is not None:
        if not _is_string_sequence(required):
            raise ToolSchemaCompatibilityError(f"{path}.required must be an array of strings")
        out["required"] = list(required)

    items = raw.get("items")
    if items is not None:
        if not isinstance(items, Mapping):
            raise ToolSchemaCompatibilityError(f"{path}.items must be an object schema")
        out["items"] = _normalize_node(
            items,
            document=document,
            path=f"{path}.items",
            ref_stack=ref_stack,
        )

    additional = raw.get("additionalProperties")
    if additional is not None:
        if isinstance(additional, bool):
            out["additionalProperties"] = additional
        elif isinstance(additional, Mapping):
            out["additionalProperties"] = _normalize_node(
                additional,
                document=document,
                path=f"{path}.additionalProperties",
                ref_stack=ref_stack,
            )
        else:
            raise ToolSchemaCompatibilityError(f"{path}.additionalProperties must be boolean or a schema")

    for keyword in ("oneOf", "anyOf", "allOf"):
        variants = raw.get(keyword)
        if variants is None:
            continue
        if not isinstance(variants, Sequence) or isinstance(variants, (str, bytes)) or not variants:
            raise ToolSchemaCompatibilityError(f"{path}.{keyword} must be a non-empty schema array")
        out[keyword] = [
            _normalize_node(
                variant,
                document=document,
                path=f"{path}.{keyword}[{index}]",
                ref_stack=ref_stack,
            )
            for index, variant in enumerate(variants)
            if _require_mapping(variant, f"{path}.{keyword}[{index}]")
        ]

    negated = raw.get("not")
    if negated is not None:
        if not isinstance(negated, Mapping):
            raise ToolSchemaCompatibilityError(f"{path}.not must be an object schema")
        out["not"] = _normalize_node(
            negated,
            document=document,
            path=f"{path}.not",
            ref_stack=ref_stack,
        )

    groups = raw.get("x-one-of")
    if groups is not None:
        if not _is_group_sequence(groups):
            raise ToolSchemaCompatibilityError(f"{path}.x-one-of must contain non-empty string arrays")
        out["x-one-of"] = [list(group) for group in groups]

    annotations: list[str] = []
    if raw.get("x-ref"):
        annotations.append(f"Reference: {raw['x-ref']}. Obtain this ID from a permitted read result; never guess it.")
    if raw.get("x-match"):
        annotations.append(f"Matching: {raw['x-match']}; preserve the user's free text.")
    if annotations:
        out["description"] = _join_description(out.get("description"), *annotations)

    nullable = raw.get("nullable")
    if nullable is True:
        _make_nullable(out, path)
    elif nullable is not None and nullable is not False:
        raise ToolSchemaCompatibilityError(f"{path}.nullable must be boolean")
    return out


def _flatten_root_object(schema: Mapping[str, Any], *, path: str) -> dict[str, Any]:
    for keyword in ("enum", "const", "not"):
        if keyword in schema:
            raise ToolSchemaCompatibilityError(
                f"{path}: root-level {keyword} cannot be represented safely as function parameters"
            )

    root_type = schema.get("type")
    compositions = any(keyword in schema for keyword in ("oneOf", "anyOf", "allOf"))
    if root_type is not None and root_type != "object":
        raise ToolSchemaCompatibilityError(f"{path}: function parameters must resolve to type object")
    if root_type is None and not compositions and "properties" not in schema:
        root_type = "object"

    properties = _property_mapping(schema.get("properties"), f"{path}.properties")
    required = set(_required_names(schema.get("required"), f"{path}.required"))
    notes: list[str] = []

    for variant_index, variant in enumerate(schema.get("allOf", [])):
        flattened = _flatten_root_object(variant, path=f"{path}.allOf[{variant_index}]")
        properties = _merge_properties(properties, flattened["properties"], mode="allOf")
        required.update(flattened.get("required", []))

    for keyword in ("oneOf", "anyOf"):
        variants = schema.get(keyword, [])
        if not variants:
            continue
        flattened_variants = [
            _flatten_root_object(variant, path=f"{path}.{keyword}[{index}]")
            for index, variant in enumerate(variants)
        ]
        variant_required = [set(variant.get("required", [])) for variant in flattened_variants]
        common_required = set.intersection(*variant_required) if variant_required else set()
        required.update(common_required)
        for variant in flattened_variants:
            properties = _merge_properties(properties, variant["properties"], mode="anyOf")
        choices = " OR ".join(_format_required_group(names) for names in variant_required)
        quantifier = "Exactly one" if keyword == "oneOf" else "At least one"
        notes.append(f"{quantifier} argument variant must be satisfied: {choices}.")

    groups = schema.get("x-one-of", [])
    if groups:
        unknown = sorted({name for group in groups for name in group} - set(properties))
        if unknown:
            raise ToolSchemaCompatibilityError(f"{path}.x-one-of references unknown properties: {unknown}")
        notes.append(
            "At least one complete argument group is required: "
            + " OR ".join(" + ".join(group) for group in groups)
            + "."
        )

    out: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    for key in ("title", "description", "minProperties", "maxProperties"):
        if key in schema:
            out[key] = _plain_value(schema[key])
    if notes:
        out["description"] = _join_description(
            out.get("description"),
            *notes,
            "The original executor schema remains authoritative before dispatch.",
        )
    if required:
        out["required"] = sorted(required)
    forbidden = FORBIDDEN_ROOT_KEYWORDS & set(out)
    if forbidden:
        raise AssertionError(f"adapter emitted forbidden root keywords: {sorted(forbidden)}")
    return out


def _merge_properties(
    current: Mapping[str, Any], incoming: Mapping[str, Any], *, mode: str
) -> dict[str, Any]:
    merged = {name: _plain_value(schema) for name, schema in current.items()}
    for name, schema in incoming.items():
        plain = _plain_value(schema)
        if name not in merged:
            merged[name] = plain
        elif merged[name] != plain:
            keyword = "allOf" if mode == "allOf" else "anyOf"
            existing = merged[name]
            if isinstance(existing, Mapping) and set(existing) == {keyword}:
                alternatives = list(existing[keyword])
            else:
                alternatives = [existing]
            if plain not in alternatives:
                alternatives.append(plain)
            merged[name] = {keyword: alternatives}
    return merged


def _resolve_pointer(document: Mapping[str, Any], reference: str, path: str) -> Mapping[str, Any]:
    current: Any = document
    for raw_part in reference[2:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, Mapping) or part not in current:
            raise ToolSchemaCompatibilityError(f"{path}: unresolved $ref {reference}")
        current = current[part]
    if not isinstance(current, Mapping):
        raise ToolSchemaCompatibilityError(f"{path}: $ref {reference} does not resolve to an object schema")
    return current


def _make_nullable(out: dict[str, Any], path: str) -> None:
    value_type = out.get("type")
    if isinstance(value_type, str):
        out["type"] = [value_type, "null"] if value_type != "null" else "null"
    elif _is_string_sequence(value_type):
        out["type"] = list(dict.fromkeys([*value_type, "null"]))
    elif "anyOf" in out:
        if {"type": "null"} not in out["anyOf"]:
            out["anyOf"].append({"type": "null"})
    else:
        raise ToolSchemaCompatibilityError(f"{path}: nullable schema needs type or anyOf")
    if isinstance(out.get("enum"), list) and None not in out["enum"]:
        out["enum"].append(None)


def _property_mapping(value: Any, path: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ToolSchemaCompatibilityError(f"{path} must be an object")
    return {str(name): _plain_value(schema) for name, schema in value.items()}


def _required_names(value: Any, path: str) -> list[str]:
    if value is None:
        return []
    if not _is_string_sequence(value):
        raise ToolSchemaCompatibilityError(f"{path} must be an array of strings")
    return list(value)


def _format_required_group(names: set[str]) -> str:
    return "[" + (" + ".join(sorted(names)) if names else "no required fields") + "]"


def _join_description(existing: Any, *parts: str) -> str:
    values = [str(existing).strip()] if isinstance(existing, str) and existing.strip() else []
    values.extend(part.strip() for part in parts if part.strip())
    return " ".join(values)


def _require_mapping(value: Any, path: str) -> bool:
    if not isinstance(value, Mapping):
        raise ToolSchemaCompatibilityError(f"{path} must be an object schema")
    return True


def _is_string_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and all(
        isinstance(item, str) for item in value
    )


def _is_group_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and bool(value) and all(
        _is_string_sequence(group) and bool(group) for group in value
    )


def _plain_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _plain_value(item) for key, item in value.items()}


def _plain_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _plain_mapping(value)
    if isinstance(value, (list, tuple)):
        return [_plain_value(item) for item in value]
    return value
