"""Tool-call validity scorer for agentic traces (PRD FR-3.2, min_tool_call_validity gate).

Scoring:
  1.0 / passed .... a call names the expected tool (or any listed tool when no
                    expected_tool is set) with schema-valid arguments
  0.5 / failed .... schema-valid call to the WRONG tool (validity survives; the
                    tool_call_validity gate still counts it via detail["schema_valid"])
  0.0 / failed .... no call, unknown tool, or arguments that fail the tool's schema

detail always carries {"schema_valid": bool} — evaluate.py aggregates that flag into
the summary's tool_call_validity, independent of pass/fail.
"""

from __future__ import annotations

import json
from typing import Any

import jsonschema

from reaplab.core.providers import extract_json
from reaplab.core.records import EvalRecord

# ---------------------------------------------------------------------------
# helpers shared with MockRunner (which synthesizes valid calls from the same defs)
# ---------------------------------------------------------------------------


def tool_name(tool: dict[str, Any]) -> str | None:
    """Name of one tool definition; accepts OpenAI {"type","function":{...}} or flat dicts."""
    fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
    name = fn.get("name")
    return name if isinstance(name, str) else None


def tool_parameters(tool: dict[str, Any]) -> dict[str, Any]:
    """JSON schema of one tool's arguments (empty object schema when unspecified)."""
    fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
    params = fn.get("parameters") or fn.get("input_schema")
    return params if isinstance(params, dict) else {"type": "object"}


def find_tool(tools: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    for t in tools:
        if tool_name(t) == name:
            return t
    return None


def synth_args(schema: dict[str, Any]) -> Any:
    """Deterministically synthesize a value satisfying a (simple) JSON schema.

    Covers the subset domain packs realistically use: object/properties/required,
    string (+enum), number, integer, boolean, array. Used by MockRunner to emit
    schema-valid tool calls without a real model.
    """
    if "enum" in schema and schema["enum"]:
        return schema["enum"][0]
    t = schema.get("type")
    if isinstance(t, list):
        t = t[0] if t else None
    if t == "object" or (t is None and "properties" in schema):
        props: dict[str, Any] = schema.get("properties") or {}
        required = schema.get("required") or list(props.keys())
        return {k: synth_args(props.get(k, {})) for k in required}
    if t == "string":
        return "sample"
    if t == "number":
        return 1.0
    if t == "integer":
        return 1
    if t == "boolean":
        return True
    if t == "array":
        items = schema.get("items")
        n = int(schema.get("minItems") or 1)
        return [synth_args(items)] * n if isinstance(items, dict) else []
    if t == "null":
        return None
    return "sample"


def normalize_tool_calls(response: Any) -> list[dict[str, Any]]:
    """Extract calls as [{"name": str, "arguments": Any}] from a RunnerResponse.

    Prefers response.tool_calls (OpenAI shape, arguments possibly a JSON string);
    falls back to parsing the text for {"name": ..., "arguments": {...}} (single
    call or a list of them). Unparseable entries are dropped.
    """
    out: list[dict[str, Any]] = []
    raw_calls = getattr(response, "tool_calls", None)
    if raw_calls:
        for c in raw_calls:
            fn = c.get("function") if isinstance(c.get("function"), dict) else c
            name = fn.get("name")
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = None
            if isinstance(name, str):
                out.append({"name": name, "arguments": args})
        return out
    try:
        obj = extract_json(getattr(response, "text", "") or "")
    except ValueError:
        return []
    candidates = obj if isinstance(obj, list) else [obj]
    for c in candidates:
        if isinstance(c, dict) and isinstance(c.get("name"), str):
            out.append({"name": c["name"], "arguments": c.get("arguments")})
    return out


# ---------------------------------------------------------------------------
# scorer
# ---------------------------------------------------------------------------


class ToolCallScorer:
    name = "tool_call"

    def score(self, item: EvalRecord, response: Any) -> tuple[float, bool, dict[str, Any]]:
        if not item.tools:
            return 0.0, False, {
                "error": f"eval item {item.id!r} has task_type=tool_call but no tools; "
                "fix the eval dataset (attach the OpenAI-format tool definitions)."
            }
        calls = normalize_tool_calls(response)
        if not calls:
            return 0.0, False, {"schema_valid": False, "error": "no tool call found in response"}

        best: tuple[float, bool, dict[str, Any]] = (
            0.0, False, {"schema_valid": False, "called_tool": calls[0]["name"], "error": "unknown tool"}
        )
        for call in calls:
            tool = find_tool(item.tools, call["name"])
            if tool is None:
                continue
            args = call["arguments"] if call["arguments"] is not None else {}
            try:
                jsonschema.validate(args, tool_parameters(tool))
                valid = True
                err = None
            except jsonschema.ValidationError as e:
                valid = False
                err = str(e).splitlines()[0][:300]
            except jsonschema.SchemaError as e:
                valid = False
                err = f"tool {call['name']!r} has an invalid parameter schema: {e}".replace("\n", " ")[:300]
            correct_tool = item.expected_tool is None or call["name"] == item.expected_tool
            detail: dict[str, Any] = {"schema_valid": valid, "called_tool": call["name"],
                                      "expected_tool": item.expected_tool}
            if err:
                detail["error"] = err
            if valid and correct_tool:
                return 1.0, True, detail
            candidate = (0.5, False, detail) if valid else (0.0, False, detail)
            if candidate[0] > best[0]:
                best = candidate
        return best
