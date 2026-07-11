"""JSON-schema scorer for structured-extraction tasks (PRD FR-3.2).

Scoring ladder:
  invalid / unparseable ............ 0.0, failed
  schema-valid, gold mismatch ...... 0.5, failed  (validity alone is half credit)
  schema-valid + gold match ........ 1.0, passed
  schema-valid, no comparable gold . 1.0, passed  (detail flags "no_gold")

Gold comparison is structural equality on the schema's *required* fields only —
optional fields may legitimately vary.
"""

from __future__ import annotations

import json
from typing import Any

import jsonschema

from reaplab.core.providers import extract_json
from reaplab.core.records import EvalRecord


class JsonSchemaScorer:
    name = "json_schema"

    def score(self, item: EvalRecord, response: Any) -> tuple[float, bool, dict[str, Any]]:
        if not item.json_schema:
            return 0.0, False, {
                "error": f"eval item {item.id!r} has task_type=json_schema but no json_schema; "
                "fix the eval dataset (add the schema the response must satisfy)."
            }
        try:
            obj = extract_json(response.text)
        except ValueError:
            return 0.0, False, {"schema_valid": False, "error": "no JSON found in response"}
        try:
            jsonschema.validate(obj, item.json_schema)
        except jsonschema.ValidationError as e:
            return 0.0, False, {"schema_valid": False, "error": str(e).splitlines()[0][:300]}
        except jsonschema.SchemaError as e:
            return 0.0, False, {
                "error": f"item {item.id!r} carries an invalid JSON schema: {e}".replace("\n", " ")[:300]
            }

        gold_obj: Any = None
        if item.gold is not None:
            try:
                gold_obj = json.loads(item.gold)
            except json.JSONDecodeError:
                gold_obj = None
        if gold_obj is None:
            return 1.0, True, {"schema_valid": True, "no_gold": True}

        required = list(item.json_schema.get("required", []))
        if not required and isinstance(gold_obj, dict):
            required = list(gold_obj.keys())
        mismatched = [
            k for k in required
            if not isinstance(obj, dict) or not isinstance(gold_obj, dict) or obj.get(k) != gold_obj.get(k)
        ]
        if not mismatched:
            return 1.0, True, {"schema_valid": True, "gold_match": True}
        return 0.5, False, {"schema_valid": True, "gold_match": False, "mismatched_fields": mismatched}
