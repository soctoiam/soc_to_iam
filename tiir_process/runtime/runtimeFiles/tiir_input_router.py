
import json
import os
import re
from collections import OrderedDict

if "deterministic_uuid_from_text" not in globals():
    with open("tiir_process_log_utils.py", "r", encoding="utf-8") as _f:
        exec(_f.read(), globals())

_CPE_REGEX = re.compile(r'cpe:2\.3:[aho\*\-]:[^\s,"\]\}\)]+', re.IGNORECASE)
_TEXT_FIELDS = [
    "description", "content", "body", "summary", "full_text", "message",
    "details", "title", "name", "value"
]


def _dedupe_cpes(items):
    seen = OrderedDict()
    for item in items:
        cpe = str(item.get("cpe23", "")).strip().lower()
        if not cpe:
            continue
        if cpe not in seen:
            normalized = dict(item)
            normalized["cpe23"] = cpe
            seen[cpe] = normalized
    return list(seen.values())


def _text_preview(text, width=140):
    text = " ".join(str(text).split())
    return text[:width] + ("..." if len(text) > width else "")


def _walk_strings(obj, path="root"):
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield from _walk_strings(value, f"{path}.{key}")
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            yield from _walk_strings(value, f"{path}[{idx}]")
    elif isinstance(obj, str):
        yield path, obj


def _extract_direct_cpes_from_json(data):
    hits = []

    def scan(obj, path="root"):
        if isinstance(obj, dict):
            for key, value in obj.items():
                key_l = str(key).lower()
                if key_l in {"cpe", "cpe23", "x_cpe23"}:
                    if isinstance(value, str):
                        hits.append({
                            "cpe23": value,
                            "grounding_status": "PREBUILT",
                            "match_score": 1.0,
                            "evidence_source": path + "." + str(key),
                        })
                    elif isinstance(value, list):
                        for idx, item in enumerate(value):
                            if isinstance(item, str):
                                hits.append({
                                    "cpe23": item,
                                    "grounding_status": "PREBUILT",
                                    "match_score": 1.0,
                                    "evidence_source": f"{path}.{key}[{idx}]",
                                })
                elif key_l == "x_detected_cpes" and isinstance(value, list):
                    for idx, item in enumerate(value):
                        if isinstance(item, dict) and item.get("cpe23"):
                            hits.append({
                                "cpe23": item.get("cpe23"),
                                "grounding_status": item.get("grounding_status", "PREBUILT"),
                                "match_score": float(item.get("match_score", 1.0)),
                                "evidence_source": f"{path}.{key}[{idx}].cpe23",
                            })
                scan(value, f"{path}.{key}")
        elif isinstance(obj, list):
            for idx, item in enumerate(obj):
                scan(item, f"{path}[{idx}]")

    scan(data)

    for path, text in _walk_strings(data):
        for match in _CPE_REGEX.findall(text):
            hits.append({
                "cpe23": match,
                "grounding_status": "PREBUILT",
                "match_score": 1.0,
                "evidence_source": f"{path}:regex",
            })

    return _dedupe_cpes(hits)


def _extract_description_from_json(data):
    blocks = []

    def add_block(label, value):
        value = str(value).strip()
        if value:
            blocks.append(f"{label}: {value}")

    if isinstance(data, dict):
        for field in _TEXT_FIELDS:
            if isinstance(data.get(field), str) and data.get(field).strip():
                add_block(field.upper(), data.get(field))

        if "containers" in data:
            try:
                descriptions = data["containers"]["cna"]["descriptions"]
                for item in descriptions:
                    if isinstance(item, dict) and item.get("value"):
                        add_block("DESCRIPTION", item["value"])
            except Exception:
                pass

            try:
                affected = data["containers"]["cna"]["affected"]
                for item in affected:
                    if not isinstance(item, dict):
                        continue
                    vendor = item.get("vendor", "")
                    product = item.get("product", "")
                    version_lines = []
                    versions = item.get("versions", [])
                    if isinstance(versions, list):
                        for version_item in versions[:8]:
                            if isinstance(version_item, dict):
                                version_lines.append(
                                    f"{version_item.get('version', '')} ({version_item.get('status', '')})"
                                )
                    affected_line = f"{vendor} {product}".strip()
                    if version_lines:
                        affected_line += " | versions: " + "; ".join(version_lines)
                    add_block("AFFECTED_PRODUCT", affected_line)
            except Exception:
                pass

        if data.get("type") == "bundle" and isinstance(data.get("objects"), list):
            for obj in data["objects"]:
                if not isinstance(obj, dict):
                    continue
                if obj.get("type") in {"vulnerability", "attack-pattern", "report"}:
                    if obj.get("name"):
                        add_block("NAME", obj["name"])
                    if obj.get("description"):
                        add_block("DESCRIPTION", obj["description"])

    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict):
            for field in _TEXT_FIELDS:
                if isinstance(first.get(field), str) and first.get(field).strip():
                    add_block(field.upper(), first.get(field))

    deduped_blocks = []
    seen = set()
    for block in blocks:
        if block not in seen:
            deduped_blocks.append(block)
            seen.add(block)
    return "\n\n".join(deduped_blocks).strip()


def tiir_route_input(input_payload):
    if input_payload is None:
        raise ValueError("input_payload is None")

    route_context = {
        "run_uuid": None,
        "input_sha256": None,
        "input_kind": None,
        "source_name": None,
        "route": None,
        "route_reason": None,
        "normalized_description": "",
        "prebuilt_cpes": [],
        "input_preview": "",
        "json_summary": {},
    }

    if isinstance(input_payload, str) and input_payload.strip().endswith(".json") and os.path.exists(input_payload.strip()):
        json_path = input_payload.strip()
        route_context["input_kind"] = "json_file"
        route_context["source_name"] = os.path.basename(json_path)

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        canonical = json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        route_context["run_uuid"] = deterministic_uuid_from_text(canonical)
        route_context["input_sha256"] = __import__("hashlib").sha256(canonical.encode("utf-8")).hexdigest()

        prebuilt_cpes = _extract_direct_cpes_from_json(data)
        normalized_description = _extract_description_from_json(data)

        route_context["prebuilt_cpes"] = prebuilt_cpes
        route_context["normalized_description"] = normalized_description
        route_context["input_preview"] = _text_preview(normalized_description or json_path)
        route_context["json_summary"] = {
            "root_type": data.get("type") if isinstance(data, dict) else type(data).__name__,
            "prebuilt_cpe_count": len(prebuilt_cpes),
            "description_chars": len(normalized_description),
        }

        if prebuilt_cpes:
            route_context["route"] = "direct_cpe_path"
            route_context["route_reason"] = "direct_cpe_found_in_json"
        elif normalized_description:
            route_context["route"] = "inference_path"
            route_context["route_reason"] = "json_without_cpe_but_with_extractable_text"
        else:
            route_context["route"] = "error_no_text"
            route_context["route_reason"] = "json_without_cpe_and_without_extractable_text"

    else:
        text = str(input_payload).strip()
        route_context["input_kind"] = "raw_text"
        route_context["source_name"] = "inline_text"
        route_context["normalized_description"] = text
        route_context["input_preview"] = _text_preview(text)
        route_context["run_uuid"] = deterministic_uuid_from_text(text)
        route_context["input_sha256"] = __import__("hashlib").sha256(text.encode("utf-8")).hexdigest()

        regex_hits = [
            {
                "cpe23": match.lower(),
                "grounding_status": "PREBUILT",
                "match_score": 1.0,
                "evidence_source": "inline_text:regex",
            }
            for match in _CPE_REGEX.findall(text)
        ]
        route_context["prebuilt_cpes"] = _dedupe_cpes(regex_hits)

        if route_context["prebuilt_cpes"]:
            route_context["route"] = "direct_cpe_path"
            route_context["route_reason"] = "direct_cpe_found_in_text"
        elif text:
            route_context["route"] = "inference_path"
            route_context["route_reason"] = "text_without_cpe_requires_inference"
        else:
            route_context["route"] = "error_no_text"
            route_context["route_reason"] = "empty_text_input"

    return route_context
