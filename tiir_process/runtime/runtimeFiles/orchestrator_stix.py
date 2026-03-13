
import datetime
import hashlib
import json
import os
import uuid

if "deterministic_uuid_from_text" not in globals():
    with open("tiir_process_log_utils.py", "r", encoding="utf-8") as _f:
        exec(_f.read(), globals())


def tiir_build_cti_object(description, detected_cpes, route_context, output_stix_file="Test_STIX.json"):
    description = (description or "Auto-generated report.").strip()
    detected_cpes = list(detected_cpes or [])

    deterministic_id = route_context.get("run_uuid") or deterministic_uuid_from_text(description)

    primary_cpe = "cpe:2.3:*:*:*:*:*:*:*:*:*:*:*"
    for entry in detected_cpes:
        cpe = entry.get("cpe23")
        if cpe and cpe != "NOT_FOUND":
            primary_cpe = cpe
            break

    stix_output = {
        "type": "CTI Object",
        "id": deterministic_id,
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "name": "Automated Threat Intel Report",
        "description": description,
        "x_detected_cpes": detected_cpes,
        "cpe": primary_cpe,
        "x_run_uuid": deterministic_id,
        "x_input_kind": route_context.get("input_kind"),
        "x_route_mode": route_context.get("route"),
        "x_route_reason": route_context.get("route_reason"),
        "x_source_name": route_context.get("source_name"),
        "x_input_sha256": route_context.get("input_sha256"),
    }

    with open(output_stix_file, "w", encoding="utf-8") as f:
        json.dump(stix_output, f, indent=2, ensure_ascii=False)

    grounded = [c for c in detected_cpes if c.get("grounding_status") in {"GROUNDED", "PREBUILT"} and c.get("cpe23") != "NOT_FOUND"]
    result = {
        "output_stix_file": output_stix_file,
        "cti_id": deterministic_id,
        "grounded_or_prebuilt_count": len(grounded),
        "primary_cpe": primary_cpe,
    }
    tiir_logger.log("orchestrator", "cti_object_written", result)
    return stix_output
