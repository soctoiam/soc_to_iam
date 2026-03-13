import os
import pandas as pd

_REQUIRED_HELPERS = [
    ("tiir_process_log_utils.py", "build_run_artifact_paths"),
    ("tiir_input_router.py", "tiir_route_input"),
    ("text2CPE_inference.py", "tiir_run_text2cpe_inference"),
    ("orchestrator_stix.py", "tiir_build_cti_object"),
    ("Loader.py", "startLoader"),
]

for _path, _symbol in _REQUIRED_HELPERS:
    if _symbol not in globals():
        with open(_path, "r", encoding="utf-8") as _f:
            exec(_f.read(), globals())

missing_symbols = [sym for _, sym in _REQUIRED_HELPERS if sym not in globals()]
if missing_symbols:
    raise RuntimeError(f"Missing helper symbols after loading: {missing_symbols}")

def _render_dataframe(df, title, columns):
    print(f"\n{title}")
    if df.empty:
        print("   No entries")
        return
    view = df[[c for c in columns if c in df.columns]].reset_index(drop=True)
    print(f"   Rows: {len(view)}")
    try:
        from IPython.display import HTML, display
        display(HTML(view.to_html(index=False, escape=False)))
    except Exception:
        with pd.option_context("display.max_colwidth", 64, "display.width", 220):
            print(view.to_string(index=False))


def _print_input_block(route_context, artifact_paths):
    print("TIIR PIPELINE REVIEW SUMMARY")
    print("=" * 88)
    print("A. INPUT")
    print(f"   Run UUID: {route_context.get('run_uuid')}")
    print(f"   Input kind: {route_context.get('input_kind')}")
    print(f"   Source: {route_context.get('source_name')}")
    preview = route_context.get("input_preview", "")
    if preview:
        print(f"   Preview: {preview}")
    if route_context.get("input_kind") == "json_file":
        summary = route_context.get("json_summary", {})
        print(f"   JSON root type: {summary.get('root_type')}")
        print(f"   JSON prebuilt CPEs: {summary.get('prebuilt_cpe_count')}")
        print(f"   Extracted text chars: {summary.get('description_chars')}")
    print(f"   Determinism log: {os.path.basename(artifact_paths['tiir_process_log'])}")


def _print_resolution_block(route_context, inference_result=None):
    print("\nB. CPE RESOLUTION")
    print(f"   Route: {route_context.get('route')}")
    print(f"   Reason: {route_context.get('route_reason')}")

    if route_context.get("route") == "direct_cpe_path":
        prebuilt = route_context.get("prebuilt_cpes", [])
        print(f"   Inference skipped. Direct CPE evidence found: {len(prebuilt)}")
        for idx, comp in enumerate(prebuilt, start=1):
            print(f"   [{idx}] {comp.get('cpe23')} | {comp.get('grounding_status')} | source={comp.get('evidence_source')}")
    elif inference_result is not None:
        summary = inference_result["pipeline_summary"]
        print(
            f"   Components: {summary['components_total']} | grounded: {summary['grounded']} | "
            f"rejected: {summary['rejected']}"
        )
        print(f"   Generation time: {summary['generation_time_s']} s")
        for idx, comp in enumerate(inference_result["final_cpe_results"], start=1):
            print(
                f"   [{idx}] {comp.get('vendor')}/{comp.get('product')} -> {comp.get('cpe23')} | "
                f"grounding_score={comp.get('match_score')} | {comp.get('grounding_status')}"
            )
    else:
        print("   No resolvable text available. Pipeline cannot continue.")


def _print_orchestrator_block(stix_output, route_context):
    print("\nC. ORCHESTRATOR / CTI OBJECT")
    print(f"   CTI object written: {os.path.basename(globals().get('CURRENT_ARTIFACT_PATHS', {}).get('stix_object', 'Test_STIX.json'))}")
    print(f"   Route mode: {route_context.get('route')}")
    print(f"   Primary CPE: {stix_output.get('cpe')}")
    print(f"   Detected CPE entries: {len(stix_output.get('x_detected_cpes', []))}")


def _print_permissions_block(loader_result):
    columns = [
        "PermissionID", "GovernedObject", "Entitlement", "OwnerGroup", "OwnerRole",
        "Criticality_before", "Criticality_after", "AssetMatchSpecificity", "MatchLogic"
    ]
    perm_df = pd.DataFrame(loader_result["permission_hits"]) if loader_result.get("permission_hits") else pd.DataFrame(columns=columns)
    _render_dataframe(perm_df, "D. IMPACTED PERMISSIONS", columns)


def _print_accounts_block(loader_result):
    columns = [
        "AccountID", "PermissionID", "GovernedObject", "givenName", "lastName", "Team", "Function",
        "Availability", "ResolutionPath", "ContactRank", "SupervisorAccountID", "Action"
    ]
    acc_df = pd.DataFrame(loader_result["account_hits"]) if loader_result.get("account_hits") else pd.DataFrame(columns=columns)
    _render_dataframe(acc_df, "E. IMPACTED IDENTITIES", columns)


def _print_outputs_block(loader_result, artifact_paths):
    print("\nF. OUTPUT FILES")
    print(f"   STIX object: {os.path.basename(loader_result['files']['stix_object'])}")
    print(f"   Permission updates: {os.path.basename(loader_result['files']['permissions_modified'])}")
    print(f"   Account updates: {os.path.basename(loader_result['files']['accounts_modified'])}")
    print(f"   Modification report: {os.path.basename(loader_result['files']['report'])}")
    print(f"   Deterministic process log: {os.path.basename(loader_result['files']['tiir_process_log'])}")
    model = loader_result.get("resolution_model", {})
    print(
        "   Resolution model: "
        f"assigned_permissions={model.get('supports_assigned_permissions')} | "
        f"owner_groups={model.get('supports_owner_groups')} | "
        f"owner_roles={model.get('supports_owner_roles')} | "
        f"supervisor_fallback={model.get('supports_supervisor_fallback')} | "
        f"legacy_sorting_fallback={model.get('legacy_sorting_fallback')}"
    )


def tiir_run_pipeline(input_payload=None, *, verbose=False, show_audit_json=False):
    global tiir_logger, CURRENT_TIIR_RUN_UUID, CURRENT_ARTIFACT_PATHS

    if input_payload is None:
        input_payload = globals().get("input_payload", globals().get("input_text"))
    if input_payload is None:
        raise ValueError("No input_payload or input_text defined.")

    route_context = tiir_route_input(input_payload)
    CURRENT_TIIR_RUN_UUID = route_context["run_uuid"]
    CURRENT_ARTIFACT_PATHS = build_run_artifact_paths(CURRENT_TIIR_RUN_UUID)
    tiir_logger = TIIRProcessLogger(CURRENT_ARTIFACT_PATHS["tiir_process_log"])
    tiir_logger.reset({
        "component": "tiir_pipeline_runner",
        "run_uuid": CURRENT_TIIR_RUN_UUID,
        "input_kind": route_context.get("input_kind"),
        "source_name": route_context.get("source_name"),
        "route": route_context.get("route"),
        "route_reason": route_context.get("route_reason"),
        "input_sha256": route_context.get("input_sha256"),
    })
    tiir_logger.log("router", "input_routed", route_context)

    _print_input_block(route_context, CURRENT_ARTIFACT_PATHS)

    if route_context["route"] == "error_no_text":
        _print_resolution_block(route_context, inference_result=None)
        tiir_logger.log("pipeline", "aborted_no_text", route_context, status="ERROR")
        return {"route_context": route_context, "status": "aborted_no_text", "artifact_paths": CURRENT_ARTIFACT_PATHS}

    inference_result = None
    final_cpe_results = route_context.get("prebuilt_cpes", [])

    if "tiir_run_text2cpe_inference" not in globals():
        raise RuntimeError("Inference helper not loaded: tiir_run_text2cpe_inference")

    if route_context["route"] == "inference_path":
        inference_result = tiir_run_text2cpe_inference(
            route_context["normalized_description"],
            verbose=verbose,
            show_audit_json=show_audit_json,
        )
        final_cpe_results = inference_result["final_cpe_results"]

    _print_resolution_block(route_context, inference_result=inference_result)

    stix_output = tiir_build_cti_object(
        description=route_context.get("normalized_description", ""),
        detected_cpes=final_cpe_results,
        route_context=route_context,
        output_stix_file=CURRENT_ARTIFACT_PATHS["stix_object"],
    )
    _print_orchestrator_block(stix_output, route_context)

    loader_result = startLoader(
        return_results=True,
        quiet=True,
        artifact_paths=CURRENT_ARTIFACT_PATHS,
        stix_path=CURRENT_ARTIFACT_PATHS["stix_object"],
    )
    _print_permissions_block(loader_result)
    _print_accounts_block(loader_result)
    _print_outputs_block(loader_result, CURRENT_ARTIFACT_PATHS)

    result = {
        "status": "completed",
        "route_context": route_context,
        "inference_result": inference_result,
        "stix_output": stix_output,
        "loader_result": loader_result,
        "artifact_paths": CURRENT_ARTIFACT_PATHS,
    }
    tiir_logger.log("pipeline", "completed", {
        "status": result["status"],
        "run_uuid": CURRENT_TIIR_RUN_UUID,
        "route": route_context.get("route"),
        "permission_hits": loader_result["counts"]["permission_hits"],
        "account_hits": loader_result["counts"]["account_hits"],
        "available_contacts": loader_result["counts"]["available_contacts"],
        "fallback_contacts": loader_result["counts"]["fallback_contacts"],
        "files": loader_result["files"],
    })
    return result


if globals().get("AUTO_RUN_TIIR_PIPELINE", True):
    _tiir_result = tiir_run_pipeline(
        input_payload=globals().get("input_payload", globals().get("input_text")),
        verbose=bool(globals().get("PIPELINE_VERBOSE", False)),
        show_audit_json=bool(globals().get("SHOW_AUDIT_JSON", False)),
    )
