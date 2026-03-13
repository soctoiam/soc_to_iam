import json
import os
import re
from collections import defaultdict

import pandas as pd

if "build_run_artifact_paths" not in globals():
    with open("tiir_process_log_utils.py", "r", encoding="utf-8") as _f:
        exec(_f.read(), globals())

WORKING_DIR = os.getcwd()

def get_accounts_csv_path():
    return globals().get("ACCOUNTS_CSV_OVERRIDE") or os.path.join(WORKING_DIR, "Accounts.CSV")

def get_permissions_csv_path():
    return globals().get("PERMISSIONS_CSV_OVERRIDE") or os.path.join(WORKING_DIR, "Permissions.CSV")

CPE_COLUMN_NAME = "Software_System"
PERM_LINK_COL = "Sorting"
ACC_LINK_COL = "SortingAttribute"

field_names = [
    "cpe_prefix", "cpe_version", "part", "vendor", "product", "version",
    "update", "edition", "language", "sw_edition", "target_sw", "target_hw", "other",
]

_MATCH_FIELD_WEIGHTS = {
    "part_exact": 0.05,
    "vendor": 0.20,
    "product_exact": 0.45,
    "version_exact": 0.25,
    "product_fuzzy_sticky": 0.20,
    "version_fuzzy_sticky": 0.10,
    "product_fuzzy_reverse": 0.15,
}

_RESOLUTION_PRIORITIES = {
    "direct_permission_assignment": 100,
    "owner_group_and_role": 90,
    "owner_group": 80,
    "owner_role": 70,
    "legacy_sorting_link": 40,
    "supervisor_fallback": 20,
}

def _is_missing(value):
    try:
        return pd.isna(value)
    except Exception:
        return value is None


def read_csv(file_path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(file_path, sep=";", engine="python")
    except Exception:
        return pd.read_csv(file_path, sep=",", engine="python")


def save_csv(df: pd.DataFrame, output_path: str):
    df.to_csv(output_path, sep=";", index=False)


def write_report(report_lines: list, output_path: str):
    with open(output_path, "w", encoding="utf-8") as report_file:
        report_file.writelines(line + "\n" for line in report_lines)


def split_cpe(cpe_string: str):
    if not isinstance(cpe_string, str) or not cpe_string.startswith("cpe:2.3"):
        return {}
    parts = cpe_string.split(":")
    parsed = {}
    for i, name in enumerate(field_names):
        parsed[name] = parts[i] if i < len(parts) else "*"
    return parsed


def _normalize_token(value):
    if _is_missing(value):
        return ""
    text = str(value).strip()
    if re.fullmatch(r"-?\d+\.0", text):
        text = text[:-2]
    return text

def parse_multivalue_field(value):
    if _is_missing(value):
        return []
    if isinstance(value, (list, tuple, set)):
        return [_normalize_token(v) for v in value if _normalize_token(v)]
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "[]"}:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            loaded = json.loads(text)
            if isinstance(loaded, list):
                return [_normalize_token(v) for v in loaded if _normalize_token(v)]
        except Exception:
            pass
    parts = re.split(r"[|,;]", text)
    return [_normalize_token(p) for p in parts if _normalize_token(p)]


def compare_cpe_parts(remote_cpe: dict, local_cpe: dict):
    matched_fields = []

    r_part = str(remote_cpe.get("part", "*")).strip().lower()
    l_part = str(local_cpe.get("part", "*")).strip().lower()
    if r_part not in {"", "*"} and l_part not in {"", "*"}:
        if r_part != l_part:
            return False, []
        matched_fields.append("part_exact")

    r_vendor = str(remote_cpe.get("vendor", "*")).strip().lower()
    l_vendor = str(local_cpe.get("vendor", "*")).strip().lower()
    if r_vendor != "*" and r_vendor != l_vendor:
        return False, []
    matched_fields.append("vendor")

    r_product = str(remote_cpe.get("product", "*")).strip().lower()
    l_product = str(local_cpe.get("product", "*")).strip().lower()
    r_version = str(remote_cpe.get("version", "*")).strip().lower()
    l_version = str(local_cpe.get("version", "*")).strip().lower()

    product_match = False
    if r_product == "*" or r_product == l_product:
        product_match = True
        matched_fields.append("product_exact")
        if r_version not in ["*", "-", ""]:
            if r_version != l_version:
                return False, []
            matched_fields.append("version_exact")
    elif l_product in r_product and l_version not in {"", "*"} and l_version in r_product:
        product_match = True
        matched_fields.extend(["product_fuzzy_sticky", "version_fuzzy_sticky"])
    elif r_product in l_product and r_version not in {"", "*"} and r_version in l_product:
        product_match = True
        matched_fields.append("product_fuzzy_reverse")

    return (product_match, matched_fields) if product_match else (False, [])


def compute_asset_match_specificity_score(matched_fields):
    score = 0.0
    for field in matched_fields or []:
        score += _MATCH_FIELD_WEIGHTS.get(field, 0.0)
    return round(min(score, 1.0), 4)


def row_matches_cpe(row, cpe_parts_remote):
    if CPE_COLUMN_NAME in row and isinstance(row[CPE_COLUMN_NAME], str):
        local_cpe_str = row[CPE_COLUMN_NAME]
        if local_cpe_str.startswith("cpe:"):
            local_cpe_dict = split_cpe(local_cpe_str)
            return compare_cpe_parts(cpe_parts_remote, local_cpe_dict)
    return False, []


def _collect_cpe_list(stix_data: dict):
    cpe_list = stix_data.get("x_detected_cpes", [])
    if not cpe_list and stix_data.get("cpe"):
        cpe_list = [{"cpe23": stix_data.get("cpe")}]
    return [c for c in cpe_list if c.get("cpe23") and c.get("cpe23") != "NOT_FOUND"]

def process_permissions(df: pd.DataFrame, stix_data: dict):
    report = []
    permission_hits = []

    cpe_list = _collect_cpe_list(stix_data)

    if "Temporal_Criticality" not in df.columns:
        df["Temporal_Criticality"] = ""
    if "Affected_Run_UUID" not in df.columns:
        df["Affected_Run_UUID"] = ""
    if "Asset_Match_Specificity" not in df.columns:
        df["Asset_Match_Specificity"] = ""

    for idx, row in df.iterrows():
        for cpe_obj in cpe_list:
            remote_cpe_str = cpe_obj.get("cpe23", "")
            remote_cpe_parts = split_cpe(remote_cpe_str)
            matched, matched_fields = row_matches_cpe(row, remote_cpe_parts)
            if not matched:
                continue

            old_crit = str(row.get("Criticality", "")).upper()
            new_crit = ""
            if old_crit == "MEDIUM":
                df.loc[idx, "Temporal_Criticality"] = "HIGH"
                new_crit = "HIGH"
            elif old_crit == "HIGH":
                df.loc[idx, "Temporal_Criticality"] = "VERY_HIGH"
                new_crit = "VERY_HIGH"
            else:
                new_crit = str(df.loc[idx, "Temporal_Criticality"])

            specificity = compute_asset_match_specificity_score(matched_fields)
            df.loc[idx, "Affected_Run_UUID"] = globals().get("CURRENT_TIIR_RUN_UUID", "")
            df.loc[idx, "Asset_Match_Specificity"] = specificity

            link_id = row.get(PERM_LINK_COL)
            hit = {
                "PermissionID": str(row.get("ID")),
                "LinkID": _normalize_token(link_id),
                "Entitlement": row.get("Entitlement"),
                "GovernedObject": row.get("Governed_Object", row.get("Entitlement")),
                "OwnerGroup": row.get("OwnerGroup", ""),
                "OwnerRole": row.get("OwnerRole", ""),
                "Matched_CPE": row.get(CPE_COLUMN_NAME),
                "Criticality_before": old_crit,
                "Criticality_after": new_crit,
                "MatchLogic": ", ".join(matched_fields),
                "AssetMatchSpecificity": specificity,
                "EvidenceCPE": remote_cpe_str,
            }
            permission_hits.append(hit)
            report.append(
                f"[PERMISSION] HIT on ID {row.get('ID')} | governed={row.get('Governed_Object', row.get('Entitlement'))} "
                f"| specificity={specificity} | logic={matched_fields}"
            )
            break

    permission_hits = sorted(
        permission_hits,
        key=lambda x: (-float(x["AssetMatchSpecificity"]), str(x["PermissionID"]))
    )
    return df, report, permission_hits


def _row_has_value(row, column_name):
    return column_name in row.index and not _is_missing(row.get(column_name))


def _build_account_hit(account_row, permission_hit, resolution_path, availability=None, contact_rank=1, action=None, fallback_reason=""):
    availability = availability or account_row.get("Availability", "")
    if not action:
        action = "notify_directly" if str(availability).lower() == "available" else "review_needed"

    return {
        "AccountID": str(account_row.get("AccountID")),
        "PermissionID": permission_hit.get("PermissionID"),
        "GovernedObject": permission_hit.get("GovernedObject"),
        "givenName": account_row.get("givenName"),
        "lastName": account_row.get("lastName"),
        "Team": account_row.get("Team"),
        "Function": account_row.get("Function"),
        "Availability": availability,
        "SupervisorAccountID": _normalize_token(account_row.get("SupervisorAccountID")),
        "ResolutionPath": resolution_path,
        "ContactRank": contact_rank,
        "Action": action,
        "OwnerGroup": permission_hit.get("OwnerGroup", ""),
        "OwnerRole": permission_hit.get("OwnerRole", ""),
        "AssetMatchSpecificity": permission_hit.get("AssetMatchSpecificity"),
        "FallbackReason": fallback_reason,
    }


def _resolve_accounts_for_permission(accounts_df: pd.DataFrame, permission_hit: dict):
    permission_id = _normalize_token(permission_hit.get("PermissionID"))
    owner_group = _normalize_token(permission_hit.get("OwnerGroup") or "")
    owner_role = _normalize_token(permission_hit.get("OwnerRole") or "")
    link_id = _normalize_token(permission_hit.get("LinkID") or "")

    candidates = []

    for _, row in accounts_df.iterrows():
        account_paths = []

        assigned_permissions = set(parse_multivalue_field(row.get("AssignedPermissions", "")))
        owned_groups = set(parse_multivalue_field(row.get("OwnedGroups", "")))
        owned_roles = set(parse_multivalue_field(row.get("OwnedRoles", "")))
        sorting_attribute = _normalize_token(row.get(ACC_LINK_COL))

        if permission_id in assigned_permissions:
            account_paths.append("direct_permission_assignment")
        if owner_group and owner_role:
            if owner_group in owned_groups and owner_role in owned_roles:
                account_paths.append("owner_group_and_role")
        else:
            if owner_group and owner_group in owned_groups:
                account_paths.append("owner_group")
            if owner_role and owner_role in owned_roles:
                account_paths.append("owner_role")
        if link_id and sorting_attribute and sorting_attribute == link_id:
            account_paths.append("legacy_sorting_link")

        account_paths = sorted(set(account_paths), key=lambda p: -_RESOLUTION_PRIORITIES[p])
        for path in account_paths:
            candidates.append(
                _build_account_hit(
                    row,
                    permission_hit,
                    resolution_path=path,
                    availability=str(row.get("Availability", "") or ""),
                    contact_rank=1 if str(row.get("Availability", "")).lower() == "available" else 2,
                    action="notify_directly" if str(row.get("Availability", "")).lower() == "available" else "review_needed",
                )
            )

    if candidates:
        primary_available = [c for c in candidates if str(c.get("Availability", "")).lower() == "available"]
        if not primary_available:
            fallback_hits = []
            supervisor_ids = {
                _normalize_token(c["SupervisorAccountID"])
                for c in candidates
                if _normalize_token(c.get("SupervisorAccountID"))
            }
            if supervisor_ids:
                sup_rows = accounts_df[accounts_df["AccountID"].astype(str).isin(supervisor_ids)]
                for _, sup in sup_rows.iterrows():
                    fallback_hits.append(
                        _build_account_hit(
                            sup,
                            permission_hit,
                            resolution_path="supervisor_fallback",
                            availability=str(sup.get("Availability", "") or ""),
                            contact_rank=2,
                            action="notify_fallback",
                            fallback_reason="primary_candidates_unavailable",
                        )
                    )
                candidates.extend(fallback_hits)

    deduped = {}
    for hit in candidates:
        key = (hit["AccountID"], hit["PermissionID"])
        existing = deduped.get(key)
        candidate_score = (_RESOLUTION_PRIORITIES.get(hit["ResolutionPath"], 0), -int(hit["ContactRank"]))
        existing_score = (_RESOLUTION_PRIORITIES.get(existing["ResolutionPath"], 0), -int(existing["ContactRank"])) if existing else None
        if existing is None or candidate_score > existing_score:
            deduped[key] = hit
    return list(deduped.values())


def process_accounts(df: pd.DataFrame, permission_hits: list):
    report = []
    account_hits = []

    if "Review_Action" not in df.columns:
        df["Review_Action"] = ""
    if "Resolution_Path" not in df.columns:
        df["Resolution_Path"] = ""
    if "Affected_Run_UUID" not in df.columns:
        df["Affected_Run_UUID"] = ""

    for permission_hit in permission_hits:
        resolved = _resolve_accounts_for_permission(df, permission_hit)
        for hit in resolved:
            account_id = str(hit["AccountID"])
            idx_matches = df.index[df["AccountID"].astype(str) == account_id].tolist()
            if idx_matches:
                idx = idx_matches[0]
                df.loc[idx, "Review_Action"] = hit["Action"]
                df.loc[idx, "Resolution_Path"] = hit["ResolutionPath"]
                df.loc[idx, "Affected_Run_UUID"] = globals().get("CURRENT_TIIR_RUN_UUID", "")
            account_hits.append(hit)
            report.append(
                f"[ACCOUNT] HIT on ID {account_id} | permission={hit['PermissionID']} "
                f"| path={hit['ResolutionPath']} | action={hit['Action']} | availability={hit['Availability']}"
            )

    account_hits = sorted(
        account_hits,
        key=lambda x: (int(x["ContactRank"]), -_RESOLUTION_PRIORITIES.get(x["ResolutionPath"], 0), x["AccountID"], x["PermissionID"])
    )
    return df, report, account_hits


def startLoader(return_results=True, quiet=False, artifact_paths=None, stix_path=None):
    artifact_paths = artifact_paths or build_run_artifact_paths(globals().get("CURRENT_TIIR_RUN_UUID", "manual"))
    stix_path = stix_path or artifact_paths["stix_object"]
    output_report_path = artifact_paths["report"]
    accounts_modified_path = artifact_paths["accounts_modified"]
    permissions_modified_path = artifact_paths["permissions_modified"]

    if not quiet:
        print("Starting Loader (advanced IAM resolution)...")

    accounts_csv_path = get_accounts_csv_path()
    permissions_csv_path = get_permissions_csv_path()

    for path in [accounts_csv_path, permissions_csv_path, stix_path]:
        if not os.path.exists(path):
            tiir_logger.log("loader", "missing_input", {"path": path}, status="ERROR")
            raise FileNotFoundError(path)

    accounts_df = read_csv(accounts_csv_path)
    permissions_df = read_csv(permissions_csv_path)
    with open(stix_path, "r", encoding="utf-8") as f:
        stix_data = json.load(f)
    if "x_detected_cpes" not in stix_data and "cpe" not in stix_data:
        raise RuntimeError("STIX object contains neither x_detected_cpes nor cpe")

    permissions_df, perm_report, permission_hits = process_permissions(permissions_df, stix_data)
    accounts_df, acc_report, account_hits = process_accounts(accounts_df, permission_hits)

    save_csv(accounts_df, accounts_modified_path)
    save_csv(permissions_df, permissions_modified_path)

    all_reports = [
        f"Run UUID: {artifact_paths.get('run_uuid')}",
        f"Accounts source: {os.path.basename(accounts_csv_path)}",
        f"Permissions source: {os.path.basename(permissions_csv_path)}",
        f"Threat Description: {stix_data.get('description', '')[:180]}...",
    ]
    all_reports.extend(perm_report)
    all_reports.extend(acc_report)
    if len(all_reports) == 4:
        all_reports.append("No matches found.")
    write_report(all_reports, output_report_path)

    available_hits = [h for h in account_hits if str(h.get("Availability", "")).lower() == "available"]
    fallback_hits = [h for h in account_hits if h.get("ResolutionPath") == "supervisor_fallback"]

    result = {
        "run_uuid": artifact_paths.get("run_uuid"),
        "permission_hits": permission_hits,
        "account_hits": account_hits,
        "counts": {
            "permission_hits": len(permission_hits),
            "account_hits": len(account_hits),
            "available_contacts": len(available_hits),
            "fallback_contacts": len(fallback_hits),
        },
        "files": {
            "report": output_report_path,
            "accounts_modified": accounts_modified_path,
            "permissions_modified": permissions_modified_path,
            "tiir_process_log": artifact_paths["tiir_process_log"],
            "stix_object": stix_path,
        },
        "resolution_model": {
            "accounts_csv": os.path.basename(accounts_csv_path),
            "permissions_csv": os.path.basename(permissions_csv_path),
            "supports_assigned_permissions": "AssignedPermissions" in accounts_df.columns,
            "supports_owner_groups": "OwnerGroup" in permissions_df.columns and "OwnedGroups" in accounts_df.columns,
            "supports_owner_roles": "OwnerRole" in permissions_df.columns and "OwnedRoles" in accounts_df.columns,
            "supports_supervisor_fallback": "SupervisorAccountID" in accounts_df.columns,
            "legacy_sorting_fallback": PERM_LINK_COL in permissions_df.columns and ACC_LINK_COL in accounts_df.columns,
        },
    }
    tiir_logger.log("loader", "mapping_completed", result)

    if not quiet:
        print(f"Permission hits: {result['counts']['permission_hits']}")
        print(f"Linked identities: {result['counts']['account_hits']}")

    return result if return_results else None
