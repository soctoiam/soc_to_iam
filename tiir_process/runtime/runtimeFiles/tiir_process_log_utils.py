
import datetime
import hashlib
import json
import os
from typing import Any


def _json_safe(value: Any):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    try:
        import pandas as pd  # type: ignore
        if isinstance(value, pd.DataFrame):
            return value.to_dict(orient="records")
        if isinstance(value, pd.Series):
            return value.to_dict()
    except Exception:
        pass
    return str(value)


def file_sha256(path: str):
    if not path or not os.path.exists(path) or not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def deterministic_uuid_from_text(text: str) -> str:
    normalized = str(text or "").strip()
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"{digest[:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-{digest[20:32]}"


def deterministic_uuid_from_json_obj(obj: Any) -> str:
    canonical = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return deterministic_uuid_from_text(canonical)


def build_run_artifact_paths(run_uuid: str, working_dir: str | None = None) -> dict:
    wd = working_dir or os.getcwd()
    return {
        "run_uuid": run_uuid,
        "stix_object": os.path.join(wd, f"Test_STIX_{run_uuid}.json"),
        "report": os.path.join(wd, f"modification_report_{run_uuid}.txt"),
        "accounts_modified": os.path.join(wd, f"accounts_modified_{run_uuid}.csv"),
        "permissions_modified": os.path.join(wd, f"permissions_modified_{run_uuid}.csv"),
        "tiir_process_log": os.path.join(wd, f"tiir_process_log_{run_uuid}.jsonl"),
    }


class TIIRProcessLogger:
    def __init__(self, path: str):
        self.path = path

    def reset(self, session_meta=None):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            event = {
                "ts_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "stage": "SESSION",
                "event": "session_start",
                "status": "INFO",
                "data": _json_safe(session_meta or {}),
            }
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def log(self, stage: str, event: str, data=None, status: str = "INFO"):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        payload = {
            "ts_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "stage": stage,
            "event": event,
            "status": status,
            "data": _json_safe(data or {}),
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


class TIIRSystemLogger:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)

    def line(self, message: str):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(str(message).rstrip() + "\n")

    def event(self, stage: str, event: str, data=None, status: str = "INFO"):
        payload = {
            "ts_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "stage": stage,
            "event": event,
            "status": status,
            "data": _json_safe(data or {}),
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


_DEFAULT_PROCESS_LOG = os.path.join(os.getcwd(), globals().get("TIIR_PROCESS_LOG_NAME", "tiir_proces_log.jsonl"))
_DEFAULT_SYSTEM_LOG = os.path.join(os.getcwd(), globals().get("TIIR_SYSTEM_LOG_NAME", "tiir_system_log.txt"))

if "tiir_logger" not in globals():
    tiir_logger = TIIRProcessLogger(_DEFAULT_PROCESS_LOG)

if "tiir_system_logger" not in globals():
    tiir_system_logger = TIIRSystemLogger(_DEFAULT_SYSTEM_LOG)
