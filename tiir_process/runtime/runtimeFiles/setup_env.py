
import gc
import io
import logging
import os
import pickle
import subprocess
import time
import warnings
from contextlib import redirect_stderr, redirect_stdout

#GITHUB_USER = "soctoiam"
#REPO = "soc_to_iam"
#BRANCH = "main"
#SCRIPT_DIR = "tiir_process/runtime/runtimeFiles/"
#BASE_URL = f"https://raw.githubusercontent.com/{GITHUB_USER}/{REPO}/{BRANCH}/{SCRIPT_DIR}"

if "TIIR_SETUP_SESSION_ID" not in globals():
    TIIR_SETUP_SESSION_ID = time.strftime("%Y%m%d_%H%M%S")
if "TIIR_SYSTEM_LOG_NAME" not in globals():
    TIIR_SYSTEM_LOG_NAME = f"tiir_system_log_{TIIR_SETUP_SESSION_ID}.log"

_LOG_UTIL = "tiir_process_log_utils.py"
subprocess.check_call(["wget", "-q", "-O", _LOG_UTIL, f"{BASE_URL}{_LOG_UTIL}"])

with open(_LOG_UTIL, "r", encoding="utf-8") as _f:
    exec(_f.read(), globals())

required_symbols = ["TIIRSystemLogger", "TIIRProcessLogger", "build_run_artifact_paths"]
missing_symbols = [s for s in required_symbols if s not in globals()]
if missing_symbols:
    raise RuntimeError(f"{_LOG_UTIL} loaded, but missing symbols: {missing_symbols}")

tiir_system_logger = TIIRSystemLogger(os.path.join(os.getcwd(), TIIR_SYSTEM_LOG_NAME))
tiir_system_logger.line(f"[{time.strftime('%H:%M:%S')}] SETUP SESSION START")
tiir_system_logger.event("setup", "session_start", {"base_url": BASE_URL, "system_log": TIIR_SYSTEM_LOG_NAME})

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

warnings.filterwarnings("ignore", message=r".*Trying to unpickle estimator.*")
warnings.filterwarnings("ignore", message=r".*unauthenticated requests to the HF Hub.*")
for logger_name in ["huggingface_hub", "transformers", "urllib3", "filelock"]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

_LAST_STAGE_MESSAGE = None

def log(msg):
    line = f"\n[{time.strftime('%H:%M:%S')}] {msg}"
    print(line)
    tiir_system_logger.line(line)


def stage(msg):
    global _LAST_STAGE_MESSAGE
    _LAST_STAGE_MESSAGE = msg
    print(f"   {msg}...", end=" ", flush=True)
    tiir_system_logger.line(f"   {msg}...")


def done(msg="Done."):
    print(msg)
    tiir_system_logger.line(f"   {msg}")


def fail(msg):
    print(f"FAILED: {msg}")
    tiir_system_logger.line(f"   FAILED: {msg}")


def run_command(command, task_name):
    result = subprocess.run(
        command,
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        tail = (result.stderr or "").strip().splitlines()
        tail = tail[-1] if tail else "unknown error"
        raise RuntimeError(f"{task_name} failed: {tail[:160]}")


class _QuietSection:
    def __enter__(self):
        self.sink = io.StringIO()
        self.redirect_out = redirect_stdout(self.sink)
        self.redirect_err = redirect_stderr(self.sink)
        self.redirect_out.__enter__()
        self.redirect_err.__enter__()
        self.warn_ctx = warnings.catch_warnings()
        self.warn_ctx.__enter__()
        warnings.filterwarnings("ignore", message=r".*Trying to unpickle estimator.*")
        warnings.filterwarnings("ignore", message=r".*unauthenticated requests to the HF Hub.*")
        return self

    def __exit__(self, exc_type, exc, tb):
        self.warn_ctx.__exit__(exc_type, exc, tb)
        self.redirect_err.__exit__(exc_type, exc, tb)
        self.redirect_out.__exit__(exc_type, exc, tb)
        return False

print("Setting up TIIR: This may take 5-8 min.")
log("STEP 1/4: Checking environment")
stage("Checking and installing required libraries")
step1_start = time.time()
install_actions = []
try:
    run_command("pip install -q -U --force-reinstall 'protobuf>=3.20.3'", "Fixing Protobuf")
    try:
        import bitsandbytes  # noqa: F401
        import kagglehub  # noqa: F401
        install_actions.append("core_libraries_already_present")
    except ImportError:
        run_command("pip install -q -U bitsandbytes", "Installing bitsandbytes")
        run_command(
            "pip install -q -U 'transformers>=4.41.2' 'peft>=0.11.1' 'accelerate>=0.30.1' 'datasets>=2.19.1'",
            "Installing HF stack",
        )
        run_command("pip install -q kagglehub", "Installing KaggleHub")
        run_command("pip install -q scipy scikit-learn pandas", "Installing data-science stack")
        install_actions.extend(["bitsandbytes", "hf_stack", "kagglehub", "data_science_stack"])
    tiir_system_logger.event("setup.step1_environment", "completed", {
        "elapsed_s": round(time.time() - step1_start, 3),
        "install_actions": install_actions,
    })
    done()
except Exception as exc:
    tiir_system_logger.event("setup.step1_environment", "failed", {
        "elapsed_s": round(time.time() - step1_start, 3),
        "error": str(exc),
    }, status="ERROR")
    fail(str(exc))
    raise

os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")

import torch
import pandas as pd
import scipy.sparse
import kagglehub
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


def _cleanup_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def _required_runtime_objects_present():
    required = ["model_cpe", "tokenizer_cpe", "vectorizer", "tfidf_matrix", "df_meta", "cpe_col"]
    return all(name in globals() and globals()[name] is not None for name in required)


def _model_loaded_on_expected_device(device_name: str):
    model = globals().get("model_cpe")
    if model is None:
        return False
    hf_device_map = getattr(model, "hf_device_map", None)
    if hf_device_map:
        if device_name.startswith("cuda") or device_name == "auto":
            return any(str(v).startswith("cuda") or str(v).isdigit() for v in hf_device_map.values())
        return any(str(v) == device_name for v in hf_device_map.values())
    try:
        first_param = next(model.parameters())
        actual_device = str(first_param.device)
        if device_name == "auto":
            return actual_device.startswith("cuda") or actual_device == "cpu"
        return actual_device == device_name
    except Exception:
        return False


def _teardown_partial_model_state():
    removed = []
    for name in ["model_cpe", "tokenizer_cpe"]:
        if name in globals():
            try:
                del globals()[name]
                removed.append(name)
            except Exception:
                pass
    _cleanup_gpu_memory()
    return removed


def _pick_device_map(n_gpus: int):
    if n_gpus <= 0:
        return "cpu"
    if n_gpus == 1:
        return "cuda:0"
    return "auto"

log("STEP 2/4: Configuring device")
stage("Inspecting available GPU devices")
step2_start = time.time()
try:
    n_gpus = torch.cuda.device_count()
    device_cpe = _pick_device_map(n_gpus)
    tiir_system_logger.event("setup.step2_device", "completed", {
        "elapsed_s": round(time.time() - step2_start, 3),
        "gpu_count": n_gpus,
        "device_cpe": device_cpe,
    })
    done()
    print(f"   GPU count: {n_gpus}")
    print(f"   text2CPE device: {device_cpe}")
    tiir_system_logger.line(f"   GPU count: {n_gpus}")
    tiir_system_logger.line(f"   text2CPE device: {device_cpe}")
except Exception as exc:
    tiir_system_logger.event("setup.step2_device", "failed", {
        "elapsed_s": round(time.time() - step2_start, 3),
        "error": str(exc),
    }, status="ERROR")
    fail(str(exc))
    raise

log("STEP 3/4: Loading text2CPE model")
hf_token = None
try:
    from kaggle_secrets import UserSecretsClient
    hf_token = UserSecretsClient().get_secret("HF_TOKEN")
except Exception:
    pass

MODEL_HANDLE = "mathismller/mistral-cpe-extractor/pyTorch/default/1"
base_model_id = "mistralai/Mistral-7B-Instruct-v0.3"

stage("Loading LoRA adapter and base model")
step3_start = time.time()
try:
    if _required_runtime_objects_present() and _model_loaded_on_expected_device(device_cpe):
        tiir_system_logger.event("setup.step3_model", "reused", {
            "elapsed_s": round(time.time() - step3_start, 3),
            "device": device_cpe,
        })
        done()
        print("   Reusing already loaded text2CPE model and tokenizer.")
        tiir_system_logger.line("   Reusing already loaded text2CPE model and tokenizer.")
    else:
        removed = _teardown_partial_model_state()
        with _QuietSection():
            try:
                adapter_path = kagglehub.model_download(MODEL_HANDLE)
                model_source = "model registry"
            except Exception:
                adapter_path = kagglehub.dataset_download("mathismller/mistral-cpe-extractor")
                model_source = "dataset registry"

            tokenizer_cpe = AutoTokenizer.from_pretrained(base_model_id, token=hf_token)
            tokenizer_cpe.pad_token = tokenizer_cpe.eos_token

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model_cpe = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                quantization_config=bnb_config,
                device_map=device_cpe,
                dtype=torch.bfloat16,
                token=hf_token,
            )
            model_cpe = PeftModel.from_pretrained(model_cpe, adapter_path)
            model_cpe.eval()

        tiir_system_logger.event("setup.step3_model", "completed", {
            "elapsed_s": round(time.time() - step3_start, 3),
            "device": device_cpe,
            "adapter_source": model_source,
            "adapter_path": adapter_path,
            "cleared_partial_state": removed,
        })
        done()
        print(f"   Adapter source: {model_source}")
        print("   text2CPE model loaded successfully.")
        tiir_system_logger.line(f"   Adapter source: {model_source}")
        tiir_system_logger.line("   text2CPE model loaded successfully.")
except Exception as exc:
    oom_hint = ""
    if "out of memory" in str(exc).lower():
        _teardown_partial_model_state()
        oom_hint = " GPU memory was cleared after the failed load attempt."
    tiir_system_logger.event("setup.step3_model", "failed", {
        "elapsed_s": round(time.time() - step3_start, 3),
        "device": device_cpe,
        "error": str(exc),
        "oom_cleanup_performed": bool(oom_hint),
    }, status="ERROR")
    fail(str(exc) + oom_hint)
    raise

log("STEP 4/4: Loading RAG knowledge base")
rag_files = ["cpe_meta.parquet", "cpe_tfidf.npz", "vectorizer.pkl"]
reviewer_runtime_files = [
    "tiir_process_log_utils.py",
    "tiir_input_router.py",
    "text2CPE_inference.py",
    "orchestrator_stix.py",
    "Loader.py",
    "tiir_pipeline_runner.py",
    "attack_json_fail.json",
    "attack_json_succ.json",
    "Accounts.CSV",
    "Permissions.CSV",
    "Accounts_advanced_demo.CSV",
    "Accounts_advanced_demo_unavailable.CSV",
    "Permissions_advanced_demo.CSV",
    "sap_demo_input.json",
]

rag_path = os.getcwd()
stage("Fetching RAG artifacts and runtime files")
step4_start = time.time()
try:
    fetched_rag = []
    for file_name in rag_files:
        if not os.path.exists(file_name):
            run_command(f"wget -q -O {file_name} {BASE_URL}{file_name}", f"Downloading {file_name}")
            fetched_rag.append(file_name)

    with _QuietSection():
        df_meta = pd.read_parquet(os.path.join(rag_path, "cpe_meta.parquet"))
        tfidf_matrix = scipy.sparse.load_npz(os.path.join(rag_path, "cpe_tfidf.npz"))
        with open(os.path.join(rag_path, "vectorizer.pkl"), "rb") as f:
            vectorizer = pickle.load(f)

    cpe_col = next((c for c in ["cpe_uri", "cpe_2_3", "cpe"] if c in df_meta.columns), df_meta.columns[0])

    fetched_runtime = []
    for fname in reviewer_runtime_files:
        if not os.path.exists(fname):
            run_command(f"wget -q -O {fname} {BASE_URL}{fname}", f"Fetching {fname}")
            fetched_runtime.append(fname)

    tiir_system_logger.event("setup.step4_rag", "completed", {
        "elapsed_s": round(time.time() - step4_start, 3),
        "rag_entries": len(df_meta),
        "cpe_col": cpe_col,
        "fetched_rag": fetched_rag,
        "runtime_files_fetched": fetched_runtime,
        "base_url": BASE_URL,
    })
    done()
    required_runtime_files = [
        "tiir_process_log_utils.py",
        "tiir_input_router.py",
        "text2CPE_inference.py",
        "orchestrator_stix.py",
        "Loader.py",
        "tiir_pipeline_runner.py",
    ]
    missing_files = [f for f in required_runtime_files if not os.path.exists(f)]
    if missing_files:
        raise RuntimeError(f"Missing runtime files after setup: {missing_files}")
    print(f"   RAG entries: {len(df_meta)} | target column: {cpe_col}")
    tiir_system_logger.line(f"   RAG entries: {len(df_meta)} | target column: {cpe_col}")
    if fetched_runtime:
        print(f"   Runtime files fetched: {len(fetched_runtime)}")
        tiir_system_logger.line(f"   Runtime files fetched: {len(fetched_runtime)}")
    else:
        print("   Runtime files already present.")
        tiir_system_logger.line("   Runtime files already present.")
except Exception as exc:
    tiir_system_logger.event("setup.step4_rag", "failed", {
        "elapsed_s": round(time.time() - step4_start, 3),
        "error": str(exc),
    }, status="ERROR")
    fail(str(exc))
    raise

log("SYSTEM READY. Proceed with Cell 2.")
tiir_system_logger.event("setup", "system_ready", {"system_log": TIIR_SYSTEM_LOG_NAME})
SETUP_COMPLETED = True
