
#!/usr/bin/env python3
"""tiir_des_simulation_simpy_v6_with_plots.py

SimPy-based discrete-event simulation (DES) for routing-only workflows:
time-to-correct-assignment (TTCA), extended with v6 load-sweep plots.

Additions relative to tiir_des_simulation_simpy_v6.py
-----------------------------------------------------
1. Restores the missing v3-style load-sweep artifacts with v6 calibration:
   - tiir_v6_load_sweep.csv
   - tiir_v6_load_mean.png
   - tiir_v6_load_p95.png
   - tiir_v6_load_sla.png
   - tiir_v6_outcomes_vs_load.png
   - tiir_v6_reassign_vs_load.png
2. Preserves the original v6 baseline outputs and analytic snapshots.
3. Includes a deterministic event-driven fallback when `simpy` is unavailable,
   so the script can still generate the sweep and plots in minimal environments.

Primary paper calibration:
- Manual service time per attempt: mean 109.3 min, median 51.5 min, cap 26.3 h
- Human misroute rate: 21.8155%
- Reroute delay: calibrated to reproduce a 10.16x TTCA inflation on reassigned tickets
- TIIR precision/recall: 0.8155 / 0.4412
"""

from __future__ import annotations

import math
import os
import heapq
from dataclasses import dataclass, replace
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import simpy  # optional at runtime
except Exception:
    simpy = None

Scenario = Literal["manual", "tiir"]
ServiceDist = Literal["lognormal_mean_median", "deterministic"]

SUBJECT_REASSIGN_RATES = (
    8.56, 14.42, 11.25, 41.87, 47.66, 4.16, 51.50, 7.94, 8.02, 8.17,
    9.84, 6.78, 15.35, 91.58, 5.49, 5.91, 18.77, 63.82, 4.11, 11.11,
)
MEAN_SUBJECT_REASSIGN_RATE = sum(SUBJECT_REASSIGN_RATES) / (100.0 * len(SUBJECT_REASSIGN_RATES))


@dataclass(frozen=True)
class TimingPreset:
    name: str
    manual_service_dist: ServiceDist
    manual_mean_min: float
    manual_median_min: Optional[float]
    manual_cap_min: Optional[float]
    note: str


TIMING_PRESETS: Dict[str, TimingPreset] = {
    "seke_orgA": TimingPreset(
        name="seke_orgA",
        manual_service_dist="lognormal_mean_median",
        manual_mean_min=109.3,
        manual_median_min=51.5,
        manual_cap_min=26.3 * 60.0,
        note="Primary calibration: Open->Assignment from Organization A.",
    ),
    "navy_manual": TimingPreset(
        name="navy_manual",
        manual_service_dist="deterministic",
        manual_mean_min=17.0,
        manual_median_min=17.0,
        manual_cap_min=None,
        note="Median manual assignment time at NAVY 311 (external comparator).",
    ),
    "freshservice_it": TimingPreset(
        name="freshservice_it",
        manual_service_dist="deterministic",
        manual_mean_min=15.80 * 60.0,
        manual_median_min=15.80 * 60.0,
        manual_cap_min=None,
        note="Global IT AFAT (contextual benchmark only).",
    ),
    "freshservice_esm": TimingPreset(
        name="freshservice_esm",
        manual_service_dist="deterministic",
        manual_mean_min=28.93 * 60.0,
        manual_median_min=28.93 * 60.0,
        manual_cap_min=None,
        note="Global ESM AFAT (contextual benchmark only).",
    ),
    "freshservice_chat": TimingPreset(
        name="freshservice_chat",
        manual_service_dist="deterministic",
        manual_mean_min=4.16 * 60.0,
        manual_median_min=4.16 * 60.0,
        manual_cap_min=None,
        note="MS Teams/Slack AFAT (contextual benchmark only).",
    ),
    "freshservice_small": TimingPreset(
        name="freshservice_small",
        manual_service_dist="deterministic",
        manual_mean_min=14.22 * 60.0,
        manual_median_min=14.22 * 60.0,
        manual_cap_min=None,
        note="Organizations <250 employees (contextual benchmark only).",
    ),
    "freshservice_large": TimingPreset(
        name="freshservice_large",
        manual_service_dist="deterministic",
        manual_mean_min=20.465 * 60.0,
        manual_median_min=20.465 * 60.0,
        manual_cap_min=None,
        note="Organizations >1000 employees (contextual benchmark only).",
    ),
}


@dataclass(frozen=True)
class ExperimentConfig:
    routers: int = 5
    n_tickets: int = 200
    arrival_rate_per_hour: float = 2.0
    runs: int = 200
    seed0: int = 1

    timing_preset: str = "seke_orgA"

    manual_misroute_rate: float = MEAN_SUBJECT_REASSIGN_RATE
    reassignment_inflation_multiple: float = 10.16
    reroute_delay_min: Optional[float] = None
    max_reassignments: int = 6

    precision: float = 0.8155
    recall: float = 0.4412
    decision_rate: Optional[float] = None

    t_auto_min: float = 1.0
    t_handoff_min: float = 15.0 / 60.0
    navy_assisted_total_min: float = 4.5

    sla_minutes: float = 60.0

    lambda_sweep_per_hour: Tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
    timing_preset_sweep: Tuple[str, ...] = (
        "seke_orgA",
        "navy_manual",
        "freshservice_chat",
        "freshservice_it",
        "freshservice_esm",
    )


@dataclass(frozen=True)
class SOCSimParams:
    routers: int
    n_tickets: int
    arrival_rate_per_hour: float
    manual_service_dist: ServiceDist
    manual_mean_min: float
    manual_median_min: Optional[float]
    manual_cap_min: Optional[float]
    manual_misroute_rate: float
    reroute_delay_min: float
    max_reassignments: int
    t_auto_min: float
    t_handoff_min: float
    coverage: float
    p_correct_given_attempt: float
    sla_minutes: float


def derive_coverage_from_pr(precision: float, recall: float) -> float:
    if not (0.0 < precision <= 1.0):
        raise ValueError("precision must be in (0,1].")
    if not (0.0 <= recall <= 1.0):
        raise ValueError("recall must be in [0,1].")
    coverage = recall / precision
    if coverage > 1.0 + 1e-9:
        raise ValueError("recall/precision > 1. Single-decision assumption violated.")
    return min(1.0, coverage)


def calibrated_reroute_delay(mean_service_min: float, manual_misroute_rate: float, inflation_multiple: float = 10.16) -> float:
    p = manual_misroute_rate
    if not (0.0 < p < 1.0):
        raise ValueError("manual_misroute_rate must be in (0,1).")
    e_k_cond = (2.0 - p) / (1.0 - p)
    d = ((inflation_multiple - e_k_cond) / (e_k_cond - 1.0)) * mean_service_min
    if d < 0.0:
        raise ValueError("calibrated reroute delay became negative.")
    return d


def params_from_cfg(cfg: ExperimentConfig) -> SOCSimParams:
    if cfg.timing_preset not in TIMING_PRESETS:
        raise KeyError(f"Unknown timing preset: {cfg.timing_preset}")
    preset = TIMING_PRESETS[cfg.timing_preset]
    coverage = cfg.decision_rate if cfg.decision_rate is not None else derive_coverage_from_pr(cfg.precision, cfg.recall)
    p_correct_given_attempt = cfg.recall / coverage if coverage > 0 else 0.0

    if cfg.decision_rate is None and abs(p_correct_given_attempt - cfg.precision) > 1e-6:
        raise ValueError("Derived p(correct | attempt) does not match reported precision.")

    reroute = cfg.reroute_delay_min
    if reroute is None:
        reroute = calibrated_reroute_delay(
            mean_service_min=preset.manual_mean_min,
            manual_misroute_rate=cfg.manual_misroute_rate,
            inflation_multiple=cfg.reassignment_inflation_multiple,
        )

    return SOCSimParams(
        routers=cfg.routers,
        n_tickets=cfg.n_tickets,
        arrival_rate_per_hour=cfg.arrival_rate_per_hour,
        manual_service_dist=preset.manual_service_dist,
        manual_mean_min=preset.manual_mean_min,
        manual_median_min=preset.manual_median_min,
        manual_cap_min=preset.manual_cap_min,
        manual_misroute_rate=cfg.manual_misroute_rate,
        reroute_delay_min=reroute,
        max_reassignments=cfg.max_reassignments,
        t_auto_min=cfg.t_auto_min,
        t_handoff_min=cfg.t_handoff_min,
        coverage=coverage,
        p_correct_given_attempt=p_correct_given_attempt,
        sla_minutes=cfg.sla_minutes,
    )


def outcome_probs(p: SOCSimParams) -> Tuple[float, float, float]:
    p_correct = p.coverage * p.p_correct_given_attempt
    p_auto_misroute = p.coverage * (1.0 - p.p_correct_given_attempt)
    p_abstain = 1.0 - p.coverage
    s = p_correct + p_auto_misroute + p_abstain
    if abs(s - 1.0) > 1e-9:
        raise ValueError(f"Outcome probabilities must sum to 1, got {s}.")
    return p_correct, p_abstain, p_auto_misroute


def _lognormal_params_from_mean_median(mean: float, median: float) -> Tuple[float, float]:
    if mean <= 0 or median <= 0:
        raise ValueError("mean and median must be > 0")
    sigma2 = 2.0 * math.log(mean / median)
    return math.log(median), math.sqrt(max(0.0, sigma2))


def sample_manual_service(p: SOCSimParams, rng: np.random.Generator) -> float:
    if p.manual_service_dist == "deterministic":
        x = p.manual_mean_min
    elif p.manual_service_dist == "lognormal_mean_median":
        if p.manual_median_min is None:
            raise ValueError("manual_median_min required for lognormal_mean_median")
        mu, sigma = _lognormal_params_from_mean_median(p.manual_mean_min, p.manual_median_min)
        x = float(rng.lognormal(mean=mu, sigma=sigma))
    else:
        raise ValueError(f"Unsupported service distribution: {p.manual_service_dist}")
    if p.manual_cap_min is not None:
        x = min(x, p.manual_cap_min)
    return x


def mean_ci_t(x: np.ndarray) -> Tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    n = x.size
    m = float(np.mean(x)) if n else float("nan")
    if n <= 1:
        return m, float("nan"), float("nan")
    s = float(np.std(x, ddof=1))
    tcrit = 1.96 if n >= 30 else 2.262
    half = tcrit * s / math.sqrt(n)
    return m, m - half, m + half


def analytic_snapshot(cfg: ExperimentConfig) -> pd.DataFrame:
    p = params_from_cfg(cfg)
    p_correct, p_abstain, p_auto_misroute = outcome_probs(p)
    mu = p.manual_mean_min
    q = 1.0 - p.manual_misroute_rate
    e_attempts = 1.0 / q
    e_reassign = p.manual_misroute_rate / q
    e_attempts_cond = (2.0 - p.manual_misroute_rate) / q
    ttca_cond_reassigned = e_attempts_cond * mu + (e_attempts_cond - 1.0) * p.reroute_delay_min
    ttca_first_pass = mu
    ttca_manual_mean = e_attempts * mu + e_reassign * p.reroute_delay_min
    rho_manual_lb = cfg.arrival_rate_per_hour * mu / (cfg.routers * 60.0)
    rho_tiir_lb = cfg.arrival_rate_per_hour * (p_abstain + p_auto_misroute) * mu / (cfg.routers * 60.0)
    rows = [
        ("timing_preset", cfg.timing_preset),
        ("manual_mean_min", mu),
        ("manual_median_min", p.manual_median_min if p.manual_median_min is not None else ""),
        ("manual_cap_min", p.manual_cap_min if p.manual_cap_min is not None else ""),
        ("manual_misroute_rate", p.manual_misroute_rate),
        ("reroute_delay_min", p.reroute_delay_min),
        ("coverage", p.coverage),
        ("p_correct", p_correct),
        ("p_auto_misroute", p_auto_misroute),
        ("p_abstain", p_abstain),
        ("mean_attempts_manual", e_attempts),
        ("mean_reassignments_manual", e_reassign),
        ("ttca_first_pass_mean_min", ttca_first_pass),
        ("ttca_reassigned_conditional_mean_min", ttca_cond_reassigned),
        ("ttca_manual_unconditional_mean_min", ttca_manual_mean),
        ("rho_manual_lower_bound", rho_manual_lb),
        ("rho_tiir_lower_bound", rho_tiir_lb),
        ("tiir_auto_total_min", p.t_auto_min + p.t_handoff_min),
        ("navy_assisted_total_min", cfg.navy_assisted_total_min),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def routing_until_correct(env, res, p: SOCSimParams, rng: np.random.Generator, forced_reassignments: int = 0):
    start = env.now
    human = 0.0
    n_re = 0
    while True:
        svc = sample_manual_service(p, rng)
        with res.request() as req:
            yield req
            yield env.timeout(svc)
        human += svc

        if n_re < forced_reassignments:
            n_re += 1
            if n_re >= p.max_reassignments:
                break
            yield env.timeout(p.reroute_delay_min)
            continue

        if rng.random() < p.manual_misroute_rate and n_re < p.max_reassignments:
            n_re += 1
            yield env.timeout(p.reroute_delay_min)
            continue

        break

    return env.now - start, human, n_re, int(n_re > 0)


def ticket_process(env, res, p: SOCSimParams, rng: np.random.Generator, scenario: Scenario, out_rows: List[dict]):
    arrival = env.now
    if scenario == "manual":
        _, human, n_re, any_re = yield from routing_until_correct(env, res, p, rng, forced_reassignments=0)
        out_rows.append({"ttca_min": env.now - arrival, "human_min": human, "reassignments": n_re, "reassigned_any": any_re, "outcome_code": -1})
        return

    p_correct, p_abstain, p_auto_misroute = outcome_probs(p)
    u = rng.random()

    if u < p_correct:
        yield env.timeout(p.t_auto_min + p.t_handoff_min)
        out_rows.append({"ttca_min": env.now - arrival, "human_min": 0.0, "reassignments": 0, "reassigned_any": 0, "outcome_code": 0})
        return

    if u < p_correct + p_auto_misroute:
        yield env.timeout(p.t_auto_min + p.t_handoff_min)
        _, human, n_re, any_re = yield from routing_until_correct(env, res, p, rng, forced_reassignments=1)
        out_rows.append({"ttca_min": env.now - arrival, "human_min": human, "reassignments": n_re, "reassigned_any": max(1, any_re), "outcome_code": 2})
        return

    yield env.timeout(p.t_auto_min + p.t_handoff_min)
    _, human, n_re, any_re = yield from routing_until_correct(env, res, p, rng, forced_reassignments=0)
    out_rows.append({"ttca_min": env.now - arrival, "human_min": human, "reassignments": n_re, "reassigned_any": any_re, "outcome_code": 1})


def run_once_simpy(cfg: ExperimentConfig, scenario: Scenario, seed: int) -> pd.DataFrame:
    if simpy is None:
        raise RuntimeError("simpy is not installed in this environment.")
    p = params_from_cfg(cfg)
    rng = np.random.default_rng(seed)
    env = simpy.Environment()
    res = simpy.Resource(env, capacity=p.routers)
    out_rows: List[dict] = []

    def generator():
        for _ in range(p.n_tickets):
            dt = rng.exponential(60.0 / p.arrival_rate_per_hour)
            yield env.timeout(dt)
            env.process(ticket_process(env, res, p, rng, scenario, out_rows))

    env.process(generator())
    env.run()
    return pd.DataFrame(out_rows)


def run_once_fallback(cfg: ExperimentConfig, scenario: Scenario, seed: int) -> pd.DataFrame:
    """Event-driven fallback for environments without simpy.

    It preserves the same model semantics:
    - Poisson external arrivals
    - shared FCFS human-routing capacity
    - reroute delays outside the human resource
    - forced first human misroute for TIIR auto-misroute
    """
    p = params_from_cfg(cfg)
    rng = np.random.default_rng(seed)

    event_heap: List[Tuple[float, int, str, int]] = []
    waiting_queue: List[Tuple[int, int, float]] = []
    current_svc: Dict[int, float] = {}

    next_event_order = 0
    next_wait_order = 0
    busy = 0

    tickets: Dict[int, dict] = {}
    out_rows: List[dict] = []

    t = 0.0
    for ticket_id in range(p.n_tickets):
        t += float(rng.exponential(60.0 / p.arrival_rate_per_hour))
        heapq.heappush(event_heap, (t, next_event_order, "new_ticket", ticket_id))
        next_event_order += 1

    def enqueue_human(now: float, ticket_id: int) -> None:
        nonlocal next_wait_order
        svc = sample_manual_service(p, rng)
        waiting_queue.append((next_wait_order, ticket_id, svc))
        next_wait_order += 1

    def maybe_start(now: float) -> None:
        nonlocal busy, next_event_order
        while busy < p.routers and waiting_queue:
            _, ticket_id, svc = waiting_queue.pop(0)
            busy += 1
            current_svc[ticket_id] = svc
            heapq.heappush(event_heap, (now + svc, next_event_order, "service_completion", ticket_id))
            next_event_order += 1

    while event_heap:
        now, _, event_type, ticket_id = heapq.heappop(event_heap)

        if event_type == "new_ticket":
            tickets[ticket_id] = {"arrival": now, "human": 0.0, "n_re": 0, "forced": 0, "outcome_code": -1}

            if scenario == "manual":
                enqueue_human(now, ticket_id)
                maybe_start(now)
                continue

            p_correct, p_abstain, p_auto_misroute = outcome_probs(p)
            u = float(rng.random())

            if u < p_correct:
                out_rows.append({"ttca_min": p.t_auto_min + p.t_handoff_min, "human_min": 0.0, "reassignments": 0, "reassigned_any": 0, "outcome_code": 0})
                del tickets[ticket_id]
            elif u < p_correct + p_auto_misroute:
                tickets[ticket_id]["forced"] = 1
                tickets[ticket_id]["outcome_code"] = 2
                heapq.heappush(event_heap, (now + p.t_auto_min + p.t_handoff_min, next_event_order, "human_arrival", ticket_id))
                next_event_order += 1
            else:
                tickets[ticket_id]["forced"] = 0
                tickets[ticket_id]["outcome_code"] = 1
                heapq.heappush(event_heap, (now + p.t_auto_min + p.t_handoff_min, next_event_order, "human_arrival", ticket_id))
                next_event_order += 1

        elif event_type == "human_arrival":
            enqueue_human(now, ticket_id)
            maybe_start(now)

        elif event_type == "service_completion":
            busy -= 1
            svc = current_svc.pop(ticket_id)
            state = tickets[ticket_id]
            state["human"] += svc

            if state["n_re"] < state["forced"]:
                state["n_re"] += 1
                if state["n_re"] >= p.max_reassignments:
                    out_rows.append({
                        "ttca_min": now - state["arrival"],
                        "human_min": state["human"],
                        "reassignments": state["n_re"],
                        "reassigned_any": int(state["n_re"] > 0),
                        "outcome_code": state["outcome_code"],
                    })
                    del tickets[ticket_id]
                else:
                    heapq.heappush(event_heap, (now + p.reroute_delay_min, next_event_order, "human_arrival", ticket_id))
                    next_event_order += 1

            elif float(rng.random()) < p.manual_misroute_rate and state["n_re"] < p.max_reassignments:
                state["n_re"] += 1
                heapq.heappush(event_heap, (now + p.reroute_delay_min, next_event_order, "human_arrival", ticket_id))
                next_event_order += 1

            else:
                out_rows.append({
                    "ttca_min": now - state["arrival"],
                    "human_min": state["human"],
                    "reassignments": state["n_re"],
                    "reassigned_any": int(state["n_re"] > 0),
                    "outcome_code": state["outcome_code"],
                })
                del tickets[ticket_id]

            maybe_start(now)

        else:
            raise ValueError(f"Unknown event type: {event_type}")

    return pd.DataFrame(out_rows)


def run_once(cfg: ExperimentConfig, scenario: Scenario, seed: int) -> pd.DataFrame:
    if simpy is not None:
        return run_once_simpy(cfg, scenario, seed)
    return run_once_fallback(cfg, scenario, seed)


def summarize_runs(cfg: ExperimentConfig, scenario: Scenario) -> pd.DataFrame:
    run_rows = []
    for i in range(cfg.runs):
        df = run_once(cfg, scenario, cfg.seed0 + i)

        row = {
            "scenario": scenario,
            "run": i,
            "mean_ttca": float(df["ttca_min"].mean()),
            "median_ttca": float(df["ttca_min"].median()),
            "p95_ttca": float(df["ttca_min"].quantile(0.95)),
            "sla_viol_rate": float((df["ttca_min"] > cfg.sla_minutes).mean()),
            "mean_human_min": float(df["human_min"].mean()),
            "reassigned_any_rate": float(df["reassigned_any"].mean()),
        }

        if scenario == "tiir":
            row["rate_auto_correct"] = float((df["outcome_code"] == 0).mean())
            row["rate_abstain"] = float((df["outcome_code"] == 1).mean())
            row["rate_auto_misroute"] = float((df["outcome_code"] == 2).mean())
        else:
            row["rate_auto_correct"] = float("nan")
            row["rate_abstain"] = float("nan")
            row["rate_auto_misroute"] = float("nan")

        run_rows.append(row)

    return pd.DataFrame(run_rows)


def aggregate_summary(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "mean_ttca",
        "median_ttca",
        "p95_ttca",
        "sla_viol_rate",
        "mean_human_min",
        "reassigned_any_rate",
    ]
    out = []
    for metric in metrics:
        vals = df[metric].to_numpy(dtype=float)
        m, lo, hi = mean_ci_t(vals)
        out.append({"metric": metric, "mean": m, "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(out)


def save_analytic_bundle(cfg: ExperimentConfig, outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    analytic_snapshot(cfg).to_csv(os.path.join(outdir, "tiir_v6_analytic_snapshot.csv"), index=False)

    preset_rows = []
    for preset_name in cfg.timing_preset_sweep:
        cfg_p = replace(cfg, timing_preset=preset_name)
        snap = analytic_snapshot(cfg_p)
        vals = {row["metric"]: row["value"] for _, row in snap.iterrows()}
        vals["timing_preset"] = preset_name
        preset_rows.append(vals)
    pd.DataFrame(preset_rows).to_csv(os.path.join(outdir, "tiir_v6_timing_preset_snapshot.csv"), index=False)

    util_rows = []
    p = params_from_cfg(cfg)
    _, p_abstain, p_auto_misroute = outcome_probs(p)
    for lam in cfg.lambda_sweep_per_hour:
        util_rows.append(
            {
                "lambda_per_hour": lam,
                "rho_manual_lower_bound": lam * p.manual_mean_min / (cfg.routers * 60.0),
                "rho_tiir_lower_bound": lam * (p_abstain + p_auto_misroute) * p.manual_mean_min / (cfg.routers * 60.0),
            }
        )
    pd.DataFrame(util_rows).to_csv(os.path.join(outdir, "tiir_v6_lower_bound_utilization.csv"), index=False)


def save_des_bundle(cfg: ExperimentConfig, outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    for scenario in ("manual", "tiir"):
        runs = summarize_runs(cfg, scenario)
        runs.to_csv(os.path.join(outdir, f"tiir_v6_{scenario}_runs.csv"), index=False)
        aggregate_summary(runs).to_csv(os.path.join(outdir, f"tiir_v6_{scenario}_summary.csv"), index=False)


def load_sweep(cfg: ExperimentConfig, outdir: str) -> pd.DataFrame:
    rows: List[dict] = []
    for lam in cfg.lambda_sweep_per_hour:
        cfg_l = replace(cfg, arrival_rate_per_hour=float(lam))
        for scenario in ("manual", "tiir"):
            runs = summarize_runs(cfg_l, scenario)
            for metric in [
                "mean_ttca",
                "median_ttca",
                "p95_ttca",
                "sla_viol_rate",
                "mean_human_min",
                "reassigned_any_rate",
            ]:
                rows.append(
                    {
                        "lambda_per_hour": float(lam),
                        "scenario": scenario,
                        "metric": metric,
                        "mean": float(runs[metric].mean()),
                    }
                )
            if scenario == "tiir":
                for metric in ["rate_auto_correct", "rate_abstain", "rate_auto_misroute"]:
                    rows.append(
                        {
                            "lambda_per_hour": float(lam),
                            "scenario": scenario,
                            "metric": metric,
                            "mean": float(runs[metric].mean()),
                        }
                    )
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(outdir, "tiir_v6_load_sweep.csv"), index=False)
    return df


def plot_load_curves(df_load: pd.DataFrame, outdir: str) -> None:
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)

    for metric, fname, ylabel in [
        ("mean_ttca", "tiir_v6_load_mean.png", "Mean TTCA (min)"),
        ("p95_ttca", "tiir_v6_load_p95.png", "Global P95 TTCA (min)"),
        ("sla_viol_rate", "tiir_v6_load_sla.png", "SLA violation rate"),
    ]:
        plt.figure()
        for scenario in ["manual", "tiir"]:
            d = df_load[(df_load["metric"] == metric) & (df_load["scenario"] == scenario)].sort_values("lambda_per_hour")
            plt.plot(d["lambda_per_hour"], d["mean"], marker="o", label=scenario)
        plt.xlabel("Arrival rate λ (tickets/hour)")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, fname), dpi=200)
        plt.close()

    plt.figure()
    for metric, label in [
        ("rate_auto_correct", "auto-correct"),
        ("rate_abstain", "abstain"),
        ("rate_auto_misroute", "auto-misroute"),
    ]:
        d = df_load[(df_load["scenario"] == "tiir") & (df_load["metric"] == metric)].sort_values("lambda_per_hour")
        plt.plot(d["lambda_per_hour"], d["mean"], marker="o", label=label)
    plt.xlabel("Arrival rate λ (tickets/hour)")
    plt.ylabel("Outcome rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "tiir_v6_outcomes_vs_load.png"), dpi=200)
    plt.close()

    plt.figure()
    for scenario in ["manual", "tiir"]:
        d = df_load[(df_load["metric"] == "reassigned_any_rate") & (df_load["scenario"] == scenario)].sort_values("lambda_per_hour")
        plt.plot(d["lambda_per_hour"], d["mean"], marker="o", label=scenario)
    plt.xlabel("Arrival rate λ (tickets/hour)")
    plt.ylabel("Human reassignment rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "tiir_v6_reassign_vs_load.png"), dpi=200)
    plt.close()


def main() -> None:
    outdir = os.path.abspath("results_tiir_v6")
    cfg = ExperimentConfig()

    save_analytic_bundle(cfg, outdir)
    save_des_bundle(cfg, outdir)
    df_load = load_sweep(cfg, outdir)
    plot_load_curves(df_load, outdir)

    if simpy is None:
        print("simpy not installed; used deterministic fallback DES for baseline + sweep + plots.")
    else:
        print("simpy installed; used SimPy DES for baseline + sweep + plots.")
    print(f"Output directory: {outdir}")


if __name__ == "__main__":
    main()
