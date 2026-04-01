"""
MiroFish Ensemble Prediction Runner
=====================================
Runs N independent simulations for the same prediction question,
then aggregates the probability estimates to reduce variance.

Usage:
    python ensemble_predict.py --simulation-id sim_xxx --runs 3
    python ensemble_predict.py --simulation-id sim_xxx --runs 5 --base-url http://localhost:5001

The simulation must already be prepared (profiles generated, config ready).
This script re-runs the simulation N times, generates a report for each run,
then outputs an aggregated probability with a confidence interval.

Requirements:
    pip install requests scipy
"""

import argparse
import json
import statistics
import time
import sys
from typing import Optional
import requests


# ─────────────────────────────────────────
# API client helpers
# ─────────────────────────────────────────

def api(base_url: str, method: str, path: str, **kwargs) -> dict:
    """Simple API wrapper with basic error handling."""
    url = f"{base_url.rstrip('/')}{path}"
    resp = getattr(requests, method)(url, timeout=600, **kwargs)
    resp.raise_for_status()
    data = resp.json()
    if not data.get("success", True):
        raise RuntimeError(f"API error on {path}: {data.get('error', 'unknown')}")
    return data


def poll(base_url: str, path: str, payload: dict,
         done_statuses: set, fail_statuses: set,
         interval: int = 10, timeout: int = 1800) -> dict:
    """Poll a status endpoint until done or failed."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        data = api(base_url, "post", path, json=payload)
        status = (data.get("data") or data).get("status", "")
        if status in done_statuses:
            return data
        if status in fail_statuses:
            raise RuntimeError(f"Task failed at {path}: {data}")
        print(f"    … {status}", flush=True)
        time.sleep(interval)
    raise TimeoutError(f"Polling {path} timed out after {timeout}s")


# ─────────────────────────────────────────
# Single-run helpers
# ─────────────────────────────────────────

def start_simulation(base_url: str, simulation_id: str) -> str:
    """Start a simulation run and return the run_id."""
    data = api(base_url, "post", "/api/simulation/run/start",
               json={"simulation_id": simulation_id})
    run_id = data["data"]["run_id"]
    print(f"    Started run: {run_id}")
    return run_id


def wait_for_simulation(base_url: str, simulation_id: str,
                        run_id: str, interval: int = 15) -> None:
    """Wait until a simulation run completes."""
    deadline = time.time() + 3600  # 1 hour max
    while time.time() < deadline:
        data = api(base_url, "get",
                   f"/api/simulation/run/status/{simulation_id}/{run_id}")
        status = data["data"].get("runner_status", "")
        print(f"    … simulation status: {status}", flush=True)
        if status in {"completed"}:
            return
        if status in {"failed", "stopped"}:
            raise RuntimeError(f"Simulation {run_id} ended with status: {status}")
        time.sleep(interval)
    raise TimeoutError("Simulation timed out after 1 hour")


def generate_report(base_url: str, simulation_id: str) -> str:
    """Kick off report generation and return report_id."""
    data = api(base_url, "post", "/api/report/generate",
               json={"simulation_id": simulation_id})
    report_id = data["data"]["report_id"]
    task_id = data["data"]["task_id"]
    print(f"    Generating report {report_id} (task {task_id}) …")

    # Poll for completion
    poll(base_url, "/api/report/generate/status",
         {"task_id": task_id},
         done_statuses={"completed"},
         fail_statuses={"failed"},
         interval=10, timeout=900)
    return report_id


def extract_probability(base_url: str, report_id: str) -> dict:
    """Extract probability fields from a completed report."""
    data = api(base_url, "get", f"/api/report/{report_id}")
    report = data["data"]
    outline = report.get("outline") or {}
    return {
        "report_id": report_id,
        "predicted_probability": outline.get("predicted_probability"),
        "probability_low": outline.get("probability_low"),
        "probability_high": outline.get("probability_high"),
        "key_upside_factors": outline.get("key_upside_factors", []),
        "key_downside_factors": outline.get("key_downside_factors", []),
        "title": report.get("title", ""),
    }


# ─────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────

def aggregate(results: list[dict]) -> dict:
    """
    Aggregate probability estimates across N runs.

    Uses:
    - Point estimate: mean of all predicted_probability values
    - Uncertainty range: min(probability_low) … max(probability_high),
      further expanded by ±1 stdev of point estimates across runs
    - Factor frequency: factors mentioned in 2+ runs are "consensus factors"
    """
    points = [r["predicted_probability"] for r in results
              if r.get("predicted_probability") is not None]
    lows = [r["probability_low"] for r in results
            if r.get("probability_low") is not None]
    highs = [r["probability_high"] for r in results
             if r.get("probability_high") is not None]

    if not points:
        return {"error": "No probability estimates could be extracted"}

    mean_p = round(statistics.mean(points))
    stdev_p = round(statistics.stdev(points)) if len(points) > 1 else 0

    # Confidence interval: widen individual run ranges by cross-run stdev
    agg_low = max(0, (min(lows) if lows else mean_p) - stdev_p)
    agg_high = min(100, (max(highs) if highs else mean_p) + stdev_p)

    # Factor frequency
    upside_counts: dict[str, int] = {}
    downside_counts: dict[str, int] = {}
    for r in results:
        for f in r.get("key_upside_factors", []):
            upside_counts[f] = upside_counts.get(f, 0) + 1
        for f in r.get("key_downside_factors", []):
            downside_counts[f] = downside_counts.get(f, 0) + 1

    # Keep factors mentioned by at least 2 runs (consensus), sorted by frequency
    n = len(results)
    consensus_upside = sorted(
        [f for f, c in upside_counts.items() if c >= min(2, n)],
        key=lambda f: -upside_counts[f]
    )
    consensus_downside = sorted(
        [f for f, c in downside_counts.items() if c >= min(2, n)],
        key=lambda f: -downside_counts[f]
    )

    return {
        "runs": n,
        "point_estimates": points,
        "aggregated_probability": mean_p,
        "probability_low": round(agg_low),
        "probability_high": round(agg_high),
        "stdev_across_runs": stdev_p,
        "consensus_upside_factors": consensus_upside,
        "consensus_downside_factors": consensus_downside,
    }


def print_report(agg: dict, results: list[dict]) -> None:
    bar = "=" * 60
    print(f"\n{bar}")
    print("  ENSEMBLE PREDICTION RESULT")
    print(bar)
    print(f"\n  Runs completed       : {agg['runs']}")
    print(f"  Per-run estimates    : {agg['point_estimates']}")
    print(f"  Std deviation        : ±{agg['stdev_across_runs']}%")
    print(f"\n  ┌─────────────────────────────────────┐")
    print(f"  │  Aggregated probability : {agg['aggregated_probability']:>3}%       │")
    print(f"  │  80% confidence range   : {agg['probability_low']}% – {agg['probability_high']}%   │")
    print(f"  └─────────────────────────────────────┘")

    if agg.get("consensus_upside_factors"):
        print("\n  Consensus upside factors (raise probability):")
        for f in agg["consensus_upside_factors"][:5]:
            print(f"    + {f}")

    if agg.get("consensus_downside_factors"):
        print("\n  Consensus downside factors (lower probability):")
        for f in agg["consensus_downside_factors"][:5]:
            print(f"    - {f}")

    print(f"\n  Individual report IDs:")
    for r in results:
        p = r.get("predicted_probability", "N/A")
        lo = r.get("probability_low", "?")
        hi = r.get("probability_high", "?")
        print(f"    {r['report_id']}  →  {p}% ({lo}%–{hi}%)")

    print(f"\n{bar}\n")


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run N independent simulations and aggregate probability estimates"
    )
    parser.add_argument("--simulation-id", required=True,
                        help="ID of a prepared simulation (profiles + config must exist)")
    parser.add_argument("--runs", type=int, default=3,
                        help="Number of independent simulation runs (default: 3)")
    parser.add_argument("--base-url", default="http://localhost:5001",
                        help="MiroFish backend URL (default: http://localhost:5001)")
    parser.add_argument("--output", default=None,
                        help="Optional path to write JSON results")
    args = parser.parse_args()

    if args.runs < 2:
        print("Warning: --runs should be at least 2 for meaningful aggregation")

    print(f"\nEnsemble runner: {args.runs} runs for simulation {args.simulation_id}")
    print(f"Backend: {args.base_url}\n")

    results = []
    for i in range(1, args.runs + 1):
        print(f"─── Run {i}/{args.runs} ───")
        try:
            run_id = start_simulation(args.base_url, args.simulation_id)
            wait_for_simulation(args.base_url, args.simulation_id, run_id)
            report_id = generate_report(args.base_url, args.simulation_id)
            prob = extract_probability(args.base_url, report_id)
            print(f"    ✓  probability = {prob['predicted_probability']}%"
                  f"  range [{prob.get('probability_low')}%–{prob.get('probability_high')}%]")
            results.append(prob)
        except Exception as e:
            print(f"    ✗  Run {i} failed: {e}", file=sys.stderr)

    if not results:
        print("All runs failed. Exiting.", file=sys.stderr)
        sys.exit(1)

    agg = aggregate(results)
    print_report(agg, results)

    output = {"aggregated": agg, "individual_runs": results}
    if args.output:
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"Results written to {args.output}")

    return output


if __name__ == "__main__":
    main()
