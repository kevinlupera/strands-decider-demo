"""Model routing: let a decision model pick the LLM tier, and measure what you save.

Most agent decisions are tiny judgements ("is this trivial or hard?") that are too
nuanced for a hand-written `if` and too small to justify a frontier LLM call. That is
exactly the lane for a decision model. Here we ask Strands Decider ONE `noul` question
per query -- "is this request complex?" -- and route:

    complex (P >= THRESHOLD) -> frontier model   (expensive, smart)
    simple                   -> cheap model       (fast, cheap)

The benchmark compares that hybrid policy against the naive "always frontier" baseline
on a small labelled dataset, and reports routing accuracy, latency, and cost.

Run:
    strands-decider serve StrandsAgents/strands-decider-2B-hobson-v19 --port 8099
    python bench.py                 # real model routing (no Bedrock calls needed)
    python bench.py --fallback      # offline heuristic, runs anywhere

COST NOTE
---------
Prices below are per-1K-token Amazon Bedrock on-demand estimates and WILL go stale.
They are a template, not a quote. Check the live Bedrock pricing page and the Strands
Decider latency notes before citing any number from this script.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import platform
import sys
from datetime import datetime, timezone

from decider_client import Decider, ServerUnavailable

HERE = pathlib.Path(__file__).parent
DATASET = HERE / "data" / "queries.jsonl"
RESULTS_JSON = HERE / "results" / "latest.json"
BENCH_MD = HERE / "docs" / "benchmark.md"

# ---- editable assumptions (verify before citing) -------------------------
# Per-1,000-token USD estimates. Replace with live numbers from the Bedrock pricing
# page for your region. Averaged (input+output)/2 to keep the demo arithmetic simple;
# real billing separates input and output rates.
PRICE_PER_1K = {
    "cheap":    0.0005,   # e.g. a Haiku-class / Nova-lite-class model (ILLUSTRATIVE)
    "frontier": 0.0100,   # e.g. a Sonnet-class / frontier model       (ILLUSTRATIVE)
}
AVG_TOKENS_PER_QUERY = 600   # rough input+output per answered query (ILLUSTRATIVE)

# Decider "is this complex?" probability at/above which we route to the frontier model.
THRESHOLD = 0.50

COMPLEXITY_Q = {
    "complex": Decider.noul(
        "Is this request complex enough to need a frontier reasoning model?",
        {
            "true": "multi-step reasoning, design, proof, or deep analysis is required",
            "false": "a short factual or arithmetic answer is enough",
        },
    )
}


def load_dataset() -> list[dict]:
    rows = []
    for line in DATASET.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def cost(tier: str) -> float:
    return PRICE_PER_1K[tier] * (AVG_TOKENS_PER_QUERY / 1000.0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Model-routing benchmark with Strands Decider")
    parser.add_argument("--fallback", action="store_true",
                        help="use the offline heuristic (no server)")
    args = parser.parse_args()

    decider = Decider(allow_fallback=args.fallback)
    try:
        print(decider.banner(), "\n")
    except ServerUnavailable as exc:
        print(exc)
        return 1

    rows = load_dataset()
    correct = 0
    hybrid_cost = 0.0
    baseline_cost = 0.0
    latencies: list[float] = []
    using_fallback = False

    print(f"{'query (truncated)':<52} {'P(complex)':>11} {'routed':>9} {'gold':>9}")
    print("-" * 84)
    for row in rows:
        q = row["query"]
        gold = row["gold_tier"]
        answers = decider.ask(q, COMPLEXITY_Q)
        p = answers["complex"]["noul"]
        if answers["complex"].get("fallback"):
            using_fallback = True
        if decider.last_latency_ms is not None:
            latencies.append(decider.last_latency_ms)

        routed = "frontier" if p >= THRESHOLD else "cheap"
        if routed == gold:
            correct += 1
        hybrid_cost += cost(routed)
        baseline_cost += cost("frontier")

        mark = "OK" if routed == gold else "XX"
        print(f"{q[:50]:<52} {p:>11.2f} {routed:>9} {gold:>9}  {mark}")

    n = len(rows)
    acc = correct / n if n else 0.0
    saved = baseline_cost - hybrid_cost
    saved_pct = (saved / baseline_cost * 100.0) if baseline_cost else 0.0

    def pct(sorted_vals: list[float], q: float) -> float | None:
        if not sorted_vals:
            return None
        idx = min(len(sorted_vals) - 1, int(round(q * (len(sorted_vals) - 1))))
        return sorted_vals[idx]

    latencies.sort()
    p50 = pct(latencies, 0.50)
    p95 = pct(latencies, 0.95)

    print("\n" + "=" * 84)
    print(f"queries                : {n}")
    print(f"routing accuracy       : {acc:.0%} ({correct}/{n}) vs gold labels")
    if latencies:
        lat_tag = "  [FALLBACK heuristic, not the real model]" if using_fallback else ""
        print(f"decider latency p50/p95: {p50:.0f} / {p95:.0f} ms{lat_tag}")
    print()
    print("COST (illustrative prices -- verify against live Bedrock pricing):")
    print(f"  baseline  always-frontier : ${baseline_cost:.5f}")
    print(f"  hybrid    decider-routed  : ${hybrid_cost:.5f}")
    print(f"  saved                     : ${saved:.5f}  ({saved_pct:.0f}% cheaper)")
    print()
    if using_fallback:
        print("NOTE: numbers above used the OFFLINE HEURISTIC, not Strands Decider 2B. "
              "Start the server to get real classifications and real latency.")
    else:
        print("The decider itself runs locally on your hardware: $0 per-inference API "
              "cost, only electricity and the one-time download.")

    # ---- persist results -------------------------------------------------
    mode = "fallback-heuristic" if using_fallback else "strands-decider-2B (real)"
    env = _environment(decider, using_fallback)
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": mode,
        "environment": env,
        "dataset": {"file": str(DATASET.relative_to(HERE)), "queries": n},
        "policy": {"question": "is this complex?", "threshold": THRESHOLD},
        "pricing_assumptions": {
            "per_1k_usd": PRICE_PER_1K,
            "avg_tokens_per_query": AVG_TOKENS_PER_QUERY,
            "disclaimer": "ILLUSTRATIVE placeholders; verify against live Bedrock pricing.",
        },
        "results": {
            "routing_accuracy": round(acc, 4),
            "correct": correct,
            "latency_ms": {"p50": p50, "p95": p95, "samples": len(latencies)},
            "cost_usd": {
                "baseline_always_frontier": round(baseline_cost, 6),
                "hybrid_decider_routed": round(hybrid_cost, 6),
                "saved": round(saved, 6),
                "saved_pct": round(saved_pct, 1),
            },
        },
    }
    _write_results(result)
    print(f"\nwrote results -> {RESULTS_JSON.relative_to(HERE)} and "
          f"{BENCH_MD.relative_to(HERE)}")
    return 0


def _environment(decider: "Decider", using_fallback: bool) -> dict:
    """Capture host + model metadata so a result is interpretable later."""
    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    if not using_fallback:
        try:
            h = decider.health()
            env["decider"] = {
                "checkpoint": h.get("checkpoint"),
                "base_model": h.get("base_model"),
                "device": h.get("device"),
                "max_length": h.get("max_length"),
                "url": decider.url,
            }
        except Exception:  # noqa: BLE001 - health is best-effort metadata
            pass
    return env


def _write_results(result: dict) -> None:
    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.write_text(json.dumps(result, indent=2) + "\n")
    BENCH_MD.parent.mkdir(parents=True, exist_ok=True)
    BENCH_MD.write_text(_render_markdown(result))


def _render_markdown(r: dict) -> str:
    res = r["results"]
    lat = res["latency_ms"]
    cost_ = res["cost_usd"]
    p50 = f"{lat['p50']:.0f}" if lat["p50"] is not None else "—"
    p95 = f"{lat['p95']:.0f}" if lat["p95"] is not None else "—"
    env = r["environment"]
    pa = r["pricing_assumptions"]["per_1k_usd"]
    price_line = ", ".join(f"{k} ${v}" for k, v in pa.items())
    dec = env.get("decider", {})
    device = dec.get("device", "—")
    checkpoint = dec.get("checkpoint", "—")
    real = r["mode"].startswith("strands-decider")
    warn = "" if real else (
        "\n> ⚠️ These numbers came from the **offline heuristic fallback**, "
        "**not** the real model. Latency is meaningless here. Start the server "
        "and re-run for real figures.\n"
    )
    return f"""# Benchmark results — model routing with Strands Decider

Auto-generated by `bench.py`. Do not hand-edit; re-run the benchmark instead.
See the protocol and interpretation notes at the bottom.
{warn}
- **Generated:** {r['generated_at']}
- **Mode:** {r['mode']}
- **Checkpoint / device:** {checkpoint} / {device}
- **Host:** {env['platform']} · Python {env['python']}
- **Dataset:** `{r['dataset']['file']}` ({r['dataset']['queries']} queries)
- **Policy:** `noul` "{r['policy']['question']}", route to frontier when P ≥ {r['policy']['threshold']}

## Results

| Metric | Value |
| --- | --- |
| Routing accuracy vs gold | {res['routing_accuracy']:.0%} ({res['correct']}/{r['dataset']['queries']}) |
| Decider latency p50 / p95 | {p50} / {p95} ms |
| Latency samples | {lat['samples']} |

## Cost (illustrative — verify against live Bedrock pricing)

Assumptions: {price_line} per 1K tokens,
~{r['pricing_assumptions']['avg_tokens_per_query']} tokens/query.
**{r['pricing_assumptions']['disclaimer']}**

| Policy | Cost (USD) |
| --- | ---: |
| Baseline — always frontier | ${cost_['baseline_always_frontier']:.5f} |
| Hybrid — decider-routed | ${cost_['hybrid_decider_routed']:.5f} |
| **Saved** | **${cost_['saved']:.5f} ({cost_['saved_pct']:.0f}% cheaper)** |

The decider runs locally: **$0 per-inference API cost**. The saving above is the
frontier-LLM spend avoided by sending trivial queries to a cheap model instead.

## Protocol (reproducible)

1. Start the model server:
   `strands-decider serve StrandsAgents/strands-decider-2B-hobson-v19 --port 8099`
2. Run `python bench.py`. One `noul` question per query; the policy threshold is
   owned by the code, not the model.
3. The dataset ships hand-labelled gold tiers (`cheap` / `frontier`). Accuracy is
   routing agreement with those labels, not answer quality.
4. Pin the checkpoint, device, host and pricing snapshot (captured above) with any
   figure you cite.

## Honesty notes

- **Measured here:** routing accuracy, decider latency, and the arithmetic cost delta
  under the stated price assumptions.
- **NOT measured here:** the frontier/cheap LLMs' own latency or answer quality — this
  benchmark isolates the *routing* decision. Prices are placeholders.
- Fill real numbers only by re-running; empty beats fabricated.
"""


if __name__ == "__main__":
    sys.exit(main())
