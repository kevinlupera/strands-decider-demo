# Strands Decider 2B — a demo of real use cases, metrics and trade-offs

A runnable companion to the launch post
[*Introducing Strands Decider 2B*](https://strandsagents.com/blog/introducing-strands-decider/).
It shows **why** you'd reach for a decision model, with two concrete use cases and the
cost/latency numbers that justify them.

> Focused, runnable example — not a framework.

## What Strands Decider is (and is not)

Strands Decider 2B is a **decision model** (a "system one" model) from AWS Strands Labs.
It is a ~1.9B-parameter model that **does not generate text**. You give it a `state` plus
one or more *typed questions* and it returns a **calibrated probability** over each
question's options, in a single forward pass:

- `noul` — yes/no. The answer is `P(true)`, which *is* the confidence.
- `choice` — pick one of N named options.
- `score` — rate on an ordered rubric.

| It is good for | It is **not** for |
|---|---|
| routing, tool selection, guardrails, triage, evals, policy classification | coding, chatbots, summarization, anything that must emit text |
| cheap, repeated, local judgements with a confidence score | hard multi-step reasoning (that's still a frontier LLM's job) |

The headline property, which frontier LLM inference APIs do **not** expose: **every
decision carries a calibrated reliability score**. You branch on a number, not on prose
the model can argue you out of.

## The two demos

### 1. Tool-call guardrail (`guardrail_demo.py`) — the launch-post scenario

An eager agent is asked *"What's the weather?"* with no city. It guesses and proposes
`get_weather(location="Seattle")`. Before that call runs, the decider answers two yes/no
questions — *are the arguments grounded in what the user said?* and *is this premature?* —
and a few lines of plain Python turn those into a typed action. `Guide` hands the model
its turn back to **ask** the user instead of inventing a city.

The seam is Strands' native intervention system: an `InterventionHandler.before_tool_call`
passed to `Agent(interventions=[...])`. Verified against the official example
([strands-labs/strands-decider, Apache-2.0](https://github.com/strands-labs/strands-decider/tree/main/examples/strands)).

```bash
python guardrail_demo.py            # real: local decider + Bedrock-driven agent
python guardrail_demo.py --fallback # pattern only, no model/Bedrock needed
```

### 2. Model routing + cost benchmark (`bench.py`)

One `noul` question per query — *"is this complex?"* — routes trivial queries to a cheap
model and hard ones to a frontier model, then compares the hybrid policy against
"always frontier" on a small labelled dataset.

```bash
python bench.py                 # real local classifications
python bench.py --fallback      # offline heuristic, runs anywhere
```

Each run **persists its results** to `results/latest.json` and regenerates
[`docs/benchmark.md`](docs/benchmark.md) — a human-readable report with the metrics,
host/checkpoint metadata, the reproducible protocol, and explicit honesty notes about
what is and isn't measured. The committed `docs/benchmark.md` is whatever was last
generated; re-run on your hardware (with the server up) to get real latency figures.

## Running it for real

```bash
pip install -r requirements.txt            # or: pip install "strands-decider[mlx]" on Apple silicon
strands-decider serve StrandsAgents/strands-decider-2B-hobson-v19 --port 8099
python guardrail_demo.py
python bench.py
```

Both scripts also run with `--fallback`, which uses a **transparent, deterministic
keyword heuristic** when no server is reachable. A fallback answer is clearly labelled and
is **not** a real decision-model classification — it only keeps the demo runnable and lets
you show the pattern and the cost math on any machine. (Same convention as
`jev-log-sentinel`'s offline fallback.)

> Heads-up: this machine has Python 3.14. PyTorch wheels can lag new Python releases, so
> the real `strands-decider` install may require a slightly older Python (3.11–3.12) in a
> venv. The `--fallback` path has no such dependency.

## Why use a decision model here — the trade-offs

**The argument is latency + cost + a confidence number, not accuracy.** A decision model
won't out-reason a frontier LLM. It wins when the decision is small, frequent, and sits on
a hot path where an LLM call is too slow or too expensive to justify.

- **Latency.** Official v19 figures: median **115 ms** on an RTX 3090, **153 ms** on an
  M3 Pro (warm, <300 tokens). That's a gate you can put *before every tool call*.
- **Cost.** The model runs **locally** → **$0 per-inference API cost** (just your hardware
  and a one-time download). A frontier LLM "self-check" on every tool call would cost real
  money and add seconds.
- **Calibration.** Reliability score on every decision (JevBench ECE ≈ **0.052**); below
  your confidence bar, escalate to a human or an LLM.
- **Honesty.** A gate asking an LLM "are *your own* arguments grounded?" can be talked into
  a "no". A separate classifier can't be negotiated with.

### Decider-local vs hosted decision model (Jev) vs frontier LLM

| | **Strands Decider 2B** (local) | **Jev** (hosted, TypeSafe AI) | **Frontier LLM** as a gate |
|---|---|---|---|
| Runs | your CPU/GPU/Mac | hosted API | hosted API |
| Per-inference cost | **$0** (your hardware) | per-call API price | per-call API price (highest) |
| Latency | ~115–153 ms local | network + inference | seconds |
| Confidence score | ✅ calibrated | ✅ calibrated | ❌ not exposed |
| Generates text | ❌ | ❌ | ✅ (overkill for a gate) |
| Ops burden | you host it | none | none |
| Data locality | **never leaves machine** | leaves network | leaves network |
| Strands integration | native `InterventionHandler` + example | generic HTTP | native but expensive |

**Rule of thumb:** local decider when data locality / zero per-call cost / self-hosting
are worth the ops; hosted decider (Jev) when you want the same typed-decision shape with
no infrastructure; frontier LLM only for the decisions that genuinely need reasoning.

### Costs (illustrative — verify before citing)

`bench.py` prints a cost comparison using **editable, illustrative** per-1K-token Bedrock
estimates in the script. They **will** go stale. Before quoting any dollar figure, check
the live [Amazon Bedrock pricing](https://aws.amazon.com/bedrock/pricing/) page for your
region and models. The decider's own inference is local, so it adds **$0** of API cost to
the routing path.

## Measured vs illustrative — what to trust

| Claim | Source |
|---|---|
| 115 ms / 153 ms latency, ECE 0.052, 72.3% JevBench accuracy | official repo `evaluation/` (v19) |
| API shape (`/v1/systemone`, `noul`/`choice`/`score`), `InterventionHandler` signature | official repo + Strands SDK docs |
| the `--fallback` numbers | this repo's hand-rolled heuristic — **not the model** |
| the dollar figures in `bench.py` | **illustrative placeholders** — verify against live pricing |

## Files

```
decider_client.py   HTTP client for the local server (based on the official _client.py) + labelled fallback
guardrail_demo.py   use case 1 — InterventionHandler tool-call gate
bench.py            use case 2 — model routing + cost/latency benchmark
data/queries.jsonl  tiny labelled dataset for the benchmark
docs/benchmark.md   auto-generated benchmark report (re-run bench.py to refresh)
results/latest.json machine-readable benchmark output (regenerable; gitignored)
```

## License / credit

The client and the guardrail are adapted from the official
[strands-labs/strands-decider](https://github.com/strands-labs/strands-decider) examples
(Apache-2.0).
