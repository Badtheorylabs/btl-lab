# Experiment: <stable ID and descriptive title>

- Project ID and primary program:
- Accountable maintainer:
- Lifecycle: proposed / active / paused / completed / archived
- Evidence status: untested / diagnostic / measured / reproduced / released
- Visibility and ownership:
- Last reviewed date:

## Decision and hypothesis

What decision will this experiment resolve? What result would disprove the hypothesis?

## Frozen comparison

Baseline and candidate source commits; upstream attribution/license; model revision and checkpoint hashes; dataset manifest and split hashes; environment/verifier version; hardware and topology; software image; precision; context and batch shape; optimizer; decoding; seeds; equal tuning budget. Identify the single changed variable and any unavoidable differences.

## Execution and budget

Exact command/config and working directory. Local preflight. Authorized accelerator/API/sandbox spend cap and authorization reference. Startup, compilation, steady-state, evaluation and checkpoint accounting. Stop conditions. Do not interpret this template as spend authorization.

## Acceptance gate

Correctness and behavioral quality requirements; primary metric and denominator; workload distribution; comparison budget; variability and repetition policy. Separate microbenchmark gains, end-to-end system speed and learning efficiency.

## Result and evidence

Run IDs, commands actually executed, logs, raw measurements, profiles, artifact hashes, checkpoint/reload evidence, quality checks, actual total spend, failures and exclusions. Link large artifacts rather than committing weights. Preserve negative outcomes.

## Decision

Accept / reject / inconclusive, with reasons and claim limits. Next gate, owner and canonical implementation path. Release status and reviewed public evidence, if any.
