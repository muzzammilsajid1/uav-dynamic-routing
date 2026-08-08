# Phase C2 v14 Confirmatory Queue Repair

**Incident date:** 2026-08-08

## What Happened

Confirmatory seed 11 completed its full frozen run at 151,552 interactions and
achieved 240/240 validation success. Its model, evaluation, generator state,
RNG state, provenance, status, recovery bundle, raw archive, and complete
archive were all successfully written.

After artifact creation, the orchestration queue attempted to write its own
status JSON. `inspect_seed()` returned the recovery-bundle path as a Python
`WindowsPath`, which the standard JSON encoder cannot serialize. The queue
process stopped before launching seed 22.

This was an orchestration-status defect, not a training, evaluation, model,
resume, provenance, or archive defect. Seed 11 is complete and must not be
rerun.

Seed 11 latest bundle SHA-256:

`0a4a0ca55a81c7d43870b29f8496d61d0b0a10fba217e3e7c53d52158315558b`

## Repair

- Queue state now stores bundle paths as strings.
- All queue records are explicitly JSON-serializable before persistence.
- A regression test round-trips a completed seed record through JSON.
- When a completed seed comes from the immediately preceding queue-only commit,
  the repaired queue compares every one of the ten actual training/evaluation
  source hashes against the current files. It may skip the seed only if all ten
  match exactly and the completed model, evaluation, status, provenance, and
  recovery-bundle inventory verify.
- The confirmatory queue file itself is deliberately excluded from that
  equivalence comparison because it does not affect environment transitions,
  model initialization, optimization, curriculum, reward, action masking, or
  validation. Its distinct hash and commit remain recorded transparently.

The repaired logic was exercised against the real completed seed-11 artifact:
it verified completion, recognized a queue-only commit difference, rehashed the
bundle, and round-tripped the resulting queue record through atomic JSON.

## Scientific Treatment

Seed 11 remains valid v13 evidence. Seeds 22, 33, 44, and 55 will run under the
v14 queue repair. The underlying ten training/evaluation sources are identical
across v13 and v14; only orchestration status serialization changed. Reports
must preserve both exact commits rather than pretending all artifacts came from
one tag.
