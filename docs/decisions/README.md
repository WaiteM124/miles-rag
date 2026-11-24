# Architecture decision records

Lightweight records of significant technical choices in this repository.

## Format

Each record is a markdown file named `ADR-NNN-short-title.md` with:

| Section | Content |
|---|---|
| **Status** | Accepted / Superseded / Proposed |
| **Date** | When the decision was made (or reconstructed) |
| **Context** | Problem and constraints |
| **Decision** | What we chose |
| **Alternatives rejected** | Options considered and why they lost |
| **Consequences** | Trade-offs, follow-ups, what becomes harder |

Mark records **reconstructed from code** when the original reasoning was never written down and the ADR was inferred from the current implementation and README.

## Index

| ADR | Title |
|---|---|
| [ADR-001](ADR-001-hybrid-retrieval-rerank.md) | Hybrid BM25 + dense retrieval with RRF and cross-encoder re-rank |
| [ADR-002](ADR-002-threshold-gated-refusal.md) | Threshold-gated structured refusal before generation |
| [ADR-003](ADR-003-offline-evaluation-harness.md) | Offline-first evaluation harness (no API key by default) |
