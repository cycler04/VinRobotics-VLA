---
name: research-paper
description: Read and evaluate research papers or long technical documents with traceable evidence and a problem-first Why then How structure. Use when the user asks to read a paper, analyze a PDF, explain a method, compare papers, connect paper claims to code, review related work, or create a research note covering the problem, modeling, training, or benchmarks.
---

# Research Paper

Read [the paper-reading workflow](../../workflows/01_read_paper.md) completely
and follow it.

Organize the synthesis as `Why` before `How`. In `How`, explain modeling,
training, and benchmarks through the specific problem or failure mode each one
addresses; do not present them as an isolated inventory of components.

Write `How` as a numbered `Method` narrative that follows the system data flow,
matching the established reports in `docs/research_finding/`. For every module
or stage, state the failure mode it addresses before explaining its input,
mechanism, output, and evidence. Do not use a traceability table as the primary
presentation of `Method`; reserve tables for evidence, benchmarks, ablations, or
multi-configuration comparisons.

Keep training as a separate `## 5. Training` section after `## 4. Method` when
the paper trains or post-trains any component. Put datasets, objectives, stages,
frozen/updated modules, hyperparameters, and compute there; do not fold training
configuration back into `Method`.

Treat that layout as a default, not a rigid template. When a paper contributes
multiple first-class artifacts such as a dataset, benchmark, model, and deployed
system, promote each artifact to its own main section. Keep collection details
under the dataset, protocol and metrics under the benchmark, and architecture
and training under the model instead of forcing the paper into generic `Method`
and `Training` sections.

For long PDFs or broad literature searches, isolate extraction in a subagent and
return only page/section references, claims, evidence, and unresolved questions
to the main context. Keep the final synthesis in the main task.

Read [the workspace overview](../../01_overview.md) before relating findings to
this workspace.
