# Evaluating Qwen Models

> **Research question:** Which benchmarks are useful for evaluating the Qwen
> family as a language, vision-language, and agent backbone, especially before
> using it inside a VLA system?
>
> **Scope:** Qwen3, Qwen3.5, Qwen3.6, and Qwen2.5-VL. Qwen-VLA's closed-loop
> policy evaluation is documented separately in
> [VLA/benchmarks.md](../VLA/benchmarks.md). Research checked on 2026-07-22.

## Short answer

There is no single meaningful "Qwen score." A useful evaluation has three
layers:

1. **language and reasoning** for instruction understanding, planning, and tool
   use;
2. **vision-language** for perception, OCR, grounding, spatial reasoning, and
   temporal understanding;
3. **task execution** for agents or robots, where an answer benchmark is no
   substitute for success in an environment.

For a robotics backbone, MMMU alone is insufficient. The minimum useful panel is
`MMMU-Pro + MathVista + OCRBench + RefCOCO/RefSpatialBench + VideoMME`, followed
by a closed-loop VLA suite. The Qwen3.5 model card reports all of these capability
families, while the Qwen-VLA paper shows that they still do not predict robot
success by themselves. [Qwen3.5 model card][qwen35-card]
[Qwen-VLA report, Sections 5 and 7][qwen-vla]

## What each benchmark family measures

| Capability | Representative benchmarks | What the score supports | What it does not establish |
|---|---|---|---|
| Knowledge | MMLU-Pro, MMLU-Redux, GPQA-Diamond | Broad academic knowledge and difficult QA | Grounded perception or action reliability |
| Instruction following | IFEval, IFBench, MultiChallenge | Constraint and format compliance | Physical feasibility or recovery from failure |
| Reasoning | MATH-500, AIME, HLE, MathVista | Text or visual reasoning under a defined answer protocol | Stable long-horizon control |
| Coding and agents | LiveCodeBench, SWE-bench, BFCL, Terminal-Bench | Code generation, repository repair, function calling, tool execution | General visual grounding or robotics |
| General VQA | MMMU/MMMU-Pro, MMStar, RealWorldQA | Multi-domain image understanding | Precise object localization and temporal control |
| OCR and documents | OCRBench, OmniDocBench, TextVQA | Text recognition and document reasoning | 3D geometry or manipulation |
| Spatial grounding | RefCOCO, RefSpatialBench, EmbSpatialBench, CountBench | Reference resolution, location, relations, and counting | Executable trajectory quality |
| Video | VideoMME, VideoMMMU, MLVU, MVBench | Temporal and long-video understanding | Closed-loop interaction with changing state |

The Qwen3 technical report explicitly separates general knowledge, alignment,
math/reasoning, coding/agents, and multilingual evaluation. Qwen2.5-VL and
Qwen3.5 add separate panels for general VQA, OCR/document understanding,
grounding, and video. These groupings are more informative than averaging all
scores into one number. [Qwen3 report, Section 4.6][qwen3]
[Qwen2.5-VL report, Section 4][qwen25-vl]

## Benchmark protocol boundaries

Metric formulas and interpretations are isolated in [metrics.md](metrics.md).
The protocol choices below determine what prediction reaches each metric and are
therefore part of the benchmark report, not the metric definition.

| Benchmark | Protocol that must be pinned | Current Qwen evidence |
|---|---|---|
| GPQA-Diamond | Prompt, answer parser, decoding and samples per question | Qwen3 draws 10 samples per question and averages their correctness; this is not `pass@10` |
| IFEval | Strict/loose evaluator and prompt/instruction aggregation | Qwen3 reports strict-prompt accuracy. Qwen3.5's bare `IFEval` label leaves the exact variant unresolved |
| OCRBench | Original/v2 release, official scorer and displayed scale | Original OCRBench totals 1,000 binary item points. Qwen2.5-VL reports raw totals such as 885; Qwen3.5 reports 89.4 without documenting whether this is exactly a `/10` normalization |
| RefCOCO | Split set, box parser and aggregation across val/testA/testB | Qwen3.5 reports `RefCOCO(avg)` but does not identify the constituent splits |
| Video-MME | Version, subtitle condition, frame/FPS sampler and answer-only parser | Qwen3.5 identifies with/without-subtitle rows but does not publish the complete frame policy |
| SWE-bench | Dataset revision, agent scaffold, tools, context, timeout and attempts | The reported value is a system result; model weights alone do not reproduce it |
| BFCL | Version, category set, handler/evaluator and category aggregation | Qwen3 uses BFCL v3 and Qwen3.5 uses BFCL-V4; the numbers are not directly comparable |

IFEval's official loose evaluator is not arbitrary fuzzy matching. It tests a
fixed set of output variants such as removing the first/last line or asterisks;
strict checks the unchanged response. [IFEval implementation][ifeval-code]
[Qwen3 evaluation settings][qwen3]

## Verified result snapshots

The tables below are **publisher-reported**, not reproduced locally. They are
included to show what conclusions the official protocols support, not to create a
universal leaderboard.

### Language and agent evolution

| Model and mode | MMLU-Pro | GPQA-Diamond | Coding/agent evidence | Supported interpretation |
|---|---:|---:|---|---|
| Qwen3-235B-A22B, thinking | not reported in the paper's Table 11 | 71.1 | BFCL v3 70.8; LiveCodeBench v5 70.7 | Strong reasoning/agent model under a long sampling budget |
| Qwen3.5-27B | 86.1 | 85.5 | SWE-bench Verified 72.4; BFCL-V4 68.5 | Much smaller native-multimodal model with strong language and agent scores |
| Qwen3.6-27B | 86.2 | 87.8 | SWE-bench Verified 77.2; Terminal-Bench 2.0 59.3 | Clearer gain in coding-agent execution than in general knowledge |

Sources: [Qwen3 report, Tables 11-12][qwen3],
[Qwen3.5-27B model card][qwen35-card], and
[Qwen3.6-27B model card][qwen36-card].

**Verified:** Qwen3 thinking and non-thinking results use different sampling
settings. Qwen3.6's coding-agent evaluations also depend on a specified scaffold,
context window, tool set, timeout, and repeated runs. Therefore, values should
only be compared when those settings match. [Qwen3 report, Section 4.6][qwen3]
[Qwen3.6 evaluation notes][qwen36-card]

### Vision-language snapshot

The official Qwen3.5-27B card reports the following selected scores:

| Capability | Benchmark | Qwen3.5-27B |
|---|---|---:|
| Expert multimodal reasoning | MMMU-Pro | 75.0 |
| Visual mathematics | MathVista-mini | 87.8 |
| Document/OCR | OCRBench | 89.4 |
| Referring/spatial | RefCOCO average | 90.9 |
| Embodied spatial reasoning | EmbSpatialBench | 84.5 |
| Long video, subtitles disabled | VideoMME | 82.8 |

These numbers support broad visual competence, but they are not action metrics.
A VLM can identify an object or answer a spatial question while still producing
unsafe, delayed, or dynamically inconsistent actions. [Qwen3.5 model card,
Vision Language table][qwen35-card]

## Recommended evaluation for this workspace

### Backbone gate

Use the same checkpoint, processor, image/video sampling policy, prompt template,
generation mode, and maximum output length for every run. Record:

- exact model revision and dtype/quantization;
- native image resolution and video frame/FPS policy;
- thinking or non-thinking mode and decoding parameters;
- benchmark version, split, metric, evaluator, and number of samples;
- latency, peak memory, and failures in addition to task score.

For early selection, run a compact panel:

| Priority | Benchmark type | Reason for a VLA project |
|---|---|---|
| P0 | RefCOCO or RefSpatialBench | Tests whether language refers to the correct object or region |
| P0 | VideoMME without subtitles | Tests temporal perception without leaking an answer through text |
| P0 | MathVista or EmbSpatialBench | Tests spatial and diagrammatic reasoning |
| P1 | OCRBench | Useful when robot scenes contain labels, displays, or signs |
| P1 | IFEval | Checks control-instruction compliance |
| P2 | MMLU-Pro/GPQA | Broad sanity check, but weakly coupled to robot control |

### Policy gate

Only promote a backbone after a closed-loop evaluation reports success rate,
generalization split, control frequency, action horizon, wall-clock latency, and
failure categories. See [VLA/benchmarks.md](../VLA/benchmarks.md).

## Limits and unknowns

- **Verified:** Official score tables mix public and internal benchmarks, and
  some agent results rely on internal scaffolds or modified task subsets.
- **Verified:** Qwen3 thinking/non-thinking, Qwen3.5, and Qwen3.6 tables do not
  all share the same decoding protocol or benchmark version.
- **Unknown:** The degree of training-data overlap for every public benchmark is
  not disclosed. A high score may combine generalization with memorization.
- **Inferred:** For VLA backbone selection, grounding and video scores are more
  task-relevant than a small difference in MMLU-Pro, but the final relationship
  must be measured through downstream policy evaluation.

## Sources

- Qwen Team. *Qwen3 Technical Report*, arXiv:2505.09388, 2025.
  [Paper][qwen3] · [Local PDF][qwen3-local]
- Qwen Team. *Qwen3.5-27B model card*, accessed 2026-07-22.
  [Model card][qwen35-card]
- Qwen Team. *Qwen3.6-27B model card*, accessed 2026-07-22.
  [Model card][qwen36-card]
- Bai et al. *Qwen2.5-VL Technical Report*, arXiv:2502.13923, 2025.
  [Paper][qwen25-vl] · [Local PDF][qwen25-vl-local]
- Wang et al. *Qwen-VLA*, arXiv:2605.30280v2, 2026.
  [Paper][qwen-vla]
- Zhou et al. *IFEval*. [Paper][ifeval] · [Official implementation][ifeval-code]

[qwen3]: https://arxiv.org/abs/2505.09388
[qwen3-local]: ../../../papers/05-gwen/gwen-overview/qwen3_technical_report_2505.09388.pdf
[qwen35-card]: https://huggingface.co/Qwen/Qwen3.5-27B
[qwen36-card]: https://huggingface.co/Qwen/Qwen3.6-27B
[qwen25-vl]: https://arxiv.org/abs/2502.13923
[qwen25-vl-local]: ../../../papers/05-gwen/gwen-overview/qwen2.5_vl_2502.13923.pdf
[qwen-vla]: https://arxiv.org/abs/2605.30280
[ifeval]: https://arxiv.org/abs/2311.07911
[ifeval-code]: https://github.com/google-research/google-research/blob/master/instruction_following_eval/evaluation_lib.py
