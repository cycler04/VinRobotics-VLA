# Metrics Used to Evaluate Qwen Models

> **Question:** What does each score in Qwen language, vision-language and
> agent evaluation mean?
>
> **Scope:** Score functions only. Dataset contents, scale, splits and licenses
> are in [datasets.md](datasets.md); prompts, preprocessing, evaluator versions
> and Qwen result tables are in [benchmarks.md](benchmarks.md); training
> objectives are in [loss.md](loss.md). Research checked on 2026-07-22.

## The boundary

- A **dataset** is the examples, modalities, annotations, splits and license.
- A **benchmark protocol** specifies which split, prompt, preprocessing and
  evaluator are used.
- A **metric** converts predictions into a score.

A score is comparable only when all three match. The same accuracy formula can
produce a different result after changing the answer parser or sampled frames;
that change belongs to the protocol, not to accuracy itself.

## Core classification and sampling metrics

For binary item correctness $c_i\in\{0,1\}$,

$$
\mathrm{Accuracy}=\frac{1}{N}\sum_{i=1}^{N}c_i.
$$

Higher is better. Multiple-choice suites such as MMLU-Pro, GPQA, MMMU-Pro,
MathVista and Video-MME commonly reduce to this form after their answer parser.
An aggregate must name its grouping axis:

- **micro/pooled accuracy** pools all items, so large groups receive more weight;
- **macro accuracy** averages group accuracies, so each named subject, domain or
  task receives equal weight.

Repeated sampling has two distinct summaries. Averaging correctness across
draws estimates expected single-draw accuracy. It is not `pass@k`. Given $n$
samples for a problem and $c$ correct samples, HumanEval's estimator is

$$
\widehat{\mathrm{pass@}k}
=1-\frac{\binom{n-c}{k}}{\binom{n}{k}},
$$

the probability that at least one of $k$ draws succeeds. `pass@k` normally rises
with $k$; mean sample correctness does not receive this any-success benefit.
[HumanEval paper and estimator][humaneval]

## Instruction-following aggregation

IFEval supplies several binary constraint checks per prompt:

$$
\mathrm{InstructionAcc}
=\frac{\text{passed constraints}}{\text{all constraints}},
\qquad
\mathrm{PromptAcc}
=\frac{1}{N}\sum_i\mathbf{1}[\text{all constraints for prompt }i\text{ pass}].
$$

Prompt accuracy is an AND aggregation and is therefore stricter for prompts
with multiple constraints. Strict versus loose checking changes the evaluator
and belongs to the benchmark protocol; the exact variants used by Qwen are
recorded in [benchmarks.md](benchmarks.md). [IFEval implementation][ifeval-code]

## Text, OCR and document-answer metrics

Generic free-form metrics are not interchangeable:

- **Exact match (EM)** is binary equality after a declared normalization.
- **Token F1** is the harmonic mean of token-overlap precision and recall.
- **ANLS** averages the best normalized edit similarity against the references,
  with low-similarity matches set to zero by a benchmark threshold.

ANLS gives partial spelling/edit credit; EM does not. Token F1 ignores token
order and may reward irrelevant overlap. Normalization, multiple-reference
handling and the ANLS threshold are part of the scorer contract.
[ST-VQA ANLS][st-vqa]

Original **OCRBench does not use EM, token F1 or ANLS**. Each of its 1,000 items
receives 0/1 from the official substring-based scorer, then the hits are summed:

$$
\mathrm{OCRBenchRaw}=\sum_{i=1}^{1000}c_i,\qquad 0\leq score\leq1000.
$$

The five category maxima are 300 text recognition, 200 scene-text VQA, 200
document VQA, 200 key-information extraction and 100 handwritten-expression
recognition. Most categories use case-insensitive reference-substring matching;
the handwritten-expression branch removes spaces and preserves case. A displayed
0–100 value may be a normalization, but conversion is valid only when the same
1,000 items and scorer are confirmed. [OCRBench scorer][ocrbench-code]

## Grounding metrics

For predicted box $B_p$ and reference box $B_g$,

$$
\mathrm{IoU}(B_p,B_g)=\frac{|B_p\cap B_g|}{|B_p\cup B_g|}.
$$

RefCOCO referring-expression comprehension commonly reports

$$
\mathrm{Acc@0.5}=\frac{1}{N}\sum_i
\mathbf{1}[\mathrm{IoU}(B_{p,i},B_{g,i})>0.5].
$$

This is thresholded grounding accuracy, not mean IoU and not detection mAP.
It says how often the referred object was localized sufficiently well, but not
how tight the successful boxes were. [RefCOCO paper][refcoco]

## Agent and tool-use metrics

**SWE-bench resolved rate** is the fraction of attempted repository instances
fully resolved by the harness:

$$
\mathrm{ResolvedRate}=\frac{\text{instances marked fully resolved}}
{\text{attempted instances}}.
$$

The current official grader requires all `FAIL_TO_PASS` and `PASS_TO_PASS` tests
to pass for full resolution; partial test repair is not resolved. The score is a
binary system outcome and does not expose patch quality, cost or time.
[SWE-bench grader][swebench-grader]

**BFCL accuracy** is versioned evaluator pass rate over function-calling cases.
The pass predicate can include function name, arguments, type/value, execution,
irrelevance and multi-turn state checks. Composite aggregation changed across
BFCL releases; therefore `BFCL v3` and `BFCL-V4` are different metric contracts,
not two measurements on a stable axis. [BFCL evaluation][bfcl-v1]
[BFCL V4][bfcl-v4]

**Elo** summarizes relative pairwise preference against a particular opponent
pool. **LLM-judge scores** summarize decisions from a particular judge, prompt
and rubric. Neither is an absolute probability of factual correctness.

## Efficiency and uncertainty

Latency, generated-token count, peak memory, throughput and monetary cost are
separate efficiency metrics. Report distributions such as median and p95
latency, plus timeout/error rate; do not average them into task accuracy without
an explicit utility function.

For $N$ binary items, always report $N$ and preferably a binomial confidence
interval. Repeated generations from the same prompt and subject-grouped items
are not fully independent; per-seed results or bootstrap intervals over the
appropriate item/group unit are more honest than extra decimal places.

## Metric reporting checklist

For every number, record:

```text
metric name and direction (higher/lower is better)
formula or exact evaluator revision
range/display scale and aggregation axis
dataset version and evaluated split
prompt, preprocessing, answer parser and decoding
sample count, seeds, denominator and confidence interval
```

The first three lines define the metric; the remaining lines bind it to a
reproducible benchmark protocol.

## Sources

- Chen et al. *Evaluating Large Language Models Trained on Code*.
  [Paper and estimator][humaneval]
- Zhou et al. *IFEval*. [Official implementation][ifeval-code]
- Biten et al. *Scene Text Visual Question Answering*. [Paper][st-vqa]
- Liu et al. *OCRBench*. [Official scorer][ocrbench-code]
- Yu et al. *Modeling Context in Referring Expressions*. [Paper][refcoco]
- SWE-bench Team. [Official grading code][swebench-grader]
- Berkeley Function Calling Leaderboard. [V1 metrics][bfcl-v1] ·
  [V4 composition][bfcl-v4]

[humaneval]: https://arxiv.org/abs/2107.03374
[ifeval-code]: https://github.com/google-research/google-research/blob/master/instruction_following_eval/evaluation_lib.py
[st-vqa]: https://arxiv.org/abs/1907.00490
[ocrbench-code]: https://github.com/qywh2023/OCRbench/blob/main/OCRBench/example.py
[refcoco]: https://arxiv.org/abs/1608.00272
[swebench-grader]: https://github.com/SWE-bench/SWE-bench/blob/master/swebench/harness/grading.py
[bfcl-v1]: https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html#evaluation-metrics
[bfcl-v4]: https://gorilla.cs.berkeley.edu/blogs/15_bfcl_v4_web_search.html
