# Qwen3.6 vs Qwen3.5: Material Differences Only

> **Research question:** Does Qwen3.6 materially change the Qwen3.5
> architecture, and what actually changed in architecture, data flow,
> pre-training, and post-training?
>
> **Research date:** 2026-07-20. This report compares the open-weight
> Qwen3.6-27B and Qwen3.6-35B-A3B checkpoints with their same-size Qwen3.5
> counterparts. Hosted Plus/Flash/Max variants are excluded from architectural
> claims because their internal configs are not public.
>
> **Baseline:** For the full unchanged architecture and training background, see
> [Qwen3.5: Architecture, Data Flow, Pre-training, and Post-training](qwen3.5_architecture_and_training.md).

## Short answer

**Qwen3.6 is not a materially new architecture.** For both open same-size pairs,
the official model overviews, configs, tokenizer artifacts, and checkpoint
structure retain the Qwen3.5 design. Qwen3.6 still loads through the
`Qwen3_5ForConditionalGeneration` or `Qwen3_5MoeForConditionalGeneration`
classes. The image/video path, early fusion, 3:1 Gated DeltaNet/full-attention
decoder, dense FFN or sparse MoE, multimodal RoPE, MTP, and 262,144-token native
context remain unchanged.

The large differences are instead:

1. **targeted model behavior:** much stronger agentic coding, terminal work,
   repository-level reasoning, skills, and front-end generation;
2. **thinking preservation:** an opt-in prompt path keeps earlier assistant
   reasoning traces across user turns, and Qwen says 3.6 was additionally
   trained to use those traces;
3. **release scope:** only 27B dense and 35B-A3B MoE open checkpoints, rather
   than the broad 0.8B-to-397B Qwen3.5 family.

Public sources do not disclose enough information to reconstruct a distinct
Qwen3.6 pre-training or post-training recipe. It is safe to say that the weights
and capability emphasis changed. It is **not** safe to claim a particular new
corpus, SFT curriculum, RL algorithm, reward, or distillation method.

## Architecture: effectively unchanged

The official model-card architecture tables are identical within each same-size
pair. Direct config comparison gives the same conclusion.

| Component                  | Qwen3.5-27B -> Qwen3.6-27B                                  | Qwen3.5-35B-A3B -> Qwen3.6-35B-A3B                     |
| -------------------------- | ----------------------------------------------------------- | ------------------------------------------------------ |
| Runtime architecture class | `qwen3_5`, unchanged                                      | `qwen3_5_moe`, unchanged                             |
| Decoder layout             | 64 layers;`16 x (3 GDN + 1 full attention)`               | 40 layers;`10 x (3 GDN + 1 full attention)`          |
| Hidden width               | 5,120                                                       | 2,048                                                  |
| Attention                  | 24 Q / 4 KV heads; head dim 256                             | 16 Q / 2 KV heads; head dim 256                        |
| Gated DeltaNet             | 48 V / 16 QK heads; head dim 128                            | 32 V / 16 QK heads; head dim 128                       |
| FFN or MoE                 | dense FFN, intermediate 17,408                              | 256 routed experts; top-8 + 1 shared; expert width 512 |
| Vision encoder             | same 27-layer, width-1,152 encoder and patch/merge settings | same                                                   |
| Tokenizer and vocabulary   | same tokenizer artifacts; padded vocabulary 248,320         | same                                                   |
| Position and context       | same multimodal RoPE; native 262,144                        | same                                                   |
| MTP                        | one MTP layer, trained with multiple steps                  | same topology                                          |

Sources: [Qwen3.5-27B model card][qwen35-27b],
[Qwen3.6-27B model card][qwen36-27b],
[Qwen3.5-35B-A3B model card][qwen35-35b], and
[Qwen3.6-35B-A3B model card][qwen36-35b]. The executable configs likewise keep
the Qwen3.5 model types and layer layouts:
[3.5-27B config][qwen35-27b-config], [3.6-27B config][qwen36-27b-config],
[3.5-35B config][qwen35-35b-config], and [3.6-35B config][qwen36-35b-config].

There are minor artifact differences, but they do not establish a new model
topology. Qwen3.6 makes several formerly implicit config defaults explicit. Its
35B MTP expert weights are packed into fused tensors rather than serialized as
individual expert tensors, while the MTP/MoE dimensions and total parameter
count remain unchanged. This is a checkpoint/runtime packaging difference, not
evidence of a new MTP algorithm.

The practical conclusion is simple: there is no new module or revised neural
data path worth re-documenting. Qwen's own repository describes 3.6 as building
on Qwen3.5 and sends deployment readers back to Qwen3.5 serving recipes.
[Qwen3.6 repository][qwen36-repo]

## Material change 1: thinking traces can cross user-turn boundaries

Qwen3.5 already supported thinking and could retain reasoning inside the current
multi-step tool loop. Its default chat template removed reasoning from older
assistant turns when constructing a later prompt. Qwen3.6 adds this condition:

```text
preserve_thinking == true
    -> serialize historical assistant reasoning as <think>...</think>
preserve_thinking absent or false
    -> retain the previous Qwen3.5-style behavior
```

The resulting input path is:

```text
stored messages
  -> split assistant reasoning_content from final content
  -> chat template
       default: drop old-turn reasoning
       preserve_thinking: keep old-turn <think> blocks
  -> tokenize the longer conversation
  -> unchanged Qwen3.5 multimodal decoder
  -> next response or tool call
```

The material template change for reasoning is an added OR condition around
historical trace retention. Compare the [Qwen3.5 template][qwen35-template] with the
[Qwen3.6 template][qwen36-template]. Qwen also explicitly states that 3.6 was
“additionally trained to preserve and leverage” historical thinking traces.
[Qwen3.6-27B Preserve Thinking section][qwen36-27b]

This distinction matters:

- it is **not** recurrent memory outside the context window;
- it does **not** add an attention or cache module;
- previous traces consume context tokens when serialized;
- it can nevertheless avoid recomputing the same reasoning and make a stable
  prefix reusable in the KV cache, which is particularly useful in long agent
  sessions.

Therefore, thinking preservation is both a small serving/template change and a
real post-training capability change. Enabling the flag on Qwen3.5 would expose
old traces to an input distribution for which Qwen has not claimed equivalent
training.

## Material change 2: training emphasis shifted to coding agents

Official sources consistently describe Qwen3.6 as a stability and real-world
utility update centered on front-end workflows, repository-level reasoning, and
iterative agent work. The strongest evidence is the matched same-size benchmark comparison published in the 3.6 model cards.

| Benchmark              | 27B: Qwen3.5 -> Qwen3.6 | Raw delta | 35B-A3B: Qwen3.5 -> Qwen3.6 | Raw delta |
| ---------------------- | ----------------------: | --------: | --------------------------: | --------: |
| SWE-bench Verified     |            75.0 -> 77.2 |      +2.2 |                70.0 -> 73.4 |      +3.4 |
| SWE-bench Pro          |            51.2 -> 53.5 |      +2.3 |                44.6 -> 49.5 |      +4.9 |
| SWE-bench Multilingual |            69.3 -> 71.3 |      +2.0 |                60.3 -> 67.2 |      +6.9 |
| Terminal-Bench 2.0     |            41.6 -> 59.3 |     +17.7 |                40.5 -> 51.5 |     +11.0 |
| SkillsBench Avg5       |            27.2 -> 48.2 |     +21.0 |                 4.4 -> 28.7 |     +24.3 |
| NL2Repo                |            27.3 -> 36.2 |      +8.9 |                20.5 -> 29.4 |      +8.9 |
| QwenWebBench           |          1,068 -> 1,487 |      +419 |                978 -> 1,397 |      +419 |

Sources and evaluation notes: [Qwen3.6-27B benchmark table][qwen36-27b] and
[Qwen3.6-35B-A3B benchmark table][qwen36-35b]. Raw deltas should be interpreted
within each benchmark only; their scales are not interchangeable. QwenWebBench
is an internal benchmark, and several agent evaluations use Qwen's stated agent
scaffolds and resource settings, so these numbers are directional evidence, not
independent proof of deployment performance.

The gain is not uniform across all capabilities. For example, Qwen3.6-27B is
slightly lower than Qwen3.5-27B on MathVista (87.4 vs 87.8) and DynaMath (85.6 vs
87.7), while many other vision scores move by less than one point. The
35B-A3B model also regresses slightly on Claw-Eval Pass^3 (50.0 vs 51.0).
This pattern is consistent with a targeted coding/agent update rather than a
general architectural leap.

## What changed in pre-training and post-training?

### Verified

- The released artifacts are post-trained checkpoints, and their cards label
  the overall training stage as “Pre-training & Post-training.”
- Qwen3.6 has new weights while retaining the same architecture.
- Qwen states that the models were additionally trained to preserve and use
  historical thinking traces.
- Official release language and the benchmark pattern identify agentic coding, front-end generation, repository reasoning, tool use, and stability as the main capability targets.

### Not publicly disclosed

No Qwen3.6 technical report or complete training recipe was available at the
research date. The release posts, repository, model cards, configs, and templates
do not state:

- whether 3.6 began from Qwen3.5 weights through continual pre-training, or how
  much pre-training was repeated;
- new pre-training token count, corpus sources, code/multimodal mixture, or data
  cutoff;
- SFT stages, dataset composition, trajectory count, or rejection-sampling
  procedure;
- RL algorithm, reward functions, judges, environments, curriculum, or compute;
- distillation, safety alignment, or evaluation-contamination controls.

Consequently, “mostly post-training” is a plausible interpretation of the short
release interval, unchanged architecture, explicitly added thinking-trace
training, and concentrated agent gains, but it is still an **inference**, not a
published recipe. The official cards continue to say both pre-training and
post-training, so this report does not claim that pre-training was absent.

## Release scope also changed

As of the research date, the official Qwen3.6 Hugging Face collection contains
only two open model topologies—27B dense and 35B-A3B MoE—plus their FP8 copies.
[Qwen3.6 collection][qwen36-collection] Qwen3.5 was released across 0.8B, 2B, 4B,
9B, 27B, 35B-A3B, 122B-A10B, and 397B-A17B variants. Thus Qwen3.6 is better
understood as a focused refresh of two practical checkpoints than as a complete
replacement for the Qwen3.5 family.

Hosted Qwen3.6 Plus/Flash/Max names should not be used to infer hidden parameter
counts or architecture. Their public product behavior may differ, but there are
no inspectable same-size configs that support an architecture-level comparison.

## Bottom line

For architecture and module-level data flow, continue using the Qwen3.5 report:
**there is no substantial Qwen3.6 delta to document.** The meaningful 3.6 work
is a weight/training update for coding agents plus an opt-in way to carry
historical reasoning through the prompt. The reported gains are largest exactly
where Qwen says it focused—terminal tasks, repository work, skills, and web/front-
end generation—while vision and general reasoning are mostly incremental or
mixed.

## Sources

All sources below are official Qwen repositories, model artifacts, or release
pages, accessed 2026-07-20.

- [Qwen3.6 official repository][qwen36-repo]
- [Qwen3.6-27B model card and benchmark methodology][qwen36-27b]
- [Qwen3.6-35B-A3B model card and benchmark methodology][qwen36-35b]
- [Qwen3.6 official model collection][qwen36-collection]
- [Qwen3.5-27B model card][qwen35-27b]
- [Qwen3.5-35B-A3B model card][qwen35-35b]
- [Qwen3.5 and Qwen3.6 chat templates][qwen35-template]
  / [Qwen3.6 template][qwen36-template]
- [Immutable same-size config artifacts][qwen35-27b-config]
  / [3.6-27B][qwen36-27b-config]
  / [3.5-35B-A3B][qwen35-35b-config]
  / [3.6-35B-A3B][qwen36-35b-config]

[qwen36-repo]: https://github.com/QwenLM/Qwen3.6
[qwen36-collection]: https://huggingface.co/collections/Qwen/qwen36
[qwen35-27b]: https://huggingface.co/Qwen/Qwen3.5-27B
[qwen36-27b]: https://huggingface.co/Qwen/Qwen3.6-27B
[qwen35-35b]: https://huggingface.co/Qwen/Qwen3.5-35B-A3B
[qwen36-35b]: https://huggingface.co/Qwen/Qwen3.6-35B-A3B
[qwen35-template]: https://huggingface.co/Qwen/Qwen3.5-27B/blob/fc05daec18b0a78c049392ed2e771dde82bdf654/chat_template.jinja
[qwen36-template]: https://huggingface.co/Qwen/Qwen3.6-27B/blob/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9/chat_template.jinja
[qwen35-27b-config]: https://huggingface.co/Qwen/Qwen3.5-27B/blob/fc05daec18b0a78c049392ed2e771dde82bdf654/config.json
[qwen36-27b-config]: https://huggingface.co/Qwen/Qwen3.6-27B/blob/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9/config.json
[qwen35-35b-config]: https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/59d61f3ce65a6d9863b86d2e96597125219dc754/config.json
[qwen36-35b-config]: https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/995ad96eacd98c81ed38be0c5b274b04031597b0/config.json
