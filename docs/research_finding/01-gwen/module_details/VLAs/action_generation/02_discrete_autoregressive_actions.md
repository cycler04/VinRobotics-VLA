# Discrete Autoregressive Action Generation

> **Scope.** VLA policies that serialize robot actions into discrete symbols
> and generate them with the VLM's autoregressive next-token machinery.
> Representative systems: RT-2, OpenVLA, and π0-FAST. Sources checked
> 2026-07-21.

## Core idea

This family converts continuous control targets into a finite vocabulary, then
trains the same causal language-model interface used for text:

```text
continuous action or chunk
        -> tokenizer
        -> [z1, z2, ..., zK]
        -> next-token cross-entropy

at inference:
context -> z1 -> z2 -> ... -> zK -> detokenizer -> continuous commands
```

The decisive feature is not merely that an integer appears somewhere in the
pipeline. The action symbols are generated **autoregressively**: each predicted
token becomes context for the next token.

## RT-2: actions as text tokens

RT-2 converts the robot action into a text-token-compatible sequence and
co-fine-tunes a pretrained VLM on robot trajectories and web-scale
vision-language tasks. Each continuous dimension follows RT-1's 256-bin
discretization; the resulting integers are represented as ordinary numeric
tokens in PaLI-X or reserved action tokens in PaLM-E. During robot-action
decoding, the vocabulary is constrained to valid action symbols.
[RT-2, §III-B](https://arxiv.org/abs/2307.15818)

This makes the policy conceptually simple:

```text
image + instruction -> VLM -> "1 128 91  ..." -> bin centers -> robot action
```

The main attraction is joint training: web answers and robot actions share the
same sequence-model interface and parameters. The cost is sequential decoding
and quantization of every motor dimension.

## OpenVLA: reserved vocabulary entries

OpenVLA uses a 7B Llama-2-based Prismatic VLM. For each action dimension it:

1. takes the 1st-to-99th-percentile training-data interval;
2. divides it into 256 bins;
3. replaces the 256 least-used Llama tokenizer entries with action tokens;
4. trains standard next-token prediction, applying cross-entropy only to the
   action outputs.

[OpenVLA, §3.2](https://arxiv.org/abs/2406.09246)

Percentile bounds reduce sensitivity to outliers, but semantics and bounds are
still dataset-specific. A token ID is not a universal physical action until the
correct dataset statistics and embodiment mapping are applied.

## FAST and π0-FAST: compress before predicting

Naive binning emits one token for every dimension at every timestep. On smooth,
high-frequency trajectories, adjacent values are strongly correlated; a model
can reduce token loss by copying recent tokens without learning the chunk's
global shape. FAST changes the target representation before autoregressive
prediction. [FAST, §§III-V](https://arxiv.org/abs/2501.09747)

The FAST tokenizer performs:

```text
1-second continuous action chunk
  -> per-dimension percentile normalization
  -> discrete cosine transform (DCT)
  -> scale and round frequency coefficients
  -> flatten low frequencies first across action dimensions
  -> byte-pair encoding (BPE)
  -> variable-length discrete token sequence
```

Low-frequency coefficients describe the overall trajectory first. Rounding
makes the coefficient matrix sparse; BPE merges recurring integer patterns and
runs of zeros. The transform is fast to decode, although coefficient rounding
means the full continuous-to-discrete pipeline is lossy. FAST+ fixes a reusable
BPE vocabulary trained on about one million one-second action chunks from
multiple embodiments. [FAST, §V-B-C](https://arxiv.org/abs/2501.09747)

π0-FAST is the π0/PaliGemma backbone trained to emit these compressed tokens
instead of using π0's flow-matching action expert. FAST therefore changes the
**action tokenizer and target sequence**, not the causal decoder's fundamental
next-token computation. In the paper's evaluated mixture it matched the
reported π0 flow model performance while training up to 5x faster; that result
does not prove parity on every robot or data distribution.

## Important boundary cases

- **RT-1:** discrete and non-autoregressive in its final action output. The
  authors removed autoregressive action conditioning because it added latency
  without a meaningful gain in their ablation. It emits per-dimension
  categorical outputs directly for the current step. See
  [continuous/direct prediction](01_continuous_regression.md).
- **π0.5:** uses FAST tokens to make large-scale pretraining efficient, then
  adds a flow-matching expert for fine-grained continuous low-level inference.
  It belongs to both the discrete-token training story and the
  [flow-expert story](04_flow_matching_transformer_expert.md), depending on the
  phase being discussed.
- **Detokenization is not the robot controller:** mapping symbols back to
  normalized numbers still precedes embodiment conversion, safety limits, and
  low-level execution.

## Strengths and limitations

Strengths:

- reuses VLM vocabulary, causal Transformer, cross-entropy loss, and scalable
  next-token training infrastructure;
- can interleave text, reasoning, and action symbols in one output space;
- defines an explicit likelihood over action-token sequences;
- FAST compresses high-frequency chunks without adding a learned action expert.

Limitations:

- autoregressive latency grows with the number of generated action tokens;
- naive bins discard within-bin precision and ignore temporal structure;
- one mistake changes the context for later tokens;
- token validity, normalization statistics, and action semantics must travel
  with the checkpoint;
- FAST adds a reconstruction/compression trade-off and variable token length;
  it improves the target representation but does not eliminate sequential
  decoding.

## Sources

- Brohan et al. *RT-2: Vision-Language-Action Models Transfer Web Knowledge to
  Robotic Control*, §III-B, arXiv:2307.15818.
  [Paper](https://arxiv.org/abs/2307.15818)
- Kim et al. *OpenVLA: An Open-Source Vision-Language-Action Model*, §3.2,
  arXiv:2406.09246v3. [Paper](https://arxiv.org/abs/2406.09246) ·
  [Official code](https://github.com/openvla/openvla)
- Pertsch et al. *FAST: Efficient Action Tokenization for Vision-Language-Action
  Models*, arXiv:2501.09747. [Paper](https://arxiv.org/abs/2501.09747) ·
  [Official project](https://pi.website/research/fast)
- Physical Intelligence et al. *π0.5: a Vision-Language-Action Model with
  Open-World Generalization*, §IV, arXiv:2504.16054.
  [Paper](https://arxiv.org/abs/2504.16054)
