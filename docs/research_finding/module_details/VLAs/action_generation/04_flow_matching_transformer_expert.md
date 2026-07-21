# Flow-Matching Transformer Action Experts

> **Scope.** A pretrained VLM supplies semantic context while a distinct set of
> Transformer weights specializes in continuous action generation through flow
> matching. Representative models: π0 and π0.5. Sources checked 2026-07-21.

## Core idea

The VLM and action generator share attention context but do different jobs:

```text
images + instruction -> pretrained VLM weights ------+
robot state -----------------------------------------+--> shared token context
noisy action chunk -> action-expert weights ---------+
                                                       |
                                              velocity field v_theta
                                                       |
                                            repeated ODE integration
                                                       |
                                                       v
                                              continuous action chunk
```

The expert is not merely a final linear projection. It is a separate
Transformer parameter set for action slots, analogous to a modality expert in a
mixture-of-experts architecture. Action tokens attend bidirectionally within
the chunk and to the multimodal prefix.

More precisely, π0 is one Transformer computation with two token-routed sets of
weights: image/language tokens use the PaliGemma-initialized expert, while state
and noisy-action tokens use the smaller action expert. They exchange
information through shared self-attention. “Separate expert” therefore does not
mean a detached encoder-decoder connected only by one context vector.

## π0 architecture

π0 starts from PaliGemma, a 3B-parameter pretrained VLM, and adds about 300M
randomly initialized action-expert parameters. The complete model has about
3.3B parameters. Images, language, and proprioceptive state form the
conditioning observation; the target is a horizon of `H=50` future actions.
[π0, §IV](https://arxiv.org/abs/2410.24164)

For a demonstrated chunk `A` and Gaussian noise `epsilon`, π0 samples a flow
time `tau` and creates an intermediate point:

```text
A_tau = tau * A + (1 - tau) * epsilon
target velocity = A - epsilon
```

The action expert learns the conditional velocity field from the noisy chunk
toward the data chunk. At inference, it starts at Gaussian noise and integrates
from `tau=0` to `tau=1`. The reported implementation uses ten forward-Euler
steps (`delta=0.1`) and caches prefix attention keys/values so that each step
recomputes only the action suffix. [π0, §IV](https://arxiv.org/abs/2410.24164)

This differs from a DDPM head: the learned target is a flow velocity along an
explicit probability path, and deployment integrates an ODE rather than
following a reverse-noise Markov chain.

## Why an expert instead of action tokens?

The π0 design preserves the VLM's pretrained perception and language path while
giving continuous robot values their own computation. The paper uses this to
predict high-frequency chunks for tasks evaluated at up to 50 Hz. Continuous
joint generation avoids producing hundreds of correlated vocabulary tokens for
one second of dexterous motion. [π0 paper](https://arxiv.org/abs/2410.24164)

The split also permits an asymmetric allocation:

- the larger VLM imports Internet-scale semantic knowledge;
- the smaller expert specializes in proprioception, noisy actions, and motor
  precision;
- attention connects the two without forcing motor values through a text
  vocabulary.

## π0.5: discrete pretraining, flow deployment

π0.5 is intentionally hybrid. Its broad first stage represents robot actions
with FAST tokens and trains them with next-token prediction alongside web,
grounding, and high-level semantic tasks. During post-training it adds the
π0-style action expert and a flow loss for continuous low-level actions.
[π0.5, §IV and Fig. 3](https://arxiv.org/abs/2504.16054)

At inference, the same model performs two different decoding operations:

```text
overall task + observation
  -> autoregressive text decoding
  -> high-level subtask, e.g. "pick up the plate"
  -> condition flow expert on that subtask
  -> ten flow-integration steps
  -> continuous low-level action chunk
```

The high-level text is generated less frequently; the action expert supplies
the fast control chunks. π0.5 therefore cannot be classified from only its
pretraining representation: it belongs to the FAST/autoregressive family during
part of training and the flow-expert family during low-level deployment.

## Relationship to Qwen-VLA

Qwen-VLA should not be treated as another π0-style Transformer expert. In π0,
VLM tokens and robot tokens select different weight sets inside one shared
Transformer computation. Qwen-VLA first computes VLM hidden states, projects
them into a separate 16-block DiT, concatenates them with noisy action tokens,
and runs joint self-attention **inside that downstream decoder**.

The Qwen-VLA paper sometimes calls this module an “action expert,” but its
architectural boundary is a DiT action decoder rather than π0's token-routed
expert weights. It is therefore documented under
[large Diffusion Transformers](05_large_diffusion_transformer.md), not as a
representative of this family. [Qwen-VLA, §§2.2-2.5](https://arxiv.org/abs/2605.30280)

## Strengths

- continuous, coherent chunks without action-token quantization;
- a generative conditional distribution can represent multiple plausible
  trajectories;
- motor-specific capacity does not require replacing the pretrained VLM path;
- cached prefix features reduce repeated work during integration;
- the expert can use bidirectional action attention while the language path
  remains autoregressive.

## Costs and unresolved questions

- ten expert evaluations are slower than a one-pass regressor at equal per-pass
  cost;
- flow-step count and numerical solver affect latency and accuracy;
- more expressive distributions are useful only if demonstrations actually
  contain meaningful modes rather than annotation noise;
- π0's results combine architecture, cross-embodiment data, and pre/post-
  training recipes, so they do not isolate the expert design;
- π0.5's benefits likewise cannot be attributed to flow matching alone because
  FAST pretraining and high-level subtask supervision also change the system.

**Implementation-status caveat.** The π0.5 paper describes autoregressive
high-level subtask generation followed by the flow expert. The public `openpi`
README, at the version checked on 2026-07-21, says π0.5 support is limited to its
flow-matching head; its standard sampling path consumes a supplied prompt and
directly runs the flow loop. The released runtime should not be described as
fully reproducing the paper's hierarchical text-decoding stage without a
specific checkpoint/code path that demonstrates it.
[Official openpi repository](https://github.com/Physical-Intelligence/openpi)

## Sources

- Black et al. *π0: A Vision-Language-Action Flow Model for General Robot
  Control*, §IV, arXiv:2410.24164v4, 2026.
  [Paper](https://arxiv.org/abs/2410.24164)
- Physical Intelligence et al. *π0.5: a Vision-Language-Action Model with
  Open-World Generalization*, §IV, arXiv:2504.16054, 2025.
  [Paper](https://arxiv.org/abs/2504.16054)
- Pertsch et al. *FAST: Efficient Action Tokenization for Vision-Language-Action
  Models*, arXiv:2501.09747. [Paper](https://arxiv.org/abs/2501.09747)
- Wang et al. *Qwen-VLA: Unifying Vision-Language-Action Modeling across Tasks,
  Environments, and Robot Embodiments*, arXiv:2605.30280v2.
  [Paper](https://arxiv.org/abs/2605.30280)
