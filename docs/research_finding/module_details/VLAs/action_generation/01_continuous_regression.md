# Continuous Regression and Direct Parallel Action Prediction

> **Scope.** One-pass prediction of continuous low-level actions or action
> chunks. “RT-1-style” is examined explicitly because the original RT-1 is
> often placed in this family incorrectly. Sources checked 2026-07-21.

## Core idea

A continuous regression decoder maps multimodal context directly to real-valued
actions:

```text
context h -> parallel action head -> A_hat in R^(H x D)
```

Here, `H` is the prediction horizon and `D` is the action dimension. A common
training objective is L1 regression:

```text
L = mean(|A_hat - A|)
```

Unlike autoregressive token models, the policy does not serialize every action
dimension into vocabulary tokens. Unlike diffusion or flow, inference does not
start from noise and repeatedly refine the chunk. All `H x D` values can be
produced in one forward pass.

## Taxonomy correction: RT-1 is not continuous regression

**Verified.** RT-1 directly predicts the robot command for the current control
step, but its output distribution is categorical. The paper discretizes each
arm and base action dimension into 256 uniform bins and trains with categorical
cross-entropy. Its decoder-only Transformer produces the action outputs with a
35M-parameter policy running at 3 Hz. [RT-1, §3.3 and Fig. 3](https://arxiv.org/abs/2212.06817)

The accurate description is therefore:

```text
RT-1 = direct, parallel, per-dimension categorical prediction
     != continuous regression
     != language-style autoregression over an action string
```

RT-1 is still a useful structural ancestor of direct regression heads: both
avoid an iterative generative sampler and immediately produce a control output.
The difference is the output distribution and loss.

RT-1 also evaluated the continuous formulation as an ablation: a multivariate
normal action output trained with MSE. It performed substantially worse than
the selected 256-bin categorical head in that paper's seen-task, unseen-task,
and robustness evaluations. This is evidence for the RT-1 setup, not a general
proof that regression is inferior; later OpenVLA-OFT results show that the
backbone, chunking, parallel slots, data, and L1-versus-MSE choice materially
change the outcome. [RT-1, Appendix D.4 and Table 13](https://arxiv.org/abs/2212.06817)

## Representative modern VLA: OpenVLA-OFT

OpenVLA-OFT provides a clean verified instance of this family. It adapts the
OpenVLA backbone using four linked choices:

1. replace sequential action-token decoding with parallel decoding;
2. predict a multi-step action chunk;
3. use continuous action values rather than 256-bin symbols;
4. optimize an L1 regression objective.

The paper's OFT recipe reports 8-step chunks in LIBERO and 25-step chunks for
the real ALOHA setting. The action chunk is emitted in one model evaluation,
which separates neural inference frequency from the robot's command execution
frequency. [OpenVLA-OFT, §I and §V-E](https://arxiv.org/abs/2502.19645)

```text
image/language tokens + optional wrist image/proprioception
                         |
                         v
                  OpenVLA backbone
                         |
                  bidirectional action slots
                         |
                         v
                MLP continuous action head
                         |
                         v
                  H-step action chunk
```

This is “direct” at the action-generator level; the system still denormalizes
the prediction and sends it through a robot-specific controller afterward.

## Why use it?

**Verified benefits in the evaluated settings.** OpenVLA-OFT found that the
combined parallel/chunked/continuous/L1 recipe increased action-generation
throughput by 26x over base OpenVLA on its LIBERO setup. Because there is no
autoregressive loop or iterative denoising loop, latency grows less directly
with the number of output action values. [OpenVLA-OFT project and paper](https://openvla-oft.github.io/)

Engineering advantages are:

- one-pass inference and a simple supervised objective;
- exact real-valued outputs without bin-quantization error;
- parallel action slots make chunking straightforward;
- no diffusion noise schedule, sampler, or integration-step hyperparameter.

## What does it give up?

L1 or MSE regression produces a point estimate. If demonstrations contain
several incompatible valid futures for the same observation, a simple
regressor may select one mode or predict a compromise rather than explicitly
model the full conditional distribution. The OpenVLA-OFT authors specifically
note that L1 can learn a median mode and do not claim that regression is
universally better than diffusion. [OpenVLA-OFT discussion of L1 versus diffusion](https://openvla-oft.github.io/)

Other limits are:

- a long chunk can become stale after the scene changes;
- open-loop execution of too much of the chunk reduces feedback;
- L1/MSE choice alone does not define action semantics or normalization;
- gains from OpenVLA-OFT combine several changes, so they do not isolate the
  benefit of continuous regression by itself.

## When the label is appropriate

Call a decoder **continuous regression** only when all of the following are
true:

- the learned target is a real-valued action tensor;
- the loss compares predicted and demonstrated values directly, such as L1 or
  MSE;
- deployment uses one feed-forward prediction rather than token-by-token
  sampling or repeated denoising/flow integration.

A policy that outputs a bin index later converted to a float is still a
categorical policy. A diffusion/flow model that ends at a continuous tensor is
still an iterative generative policy, not direct regression.

## Sources

- Brohan et al. *RT-1: Robotics Transformer for Real-World Control at Scale*,
  §3.3 and Appendix D.4, arXiv:2212.06817v2.
  [Paper](https://arxiv.org/abs/2212.06817)
- Kim, Finn, and Liang. *Fine-Tuning Vision-Language-Action Models: Optimizing
  Speed and Success*, arXiv:2502.19645, 2025.
  [Paper](https://arxiv.org/abs/2502.19645) ·
  [Official project](https://openvla-oft.github.io/) ·
  [Official code](https://github.com/moojink/openvla-oft)
