# Modern VLA Action-Generation Families

> **Question.** How do the main modern VLA action-generation families turn
> visual-language context into robot commands, and where do representative
> models actually belong?
>
> **Scope.** Low-level action generation, not the downstream denormalizer,
> safety filter, controller, or actuator interface. Sources were checked on
> 2026-07-21. The five requested families are treated as a useful engineering
> taxonomy, not as mutually exclusive scientific categories.

## Short answer

There is no single axis called “action-decoder type.” At least four design
choices are being mixed:

1. **representation:** continuous values or discrete symbols;
2. **factorization:** one parallel pass, token-by-token generation, or iterative
   refinement from noise;
3. **training objective:** regression, categorical next-token prediction,
   diffusion noise prediction, or flow matching;
4. **architecture and scale:** a small readout, a token-routed Transformer
   expert, or a separate decoder-centric Diffusion Transformer.

This explains the important overlaps. Qwen-VLA is a flow-matching model with a
separate 1.15B-parameter DiT action decoder downstream of a 4B VLM. π0.5 uses
FAST tokens during pretraining but a continuous flow expert for low-level
deployment. Conversely, RT-1 does **not** perform continuous regression: it
predicts one of 256 bins for each action dimension with categorical
cross-entropy. [RT-1, §3.3](https://arxiv.org/abs/2212.06817)

## Family map

| Requested family | Defining computation | Best representative placement | Important correction or overlap |
| --- | --- | --- | --- |
| [Continuous regression](01_continuous_regression.md) | Predict a continuous action or chunk in one forward pass with L1/MSE-style loss | OpenVLA-OFT | RT-1 is a parallel **categorical** policy, not continuous regression |
| [Discrete autoregressive actions](02_discrete_autoregressive_actions.md) | Serialize action symbols and generate them with next-token prediction | RT-2, OpenVLA, π0-FAST | FAST changes the tokenizer, not the autoregressive decoder |
| [Compact diffusion or flow decoder](03_compact_diffusion_flow.md) | A relatively small conditional denoiser iteratively refines an action chunk | Diffusion Policy variants and compact VLA heads | “Compact” is an architecture/scale distinction, not a different probabilistic objective |
| [Flow-matching Transformer expert](04_flow_matching_transformer_expert.md) | Robot tokens use specialized weights inside a shared Transformer computation | π0, π0.5 | π0.5 also uses FAST-token pretraining; this is not Qwen-VLA's downstream-decoder topology |
| [Decoder-centric DiT](05_large_diffusion_transformer.md) | A substantial Transformer is itself the iterative diffusion/flow action decoder | RDT-1B; Dita; Qwen-VLA with caveats | Dita is 334M and calls itself lightweight; Qwen-VLA uses a separate flow-matching DiT after its VLM |

## The common input/output contract

Despite different decoders, most systems can be compared through the same
abstract contract:

```text
images + instruction + optional robot state
                    |
                    v
       multimodal context / prefix
                    |
                    v
       action-generation mechanism
                    |
                    v
 normalized action or action chunk
                    |
                    v
 denormalization + embodiment mapping + safety/controller
```

The documents in this directory stop at the normalized action chunk. A model
that outputs a correct tensor still needs dataset-specific semantics such as
absolute versus delta commands, joint versus end-effector space, rotation
representation, control frequency, and gripper convention.

## Selection intuition

- Choose **parallel regression** when low latency and a simple adaptation path
  matter more than explicitly representing multiple valid trajectory modes.
- Choose **discrete autoregression** when reusing an existing VLM vocabulary,
  training stack, and next-token objective is the central advantage. FAST makes
  this route much more viable for high-frequency chunks.
- Choose **diffusion or flow** when the action distribution is multimodal or a
  coherent high-dimensional trajectory should be generated jointly, accepting
  iterative sampling cost.
- Choose a **small head** when compute and modularity dominate; scale the
  action Transformer when heterogeneous embodiment/action data appear to need
  more capacity and tighter conditioning.

These are design hypotheses, not universal rankings. Reported success rates
are tied to different datasets, robots, control rates, and fine-tuning recipes,
so they do not establish a decoder family as globally superior.

## Primary sources

- Brohan et al. *RT-1: Robotics Transformer for Real-World Control at Scale*,
  arXiv:2212.06817v2, 2023. [Paper](https://arxiv.org/abs/2212.06817)
- Brohan et al. *RT-2: Vision-Language-Action Models Transfer Web Knowledge to
  Robotic Control*, arXiv:2307.15818, 2023.
  [Paper](https://arxiv.org/abs/2307.15818)
- Kim et al. *OpenVLA: An Open-Source Vision-Language-Action Model*,
  arXiv:2406.09246v3, 2024. [Paper](https://arxiv.org/abs/2406.09246)
- Black et al. *π0: A Vision-Language-Action Flow Model for General Robot
  Control*, arXiv:2410.24164v4, 2026.
  [Paper](https://arxiv.org/abs/2410.24164)
- Pertsch et al. *FAST: Efficient Action Tokenization for Vision-Language-Action
  Models*, arXiv:2501.09747, 2025.
  [Paper](https://arxiv.org/abs/2501.09747)
- Wang et al. *Qwen-VLA: Unifying Vision-Language-Action Modeling across Tasks,
  Environments, and Robot Embodiments*, arXiv:2605.30280v2, 2026.
  [Paper](https://arxiv.org/abs/2605.30280)
