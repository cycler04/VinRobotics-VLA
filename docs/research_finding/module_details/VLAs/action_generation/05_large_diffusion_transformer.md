# Large or Decoder-Centric Diffusion Transformers

> **Scope.** Action generators that place substantial Transformer capacity
> inside the iterative diffusion/flow path instead of using a shallow action
> readout. Requested examples: RDT-1B, Dita, and Qwen-VLA. Sources checked
> 2026-07-21.

## Taxonomy correction first

“Large standalone Diffusion Transformer” is not accurate for all three models:

| Model | Verified action-generator scale | Objective | Is the decoder standalone from the whole VLA? |
| --- | ---: | --- | --- |
| RDT-1B | 1.2B-parameter RDT | diffusion | Decoder-centric policy, but still conditioned by separate language/vision encoders |
| Dita | 334M parameters for the published policy | DDPM diffusion | Integrated policy with DINOv2, CLIP, Q-Former, and causal DiT; the authors call it lightweight |
| Qwen-VLA | about 1.15B action expert plus Qwen3.5-4B VLM | conditional flow matching | No; the DiT is explicitly attached to and conditioned by the VLM |

The defensible common label is **large or decoder-centric DiT action
generation**. RDT-1B is the unambiguous billion-parameter example. Dita belongs
because it moves denoising into the main Transformer token sequence rather than
because it is billion-scale. Qwen-VLA overlaps with the
[flow-expert family](04_flow_matching_transformer_expert.md).

## What changes relative to a compact head?

```text
Compact head:
multimodal backbone -> one fused embedding -> small MLP denoiser x N

Decoder-centric DiT:
language/image/state tokens + noisy action tokens
        -> many Transformer blocks with attention-based conditioning x N
        -> refined action chunk
```

The expected benefit is greater capacity and finer token-level conditioning for
heterogeneous images, histories, embodiments, and action spaces. The cost is
that every diffusion or flow step invokes a substantial Transformer.

## RDT-1B

RDT-1B is a Robotics Diffusion Transformer scaled to 1.2B parameters for
bimanual manipulation. It uses separate SigLIP vision and T5-XXL language
encoders, proprioceptive inputs, and a scalable Transformer denoiser. The policy
was pretrained on multi-robot data and predicts the next 64 robot actions.
[RDT-1B paper](https://arxiv.org/abs/2410.07864) ·
[official model card](https://huggingface.co/robotics-diffusion-transformer/rdt-1b)

The denoiser itself has 28 layers, width 2,048, and 32 attention heads. It
alternates cross-attention to language and image conditions. Training corrupts
actions with a 1,000-step DDPM schedule and learns a clean-action estimate with
MSE; deployment uses five DPM-Solver++ steps to produce a 64-action chunk.
[RDT-1B, §4.1, §5, and Appendix H](https://proceedings.iclr.cc/paper_files/paper/2025/file/49f80e4d2471ad4f2edf4f5f1ab62339-Paper-Conference.pdf)

A central design is the Physically Interpretable Unified Action Space. It
allocates slots for common physical quantities—joint and end-effector
positions/velocities, grippers, and mobile-base motion—so heterogeneous robots
can be padded/masked into one interface without pretending their raw vectors
already have identical meanings. The model card explicitly warns that an
unseen embodiment still requires target-robot fine-tuning; a unified tensor is
not zero-shot embodiment transfer.

RDT's conditioning blocks alternate access to language, images, and robot-state
information, letting a large denoiser combine modalities while refining the
action chunk. Its reported strength is action-model capacity for high-
dimensional, multimodal bimanual control; its practical costs include a large
iterative network and sensitivity to control latency/action-horizon choices.

## Dita

Dita was proposed as a reaction to compact diffusion heads. Its authors argue
that conditioning a shallow denoiser on one early-fused embedding can hide
small visual changes that matter for action deltas. Dita instead concatenates:

- frozen-CLIP language tokens;
- DINOv2 image patch features selected by an instruction-conditioned Q-Former;
- diffusion timestep embeddings;
- padded, noised 7D action tokens.

These tokens enter one causal Transformer, so the action chunk is denoised
in-context while attending directly to historical visual tokens. Training uses
a DDPM MSE noise-prediction objective. The 12-block LLaMA2-style Transformer has
width 768; the full policy reports 334M parameters, of which 221M are trainable.
[Dita, §3 and Appendix A](https://arxiv.org/abs/2503.19757)

The reported base setup uses two historical image observations and a trajectory
length of 16. The paper's wording about “16 action chunks” is ambiguous; the
current official configuration uses `traj_length=16` and
`num_pred_action=15`, so this report does not turn that wording into a stronger
16-action claim. Training uses a 1,000-step DDPM schedule; the main zero-shot
evaluation uses 20-step DDIM, while an ablation found 10 steps strongest in one
reported setting. [Dita paper](https://arxiv.org/abs/2503.19757) ·
[official repository](https://github.com/RoboDita/Dita)

**Verified limitation of the requested label.** The paper reports 334M
parameters and explicitly describes Dita as a lightweight open-source baseline.
It is “large-head” relative to a three-layer MLP and decoder-centric in its
conditioning, but it is not in the same scale class as RDT-1B.

The paper's ablations support its architecture only within the tested setups:
longer trajectories helped on ManiSkill2, two observation frames beat one or
three in the reported configuration, and ten DDIM steps worked best among the
evaluated step counts for the cited Google Robot task. These are not universal
decoder laws. [Dita, §4.6](https://arxiv.org/abs/2503.19757)

The name should also be kept exact: the official model is **Dita**. Neither
“DiTA” nor “DiT-Action” is the canonical title in its paper, project page, or
repository.

## Qwen-VLA

Qwen-VLA extends a pretrained Qwen3.5-4B vision-language backbone with an
approximately 1.15B-parameter, 16-block single-stream DiT action expert. VLM
hidden states and the noised action chunk are concatenated and processed with
joint self-attention; AdaLN supplies the flow timestep and multi-section RoPE
distinguishes token axes. The expert learns a conditional flow field and uses a
small number of Euler integration steps at inference.
[Qwen-VLA, §§2.1-2.5](https://arxiv.org/abs/2605.30280)

Its unified output tensor has shape `H x K`. Real action channels occupy a
prefix; unused channels/timesteps are padded and masked out of the loss. Text
prompts describe embodiment and control conventions, while per-dataset
percentile normalization preserves dataset-specific scaling. This unifies the
network interface across manipulation, navigation, and trajectory prediction,
not the physical semantics of all robots.

The default architecture does not use robot proprioception. The paper reports
only a small gain from adding state in one RoboTwin-2.0 ablation and lists
memory, failure recovery, force/tactile feedback, and stronger long-horizon
evaluation among remaining gaps. These limits matter when comparing it with
RDT-1B or π0, both of which condition their action path on robot state.

**Important boundary.** Qwen-VLA is not a standalone DiT that replaces the VLM.
The pretrained VLM performs perception, grounding, reasoning, and text
generation; the DiT is a motor expert conditioned on its hidden states. It is
placed here because its action expert itself is billion-scale.

**Current availability.** As checked on 2026-07-21, the official Qwen-VLA
repository exposes the paper-facing README and assets but no implementation,
checkpoint, package, or release. Its architecture and reported results are
public; a runnable official open-source policy is not yet evidenced by that
repository. [Official Qwen-VLA repository](https://github.com/QwenLM/Qwen-VLA)

## Cross-model comparison

| Property | RDT-1B | Dita | Qwen-VLA |
| --- | --- | --- | --- |
| Continuous generator | Diffusion Transformer | DDPM causal Transformer | Flow-matching DiT expert |
| Conditioning topology | Dedicated multimodal conditioning blocks | Raw multimodal tokens in one in-context sequence | Joint self-attention over VLM states and noisy actions |
| Published horizon example | 64 actions | Trajectory length 16; current repo config predicts 15 actions | 16-action manipulation chunks; 8-waypoint navigation chunks in reported SFT/evaluation |
| Embodiment mechanism | Physically interpretable unified action slots | Common 7D EEF representation in reported core setup | Text embodiment prompt + padded/masked tensor + per-dataset normalization |
| Main classification caveat | Clearly large, but still uses external encoders | Decoder-centric, not billion-scale | Large expert attached to a VLM, also a flow-expert model |

## Trade-offs

Potential advantages:

- more capacity for multimodal and cross-embodiment action distributions;
- attention can preserve fine-grained links between image/history tokens and
  action timesteps;
- a large action model can scale independently from the semantic backbone;
- joint trajectory denoising captures temporal correlation.

Costs and unknowns:

- repeated passes through hundreds of millions or billions of parameters can
  dominate control latency;
- model size, pretraining data, input modalities, and objective change together
  in published comparisons, so scale benefits are not cleanly isolated;
- padded unified tensors do not solve semantic incompatibilities across
  coordinate frames, units, or control conventions;
- benchmark superiority in one robot/simulator does not establish better
  robustness or real-time behavior elsewhere;
- “standalone” should not be used unless the boundary excludes the encoders and
  conditioning modules explicitly.

## Sources

- Liu et al. *RDT-1B: a Diffusion Foundation Model for Bimanual Manipulation*,
  arXiv:2410.07864v2, 2025. [Paper](https://arxiv.org/abs/2410.07864) ·
  [Official project](https://rdt-robotics.github.io/rdt-robotics/) ·
  [Official model card](https://huggingface.co/robotics-diffusion-transformer/rdt-1b)
- Hou et al. *Dita: Scaling Diffusion Transformer for Generalist
  Vision-Language-Action Policy*, arXiv:2503.19757v2, ICCV 2025.
  [Paper](https://arxiv.org/abs/2503.19757) ·
  [Official project](https://robodita.github.io/)
- Wang et al. *Qwen-VLA: Unifying Vision-Language-Action Modeling across Tasks,
  Environments, and Robot Embodiments*, arXiv:2605.30280v2, 2026.
  [Paper](https://arxiv.org/abs/2605.30280) ·
  [Official repository](https://github.com/QwenLM/Qwen-VLA)
