# Losses Used in Qwen Vision-Language-Action Models

> **Question:** How do Qwen-VLA and the specialized Qwen robotics models train
> actions, language, policies and future video?
>
> **Scope:** Published objectives in Qwen-VLA, Qwen-RobotManip,
> Qwen-RobotNav and Qwen-RobotWorld. Research checked on 2026-07-21.

## Short answer

These models do not share one interchangeable “VLA loss”:

| Model/stage | Learned output | Main published objective |
|---|---|---|
| Qwen-VLA pretraining/SFT | action chunks + text | masked flow-matching MSE + next-token NLL |
| Qwen-VLA RL | closed-loop action policy + value | PPO clipped actor objective + clipped value MSE |
| Qwen-RobotManip | continuous action chunks + text | masked flow matching + next-token NLL; action-only SFT by default |
| Qwen-RobotNav | eight 3D waypoints + text | trajectory MSE + next-token NLL |
| Qwen-RobotWorld | future video latents | conditional flow matching with condition masks |

Loss scale depends on action representation, horizon, timestep distribution,
mask and reduction. It must be paired with score definitions in
[metrics.md](metrics.md) and the evaluation protocol in
[benchmarks.md](benchmarks.md).

## Qwen-VLA action flow matching

Let the clean action chunk be $Y_0\in\mathbb{R}^{H\times K}$ and sample Gaussian
noise $Y_1\sim\mathcal{N}(0,I)$. At noise time $\tau$, training constructs

$$
Y_\tau=(1-\tau)Y_0+\tau Y_1,
$$

and asks the action expert to predict the constant velocity $Y_1-Y_0$.

Because datasets have different action dimensions and episodes can end inside a
chunk, Qwen-VLA uses mask $M_{h,k}$ and first averages over valid timesteps for
each action channel:

$$
\ell_k=
\frac{\sum_h M_{h,k}\left\|
v_\theta(Y_\tau,\tau,o,x)_{h,k}-(Y_1-Y_0)_{h,k}
\right\|_2^2}
{\sum_h M_{h,k}}.
$$

It then gives each of the $c$ active channels equal weight:

$$
\mathcal{L}_{\mathrm{act}}
=\mathbb{E}\left[\frac{1}{c}\sum_{k<c}\ell_k\right].
$$

This two-level reduction is important. Padding has no gradient, longer valid
chunks do not automatically receive more weight, and a high-DoF embodiment does
not dominate only because it has more active scalar channels. It does **not**
make channel units semantically equivalent; dataset-specific normalization and
action definitions still matter. [Qwen-VLA, Sections 2.4–2.5][qwen-vla]

## Joint vision-language and action training

The auxiliary language objective is next-token negative log likelihood:

$$
\mathcal{L}_{\mathrm{vl}}
=-\sum_i \log p_\theta(w_i\mid w_{<i},o_{1:t}),
$$

and the joint objective is

$$
\mathcal{L}
=\lambda_{\mathrm{act}}\mathcal{L}_{\mathrm{act}}
+\lambda_{\mathrm{vl}}\mathcal{L}_{\mathrm{vl}}.
$$

The stages use it differently:

| Stage | Trainable part and signal | Published weighting/noise rule |
|---|---|---|
| Text-to-action alignment | Action expert; text/embodiment prompt conditions flow matching | VLM frozen; Sigmoid-Normal time sampling |
| Continual pretraining | VLM and action expert; VL and action data are co-trained | weights tuned to balance gradient magnitude; numeric values not disclosed; Beta time sampling |
| Supervised fine-tuning | mixed manipulation, navigation and VL targets | VL 0.1; manipulation action 1.0; navigation action 1.0; Beta time sampling |

Manipulation uses action horizon 16 and navigation horizon 8 during SFT. The
paper's ablations show that the timestep distribution is not a cosmetic detail:
Sigmoid-Normal works best for text-to-action alignment, while Beta works best
for SFT. [Qwen-VLA, Sections 3–5.2][qwen-vla]

## Qwen-VLA reinforcement learning

Qwen-VLA starts RL from the multi-task SFT checkpoint and applies PPO. With
probability ratio

$$
r_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)},
$$

the clipped actor objective is

$$
\mathcal{L}_{\mathrm{actor}}
=-\mathbb{E}\left[
\min\left(r_t\hat A_t,
\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)\hat A_t\right)
\right].
$$

The published total is

$$
\mathcal{L}_{\mathrm{RL}}
=\mathcal{L}_{\mathrm{actor}}+c_v\mathcal{L}_{\mathrm{value}},
$$

where the critic uses a clipped MSE. Reported settings are
$\epsilon=0.2$, discount $\gamma=0.99$, GAE $\lambda=0.95$, $c_v=1$, four PPO
epochs, actor learning rate $5\times10^{-6}$ and value-head learning rate
$10^{-4}$. The VLM features are stop-gradient inputs to the value head.

Flow matching does not directly expose a tractable action log-probability. The
paper converts the probability-flow ODE into an SDE with Gaussian transitions
and recomputes the log-probability at one sampled denoising step. Reward is
sparse and binary—1 only for episode success, otherwise 0—and no learned reward
model is used. Rewards and advantages operate at action-chunk level with
$H=16$. The published total equation does not show an entropy or KL penalty, so
one should not add those terms to the documented objective. [Qwen-VLA, Section
4.2][qwen-vla]

SFT supplies most of the reported downstream gain; PPO adds a smaller increase
on its SimplerEnv rollout domain and mostly small changes elsewhere. This is a
reminder that a sophisticated loss does not by itself imply a large behavioral
gain.

## Qwen-RobotManip

Qwen-RobotManip changes the interpolation direction. For clean action $a$,
noise $\epsilon\sim\mathcal{N}(0,I)$ and $t\sim\mathrm{Beta}(1,1.5)$:

$$
x_t=(1-t)\epsilon+ta,
\qquad
u=a-\epsilon.
$$

The model regresses $u$ with squared error. Its published masked reduction
averages over valid entries per sample and then over the batch. The validity
mask combines action-dimension validity, timestep/episode-boundary validity and
per-hand visibility. This differs from Qwen-VLA's equal-per-channel reduction,
so raw action-loss values are not comparable across the two papers.

RobotManip combines this loss with next-token likelihood during pretraining:

$$
\mathcal{L}=\mathcal{L}_{\mathrm{FM}}
+0.1\,\mathcal{L}_{\mathrm{VLM}}.
$$

Its pretraining samples VLA and VL data at a 9:1 ratio and repeats each action
sample with eight independently sampled noise/timestep pairs. Default
task-specific SFT uses only flow matching, without the VLM term. The report does
not document an RL, preference or distillation stage. [Qwen-RobotManip,
Sections 3–4][robotmanip]

An important negative result appears in its context ablation: a configuration
can obtain low training loss by copying recent actions yet perform poorly at
task success. Optimization loss and closed-loop competence must therefore be
reported separately.

## Qwen-RobotNav

Qwen-RobotNav predicts eight waypoints with three coordinates each. Its
trajectory term is direct regression:

$$
\mathcal{L}_{\mathrm{traj}}
=\left\|\hat W-W^*\right\|_2^2,
\qquad
\mathcal{L}=\mathcal{L}_{\mathrm{traj}}
+\lambda\mathcal{L}_{\mathrm{VL}},
\quad \lambda=1.0.
$$

The trajectory term is active only for navigation samples; the VL term is the
standard next-token objective. Training mixes 85% trajectory data with 15%
navigation-related VL data. Coordinates are normalized per dataset using the
99th percentile and mapped to $[-1,1]$, so a numeric MSE has no universal
distance unit. The report does not describe flow matching, RL, preference or
distillation losses. [Qwen-RobotNav, Sections 2.5–2.6][robotnav]

## Qwen-RobotWorld

Qwen-RobotWorld is a future-video model, not an action policy. It encodes video
with a VAE, corrupts the latent with Gaussian noise, and learns conditional flow
matching. Noise time follows a log-normal distribution with a shift adapted to
sequence length. A frozen Qwen2.5-VL encoder supplies language/action guidance.

Conditioning frames are excluded from the denoising loss:

- in text-image-to-video, the first-frame latent is fixed at $t=0$;
- in Scene2Robot, both the scene condition and robot-reference segment are
  fixed at $t=0$;
- only the future generation segment receives denoising gradients.

The report does not print the exact velocity/MSE equation, objective weights,
a separate VAE reconstruction loss, or whether the VAE itself is trained. These
details remain `Unknown`. Its video loss must not be compared numerically with
robot action loss or treated as evidence of executable policy success.
[Qwen-RobotWorld, Sections 3–4][robotworld]

## Comparison checklist

Before interpreting two loss curves, match all of the following:

- output representation, units and normalization statistics;
- horizon, active dimensions and exact mask;
- per-token, per-channel, per-entry or per-sample reduction;
- flow interpolation direction, target velocity and noise-time distribution;
- VL/action sampling ratio and objective coefficients;
- rollout policy, reward, discount, advantage and PPO clipping for RL;
- dataset, split, batch composition and checkpoint stage.

Log `action_loss`, `vl_loss`, `actor_loss`, `value_loss`, reward and success rate
as separate fields. A weighted total alone cannot show which component changed.

## Sources

- Wang et al. *Qwen-VLA*. [Paper][qwen-vla] · [Local PDF][qwen-vla-local]
- Qwen Team. *Qwen-RobotManip*. [Paper][robotmanip] ·
  [Local PDF][robotmanip-local]
- Qwen Team. *Qwen-RobotNav*. [Paper][robotnav] · [Local PDF][robotnav-local]
- Qwen Team. *Qwen-RobotWorld*. [Paper][robotworld] ·
  [Local PDF][robotworld-local]

[qwen-vla]: https://arxiv.org/abs/2605.30280v2
[qwen-vla-local]: ../../../papers/05-gwen/vla-specific/qwen_vla_2605.30280.pdf
[robotmanip]: https://arxiv.org/abs/2606.17846
[robotmanip-local]: ../../../papers/05-gwen/vla-specific/qwen_robotmanip_2606.17846.pdf
[robotnav]: https://arxiv.org/abs/2606.18112
[robotnav-local]: ../../../papers/05-gwen/vla-specific/qwen_robotnav_2606.18112.pdf
[robotworld]: https://arxiv.org/abs/2606.17030
[robotworld-local]: ../../../papers/05-gwen/vla-specific/qwen_robotworld_2606.17030.pdf
