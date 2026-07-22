# Qwen-VLA Evaluation and Benchmark Evidence

> **Question:** What do the published Qwen-VLA evaluations actually establish,
> and what remains unverified?
>
> **Scope:** Qwen-VLA arXiv:2605.30280v2 and its official repository, checked on
> 2026-07-22. Architecture and training are covered in
> [qwen_vla.md](../../qwen_models/qwen_vla.md).

## Short answer

Qwen-VLA is evaluated as one generalist checkpoint across manipulation,
navigation, real-world transfer, and out-of-distribution settings. Its strongest
published evidence is broad coverage with one policy, not proof that every score
is directly comparable to every specialist baseline.

The paper reports:

- high in-distribution simulation success across single- and dual-arm platforms;
- real-world ALOHA transfer with a large benefit from pretrained initialization;
- competitive continuous-environment navigation;
- non-zero transfer to unseen static and dynamic manipulation tasks;
- ablations connecting VL co-training, SFT, and RL to measured policy success.

All values below are **author-reported**. No Qwen-VLA checkpoint or evaluation
code was available in the official repository at the time checked, so the results
have not been reproduced in this workspace. [Official repository][qwen-vla-repo]

**Note: OOD** stands for  **Out-of-Distribution,** refers to  **testing a model on data that is meaningfully different from the data distribution it was trained on** . OOD evaluation measures a model's ability to generalize beyond its training experience.

## Benchmark protocol boundaries

Formulas, direction and interpretation are isolated in
[metrics.md](metrics.md). A VLA benchmark additionally fixes the environment,
task IDs, split, robot, cameras, controller, reset/timeout rules, success
predicate, seeds and rollout count.

| Suite                                  | Protocol boundary needed to interpret the result                                                                                  |
| -------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| LIBERO / Simpler / RoboCasa / RoboTwin | Exact task variants, simulator revision, initial-state distribution, rollout denominator and success predicate                    |
| ALOHA                                  | Real robot setup, camera/calibration, operator reset/intervention policy, trial count and OOD-axis construction                   |
| R2R / RxR VLN-CE                       | `Val-Unseen` release, language subset, sensor panorama, waypoint policy, geodesic distance and success threshold                |
| DOMINO                                 | `DOMINO@alpha`, task level, embodiment, clean/randomized condition, current-frame/history input and zero-shot/fine-tuned status |
| EVT-Bench                              | Single-target task, single-view setting and exact per-step tracking predicate                                                     |
| NAVSIM                                 | v1 PDMS rather than v2 EPDMS,`navtest`, sensor/history inputs and pseudo-simulator revision                                     |
| EBench                                 | Revision, task-family selection, instance split, per-task partial-credit rubric, seeds and cross-task aggregation                 |
| RobotWorld suites                      | Generation setup, evaluated subset, judge model/prompt and component aggregation                                                  |

Qwen-RobotNav's NAVSIM result supplies ground-truth trajectories from the three
previous frames as history. Its table calls the measures “closed-loop,” but
NAVSIM v1 PDMS uses a four-second non-reactive pseudo-simulation, so it must not
be read as a reactive closed-loop success probability. [NAVSIM metrics][navsim]
[Qwen-RobotNav, Section 5][robotnav]

Qwen-RobotManip names EBench Table Top, Simple PnP and Long Horizon as “splits,”
while current EBench materials distinguish task families from data splits such
as Validation-Train, Validation-Unseen and Test. Pinning the EBench revision is
therefore part of reproduction, not a cosmetic label. [EBench][ebench]

## Main simulation results

The paper uses action chunk length `H = 16` and reports average task success rate
under the StarVLA protocol. Qwen-VLA is trained jointly across embodiments;
specialist baselines are fine-tuned separately for each benchmark.

| Model                                   |         LIBERO |   RoboCasa-GR1 | Simpler-WidowX |  RoboTwin Easy |  RoboTwin Hard |
| --------------------------------------- | -------------: | -------------: | -------------: | -------------: | -------------: |
| Qwen-VLA-Base                           |           90.8 |           40.4 |           64.3 |           64.3 |           66.4 |
| Qwen-VLA-Instruct                       | **97.9** | **56.7** | **73.7** | **86.1** | **87.2** |
| Best specialist value listed in Table 4 |           98.6 |           58.3 |           64.6 |           86.0 |           85.0 |

**Verified from the paper:** Qwen-VLA-Instruct is competitive with or above the
listed specialists on most columns while using one generalist policy.

**Comparison limit:** the specialist and generalist rows have different training
regimes. The table demonstrates a strong generalist result under the authors'
protocol; it does not isolate equal data, compute, architecture, or tuning budget.
[Qwen-VLA, Section 5.1.1 and Table 4][qwen-vla]

## Real-world ALOHA transfer

Both Qwen-VLA ALOHA variants use the same architecture. One trains from scratch;
the other fine-tunes from Qwen-VLA-Base.

| Setting                                        | From scratch | From Qwen-VLA-Base | Difference |
| ---------------------------------------------- | -----------: | -----------------: | ---------: |
| Six in-domain task categories, average success |         48.5 |     **83.6** |   +35.1 pp |
| Five OOD categories, average success           |         36.2 |     **76.9** |   +40.7 pp |

The **OOD** categories are color, object instance, position, background, and
instruction. This is the cleanest evidence in the report that broad pretraining,
not architecture alone, improves transfer because the two variants share the
same architecture. It remains one robot platform and a finite set of laboratory
conditions. [Qwen-VLA, Section 5.1.2 and Tables 5-6][qwen-vla]

## Navigation

Qwen-VLA is evaluated on the `Val-Unseen` splits of R2R and RxR in VLN-CE using
a sliding-window waypoint action.

| Model              |         R2R OS |         R2R SR |        R2R SPL |         RxR SR |        RxR SPL |       RxR nDTW |
| ------------------ | -------------: | -------------: | -------------: | -------------: | -------------: | -------------: |
| Qwen-VLA-Base      |           61.7 |           53.8 |           49.4 |           55.1 |           45.8 |           56.2 |
| Qwen-VLA-Instruct  | **69.0** | **57.5** |           51.2 | **59.6** | **47.8** |           57.1 |
| StreamVLN baseline |           64.2 |           56.9 | **51.9** |           52.9 |           46.0 | **61.9** |

Qwen-VLA-Instruct leads the listed open baselines in success rate, but not every
path-quality metric. This matters: a higher SR with lower nDTW means destination
success and trajectory fidelity should remain separate conclusions.
[Qwen-VLA, Section 5.1.3 and Table 7][qwen-vla]

## Static and dynamic OOD manipulation

| Benchmark      | Training/evaluation distinction                                |     Qwen-VLA-Base |           Qwen-VLA-Instruct | Strong comparison in paper    |
| -------------- | -------------------------------------------------------------- | ----------------: | --------------------------: | ----------------------------- |
| SimplerEnv-OOD | Fine-tune on Bridge pick-and-place; test six unseen task types |           25.3 SR |           **32.0 SR** | pi0.5: 12.6 SR                |
| DOMINO         | Zero-shot on all 35 dynamic suites; no dynamic fine-tuning     | 21.1 SR / 37.4 MS | **26.6 SR / 39.5 MS** | LingBot-VA: 24.1 SR / 36.1 MS |

SimplerEnv-OOD probes unseen spatial instructions, primitives, and color-object
bindings. DOMINO probes moving-object manipulation and continuous execution
quality. These are more informative for robustness than the in-distribution
average, although absolute success remains low on both suites.
[Qwen-VLA, Sections 5.1.4-5.1.5 and Tables 8-9][qwen-vla]

## What the ablations support

### Post-training stages

RL rollouts are collected only in SimplerEnv with binary success reward.

| Stage |        Simpler |       RoboCasa |           RoboTwin E/H |         LIBERO |    Simpler OOD |          DOMINO SR/MS |
| ----- | -------------: | -------------: | ---------------------: | -------------: | -------------: | --------------------: |
| CPT   |           64.3 |           40.4 |            64.3 / 66.4 |           90.8 |           25.3 |           21.1 / 37.4 |
| + SFT |           70.8 |           56.0 |            86.3 / 87.1 |           97.8 |           31.6 |           25.7 / 39.1 |
| + RL  | **73.7** | **56.7** | 86.1 / **87.2** | **97.9** | **32.0** | **26.6 / 39.5** |

**Verified:** SFT supplies most of the gain. RL adds +2.9 pp on its rollout
environment and small changes elsewhere, including -0.2 pp on RoboTwin-Easy.
This supports modest transfer without obvious broad forgetting; it does not
support a claim that RL uniformly improves every task. [Qwen-VLA, Section 5.2.3][qwen-vla]

### VL co-training and state

- Mixing VL data with action data improves RoboCasa-GR1 by 4.9 pp and RoboTwin
  2.0 by 4.6 pp in the reported ablation, while LIBERO and Simpler are similar.
- Adding joint state changes RoboTwin success by at most +0.7 pp on Easy and
  +1.3 pp on Hard, so the default model omits proprioceptive state.

These are benchmark-specific findings, not universal statements about all VLAs.
[Qwen-VLA, Sections 5.2.2 and 5.2.4][qwen-vla]

## Related Qwen robotics models

Qwen-VLA is the unified manipulation/navigation model. Three later Qwen reports
specialize or change the problem, so their scores should not be merged into the
Qwen-VLA table.

| Model           | Output/problem                                | Strong published evidence                                                           | Important counter-evidence                                                                                                                |
| --------------- | --------------------------------------------- | ----------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| Qwen-RobotManip | Continuous manipulation policy                | LIBERO-Plus 89.0; RoboTwin-Clean2Rand Hard 62.6; EBench 45.6 SR                     | Several new OOD suites are author-proposed; external reproduction is not reported                                                         |
| Qwen-RobotNav   | Navigation policy                             | Panoramic RxR Val-Unseen 76.5 SR/65.7 SPL; VLNVerse fine-grained 63.75 SR/57.93 SPL | EVT tracking rate is highest, but success 77.4/78.6 trails specialist values 86.0-86.9; driving results depend strongly on history priors |
| Qwen-RobotWorld | Language-conditioned future-video world model | Best open-source total in the reported DreamGen and WorldModelBench tables          | It does not output executable robot actions; some QA/IF metrics use Qwen2.5-VL as judge, creating potential evaluator-family bias         |

The most useful RobotManip finding is that in-distribution scores are already
near saturation, while OOD tests separate scratch and pretrained variants. For
example, its scratch model scores 78.3 on LIBERO-Plus versus 89.0 pretrained, and
22.6 versus 62.6 on RoboTwin-Clean2Rand Hard. This supports treating controlled
distribution shift as a primary foundation-model evaluation, not an appendix.
[Qwen-RobotManip, Tables 3-5][robotmanip]

Qwen-RobotNav provides a useful negative result: tracking rate and task success
can move in different directions. Its 4B model reports 90.0 tracking rate but
77.4 success, below ABot-N0's 86.9 success. A single navigation headline score
would hide this behavior. [Qwen-RobotNav, Table 6][robotnav]

Qwen-RobotWorld belongs in evaluation research because world models may become
simulators, policy critics, or synthetic-data generators. Its current tables do
**not** establish that training a policy on generated videos improves executable
control. [Qwen-RobotWorld, Section 5][robotworld]

## Evaluation gaps

- **Verified:** Most quantitative tasks remain short-horizon and benchmark-driven;
  the paper names long-duration deployment and failure recovery as open problems.
- **Verified:** Real-world OOD evidence is from ALOHA and a bounded set of visual
  and instruction shifts; it is not cross-lab replication.
- **Verified:** The official repository currently presents the report and results
  but no downloadable model, inference implementation, or evaluation harness.
- **Unknown:** confidence intervals, run-to-run variance, and the exact number of
  evaluation rollouts are not reported consistently beside every aggregate.
- **Unknown:** end-to-end control latency, missed deadlines, memory use, and
  hardware-dependent throughput.
- **Unknown:** collision severity, unsafe near-misses, intervention rate, and
  recovery after partial failure.
- **Inferred:** A deployment decision should weight OOD success, latency, and
  safety failures more heavily than small in-distribution score differences.

## Minimal reproduction protocol

For each checkpoint and task, record:

1. environment and code revision, task IDs, seeds, and number of rollouts;
2. robot embodiment, cameras, image preprocessing, control frequency, and action
   chunk horizon;
3. action semantics, units, coordinate frame, normalization, and gripper
   convention;
4. success predicate, timeout, intervention and collision definitions;
5. mean success with confidence interval plus per-task results;
6. inference latency distribution, deadline misses, peak memory, and hardware;
7. failure taxonomy: perception, grounding, planning, control, recovery, or
   environment fault.

Without these fields, a reproduced aggregate is difficult to compare with the
paper and insufficient for a robot deployment decision.

## Sources

- Wang et al. *Qwen-VLA: Unifying Vision-Language-Action Modeling across Tasks,
  Environments, and Robot Embodiments*. arXiv:2605.30280v2, 2026.
  [Paper][qwen-vla] · [Local PDF][qwen-vla-local]
- Qwen Team. *Qwen-VLA official repository*. Accessed 2026-07-22.
  [Repository][qwen-vla-repo]
- Qwen Team. *Qwen-RobotManip*. arXiv:2606.17846, 2026.
  [Paper][robotmanip] · [Local PDF][robotmanip-local]
- Qwen Team. *Qwen-RobotNav*. arXiv:2606.18112, 2026.
  [Paper][robotnav] · [Local PDF][robotnav-local]
- Qwen Team. *Qwen-RobotWorld*. arXiv:2606.17030, 2026.
  [Paper][robotworld] · [Local PDF][robotworld-local]
- Fang et al. *DOMINO*. [Paper][domino]
- NAVSIM Team. [Official metric definitions][navsim]
- Intern Robotics. [EBench documentation][ebench]

[qwen-vla]: https://arxiv.org/abs/2605.30280v2
[qwen-vla-local]: ../../../papers/05-gwen/vla-specific/qwen_vla_2605.30280.pdf
[qwen-vla-repo]: https://github.com/QwenLM/Qwen-VLA
[robotmanip]: https://arxiv.org/abs/2606.17846
[robotmanip-local]: ../../../papers/05-gwen/vla-specific/qwen_robotmanip_2606.17846.pdf
[robotnav]: https://arxiv.org/abs/2606.18112
[robotnav-local]: ../../../papers/05-gwen/vla-specific/qwen_robotnav_2606.18112.pdf
[robotworld]: https://arxiv.org/abs/2606.17030
[robotworld-local]: ../../../papers/05-gwen/vla-specific/qwen_robotworld_2606.17030.pdf
[domino]: https://arxiv.org/abs/2603.15620
[navsim]: https://github.com/autonomousvision/navsim/blob/main/docs/metrics.md
[ebench]: https://internrobotics.github.io/EBench-doc/
