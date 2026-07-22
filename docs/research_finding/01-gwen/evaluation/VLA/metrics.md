# Metrics Used to Evaluate Qwen VLA Models

> **Question:** What does each manipulation, navigation, driving and world-model
> score mean?
>
> **Scope:** Score functions only. Dataset/environment contents and scales are
> in [datasets.md](datasets.md); tasks, splits, sensors, evaluator settings and
> Qwen result tables are in [benchmarks.md](benchmarks.md); training objectives
> are in [loss.md](loss.md). Research checked on 2026-07-22.

## The boundary

- A **dataset** is the examples, modalities, annotations, splits and license.
- A **benchmark protocol** specifies which split, prompt, preprocessing and
  evaluator are used.
- A **metric** converts predictions or rollouts into a score.

For embodied evaluation, the evaluator first converts a trajectory into events
such as success, collision or path distance. The metric aggregates those events.
Changing the robot, scene distribution, timeout, success predicate or number of
trials changes the benchmark protocol even when the formula is still called SR.

## Binary rollout success

For $N$ rollouts and binary terminal indicator $S_i$,

$$
\mathrm{SR}=\frac{1}{N}\sum_{i=1}^{N}S_i,
\qquad S_i\in\{0,1\}.
$$

Higher is better. SR estimates the probability of satisfying a benchmark's
complete success predicate under its rollout distribution. It does not reveal
partial progress, collision severity, intervention, smoothness or completion
time.

Aggregation must be explicit:

$$
\mathrm{MacroSR}=\frac{1}{T}\sum_{t=1}^{T}\mathrm{SR}_t,
\qquad
\mathrm{PooledSR}=\frac{\sum_t n_t\mathrm{SR}_t}{\sum_t n_t}.
$$

Macro SR weights every task equally; pooled SR weights tasks by rollout count.
They coincide only under special cases such as equal denominators.

Because SR is an empirical proportion, report the number of trials and a Wilson
or exact-binomial interval when an IID binomial interpretation is reasonable.
Tasks and seeds create clusters, so per-task/per-seed values or a cluster-aware
bootstrap are often more informative than a narrow item-level interval.
[NIST binomial intervals][nist]

## DOMINO Manipulation Score

DOMINO reports SR and a separate partial-credit `Manipulation Score (MS)`. Its
base progress ratio is

$$
\rho=1-
\frac{\|p_{ee}^{T_{end}}-p_{obj}^{T_{end}}\|_2}
{\|p_{ee}^{0}-p_{obj}^{T_{end}}\|_2}.
$$

For dual-arm tasks, route completion is

$$
RC=100\max(\rho_{left},\rho_{right}),
$$

and successful episodes are assigned $RC=100$. MS starts from RC, multiplies it
by 0.5 if the target exits the safe workspace or field of view, and by 0.8 after
a collision with environmental clutter. The penalties may compound.

Thus MS combines end-effector progress with safety/visibility penalties; it is
not a probability and must not be averaged with SR. The paper states
$RC\in[0,100]$ but does not make the clipping rule for a negative raw $\rho$ or
the zero-denominator case explicit; those details require the pinned evaluator.
[DOMINO paper, metric definition][domino]

## Vision-and-language navigation

Let executed path $Q=(q_1,\ldots,q_m)$, reference path
$R=(r_1,\ldots,r_n)$, geodesic distance $d(\cdot,\cdot)$ and success threshold
$d_{th}$.

- **NE (Navigation Error)** is the final geodesic goal distance,
  $d(q_m,r_n)$; lower is better.
- **OS/OSR (Oracle Success Rate)** counts an episode if any visited point lies
  within $d_{th}$ of the goal. OS can exceed SR when the agent visits the goal
  region but does not finish there.
- **SR** counts an episode if its final state satisfies the success predicate,
  commonly final NE within $d_{th}$ under the specified stopping rule.

With shortest-path length $l_i$ and executed path length $p_i$,

$$
\mathrm{SPL}=\frac{1}{N}\sum_{i=1}^{N}
S_i\frac{l_i}{\max(p_i,l_i)}.
$$

SPL rewards success and discounts inefficient travel. An unsuccessful episode
always contributes zero, even if it follows most of the route.
[SPL paper][spl]

Path fidelity is captured by normalized Dynamic Time Warping:

$$
\mathrm{nDTW}(R,Q)=
\exp\left(-\frac{\mathrm{DTW}(R,Q)}{|R|d_{th}}\right).
$$

nDTW lies in $(0,1]$, preserves path order and gives graded signal even when the
agent fails to reach the goal. `SDTW = SR × nDTW` instead gates path fidelity by
terminal success. Distance type and $d_{th}$ are protocol parameters.
[nDTW paper][ndtw]

## Active target tracking in EVT-Bench

EVT-Bench defines **Tracking Rate** as

$$
\mathrm{TR}=\frac{S}{L},
$$

where $S$ is the number of successfully tracked steps and $L$ is the total
number of steps. Higher is better. The paper does not fully formalize the
per-step tracking predicate, so it should not be paraphrased as merely “target
visible” without pinning the code.

Its **Collision Rate** is the fraction of episodes terminated by collision with
the target humanoid; lower is better. Its SR uses a distinct terminal rule: at
the end, the agent must remain oriented toward the target and at a safe distance
of 1–3 m. TR can therefore be high while SR is lower.
[TrackVLA/EVT-Bench metric definition][trackvla]

## NAVSIM v1 PDMS

NAVSIM v1's PDM Score combines safety multipliers with weighted planning
quality:

$$
\mathrm{PDMS}
=NC\cdot DAC\cdot
\frac{5\,TTC+5\,EP+2\,C}{12}.
$$

The components are:

- `NC`: no at-fault collisions;
- `DAC`: drivable-area compliance;
- `TTC`: time-to-collision score;
- `EP`: ego-progress score;
- `C`: comfort score.

`NC` is **No at-fault Collisions**, despite Qwen-RobotNav's table expanding it
as “Navigation Compliance.” The product makes NC and DAC gating factors. PDMS is
computed in a four-second, non-reactive pseudo-simulation where background
actors follow logged futures; it is not a physical success probability or a
reactive closed-loop rollout metric.

Current NAVSIM v2 uses `EPDMS`, with different components and multipliers. PDMS
and EPDMS are not interchangeable, so the version is part of the metric name.
[NAVSIM metric definitions][navsim]

## EBench Score

EBench reports binary SR and a task-specific partial-progress `Score`. The Score
is assembled from weighted subgoals rather than one universal completion-rate
formula. For example, the published task rubric can assign Dishwasher credit
for opening, placing each bowl and closing, while Peg-in-Hole divides credit
between removal and insertion.

The score answers “how much of this task's rubric was satisfied,” not “what
fraction of one generic trajectory was completed.” Cross-task aggregation must
therefore pin the task set, weights and suite revision. The available sources do
not specify one universal overall aggregation equation across all
tasks/instances/seeds; it remains `Unknown`, not an assumed mean of substeps.
[EBench task rubrics][ebench]

## World-model metrics

Qwen-RobotWorld generates future video, not robot actions. Its metric families
must remain namespaced:

| Suite | What the reported components score | Important interpretation limit |
|---|---|---|
| EWMBench | Scene consistency; motion HSD/Dyn/nDTW; semantic diversity, BLEU, CLIP and logic | Mixed scales; not policy SR |
| DreamGen | Perceptual alignment (PA) and instruction following (IF) across environment/object/behavior groups | IF uses Qwen2.5-VL as evaluator |
| PBench | Domain QA plus VBench-derived visual Quality; Overall is their mean | Domain QA uses Qwen2.5-VL as evaluator |
| WorldModelBench | Instruction following, frame/temporal commonsense and five physics-violation categories | Uses its own human-aligned judge, not Qwen2.5-VL |

These scores measure generated observation quality. They do not establish that a
downstream controller can complete a task. Judge identity, prompt and component
scales are part of the metric contract. Shared Qwen lineage in DreamGen/PBench
is a potential confound, not demonstrated evaluator bias.
[Qwen-RobotWorld, Section 5][robotworld]

## Operational metrics

Robot evaluation should report task quality beside, not blended with:

- median and p95 inference latency, control deadline misses and throughput;
- collision/near-miss and human-intervention rates;
- time, distance and energy to success;
- recovery rate after a defined disturbance;
- peak accelerator memory and hardware/configuration.

Each denominator matters: “collision per episode,” “collision per metre” and
“fraction of episodes terminated by collision” answer different questions.

## Metric reporting checklist

```text
metric name, formula, direction and range
suite/evaluator revision and component weights
task set, split, robot/environment and success predicate
sensor/controller/history inputs, timeout and reset/intervention rules
aggregation axis, trials per task, seeds and confidence interval
latency, safety and failure metrics kept as separate fields
```

## Sources

- Fang et al. *DOMINO*. [Paper][domino]
- Anderson et al. *On Evaluation of Embodied Navigation Agents*. [Paper][spl]
- Ilharco et al. *General Evaluation for Instruction Conditioned Navigation*.
  [Paper][ndtw]
- Wang et al. *TrackVLA*. [Paper][trackvla]
- NAVSIM Team. [Official metric definitions][navsim]
- Intern Robotics. [EBench task showcase][ebench]
- Qwen Team. *Qwen-RobotWorld*. [Paper][robotworld] ·
  [Local PDF][robotworld-local]
- NIST. [Binomial proportion confidence intervals][nist]

[domino]: https://arxiv.org/abs/2603.15620
[spl]: https://arxiv.org/abs/1807.06757
[ndtw]: https://arxiv.org/abs/1907.05446
[trackvla]: https://proceedings.mlr.press/v305/wang25f.html
[navsim]: https://github.com/autonomousvision/navsim/blob/main/docs/metrics.md
[ebench]: https://internrobotics.github.io/EBench-doc/evaluation/task-showcase/
[robotworld]: https://arxiv.org/abs/2606.17030
[robotworld-local]: ../../../papers/05-gwen/vla-specific/qwen_robotworld_2606.17030.pdf
[nist]: https://itl.nist.gov/div898/handbook/prc/section2/prc241.htm
