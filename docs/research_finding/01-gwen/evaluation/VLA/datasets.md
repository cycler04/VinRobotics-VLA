# Datasets and Environments Used to Evaluate Qwen VLA Models

> **Question:** What are the evaluation datasets, task types, modalities,
> embodiments, scales and split boundaries behind the published Qwen robotics
> results?
>
> **Scope:** Qwen-VLA plus selected suites in Qwen-RobotManip, Qwen-RobotNav and
> Qwen-RobotWorld. Research checked on 2026-07-22.

## Dataset, environment, and rollout are different

A robot evaluation may combine three artifacts:

1. a **dataset** of demonstrations, instructions or trajectories;
2. an **environment** that generates observations and applies actions;
3. a **rollout protocol** specifying tasks, seeds, resets and termination.

LIBERO and RoboTwin provide both training trajectories and simulation tasks;
SimplerEnv is primarily a real-to-sim evaluation environment; ALOHA is an
in-house real system. Calling all of them “datasets” hides these differences.
Score definitions are kept in [metrics.md](metrics.md); rollout protocols and
Qwen result values are in [benchmarks.md](benchmarks.md); training objectives
are in [loss.md](loss.md).

## Evaluation suite map

| Suite | Type and embodiment | Published scale | Observation → output | Released partitions/configurations |
|---|---|---:|---|---|
| LIBERO | Simulated single-arm tabletop tasks and demos | Original corpus: 130 tasks | RGB/state + language → continuous manipulation action | Spatial, Object, Goal and LIBERO-100/90/10 suites |
| Simpler-WidowX | Real-to-sim WidowX environment | Public SimplerEnv: 4 WidowX task families | RGB + language → 7D delta Cartesian/rotation/gripper action at 5 Hz | WidowX task families and visual variants |
| RoboCasa-GR1 | Simulated GR1 bimanual humanoid kitchen | 24 atomic tasks described by Qwen-VLA | kitchen vision/state + language → bimanual action | Bounded GR1 configuration; distinct from RoboCasa365 |
| RoboTwin 2.0 | SAPIEN dual-arm simulation | 50 tasks, 5 embodiments, >100,000 released trajectories | multiview RGB/state + language → dual-arm action | Task/embodiment packages; clean and randomized conditions |
| ALOHA in-house | Real bimanual robot | 6 in-domain task categories + 5 OOD axes; demonstration count unknown | 3 RGB cameras + language → bimanual action | No reusable public package or split manifest |
| R2R in VLN-CE | Continuous indoor navigation in Matterport3D | 4,475 unique paths; v1.3 has 10,819 train and 1,839 val-unseen instruction episodes | RGB/depth + English instruction → waypoint/continuous navigation | train, val-seen, val-unseen and test |
| RxR in VLN-CE | Multilingual continuous indoor navigation | 126,069 instructions/path pairs | panoramic/continuous vision + EN/HI/TE instruction → navigation path | train, val-seen, val-unseen, test-standard and test-challenge |
| SimplerEnv-OOD | Static OOD WidowX task collection | 6 held-out tasks in 3 scenes | RGB + new instruction relation/primitive → 7D action | Withheld relations, primitives and color-object bindings |
| DOMINO | Dynamic manipulation dataset and simulator | 35 tasks, 5 embodiments, about 117,000 expert trajectories | synchronized multiview RGB/proprioception + language → continuous action | clean and domain-randomized settings |

"Published scale" describes the public source or dataset description, not an
evaluation rollout denominator. Trial counts and seeds belong to the benchmark
protocol.

## Simulation manipulation

### LIBERO

LIBERO is a lifelong-learning benchmark with human-teleoperated demonstrations.
The original collection contains 130 tasks:

- `LIBERO-Spatial`: 10 tasks emphasizing spatial relations;
- `LIBERO-Object`: 10 tasks varying object identity;
- `LIBERO-Goal`: 10 tasks varying the requested goal;
- `LIBERO-100`: 100 tasks, also organized as LIBERO-90 pretraining and
  LIBERO-10 downstream tasks.

Observations can include workspace RGB, wrist RGB and robot/environment state;
language specifies the task. The native action is continuous and the official
examples use seven values, but a runner must read the environment controller
rather than infer physical semantics from vector length.

Code is MIT and the released datasets are CC BY 4.0. [LIBERO repository][libero]
[LIBERO paper][libero-paper]

### SimplerEnv and Simpler-WidowX

SimplerEnv evaluates real robot policies in simulation with visual matching and
variant aggregation. It supports Google Robot and WidowX/Bridge embodiments;
the current public environment lists six Google task families and four WidowX
task families. The project validated sim-to-real correlation against roughly
1,500 real evaluation episodes per robot domain, but that number is **not** the
Qwen-VLA evaluation denominator.

For WidowX, the policy receives images and a language instruction and emits a
seven-dimensional delta action: translation, axis-angle rotation and gripper.
The public environment runs WidowX control at 5 Hz. [SimplerEnv project][simpler]
[Repository][simpler-repo]

### RoboCasa-GR1 versus RoboCasa365

Qwen-VLA evaluates 24 atomic kitchen tasks using the GR1 bimanual humanoid
configuration. This is a bounded configuration of the RoboCasa ecosystem, not
the newer RoboCasa365 benchmark.

Current RoboCasa365 has 365 tasks—65 atomic and 300 composite—across more than
2,500 kitchen scenes and 3,200 objects. Its corpus reports more than 600 hours
of human demonstrations and more than 1,600 hours of synthetic robot data. The
pretraining portion covers 300 tasks with 100 human demonstrations per task
(30,000 demonstrations), with separate target scenes/tasks for evaluation.

The 24-task Qwen-VLA number and a RoboCasa365 result therefore describe
different task populations and must remain separate. RoboCasa code is MIT;
released assets/datasets are CC BY 4.0. [RoboCasa project][robocasa]
[RoboCasa paper][robocasa-paper]

### RoboTwin 2.0

RoboTwin 2.0 is a SAPIEN-based dual-arm platform with 50 tasks and five supported
robot embodiments. Its public release contains more than 100,000 trajectories,
stored per episode in HDF5 with observations and actions. Evaluation separates a
clean/easier condition from a randomized/harder condition.

Camera layouts, degrees of freedom and action dimensions vary by embodiment, so a converter must
retain embodiment metadata and controller semantics. The repository code is MIT;
the consulted source did not establish one license covering every downloaded
trajectory and asset. [RoboTwin tasks][robotwin-tasks]
[Collection guide][robotwin-data]

## Real-world ALOHA

The Qwen-VLA ALOHA platform has two 6-DoF arms with parallel-jaw grippers and
three RGB cameras: two wrist views and one first-person view. Its six in-domain
task categories are:

1. pick and place;
2. table cleaning;
3. bowl stacking;
4. bowl pick/object placement;
5. towel folding;
6. fine-grained manipulation.

The OOD evaluation changes color, object instance, position, background or
lighting, and instruction wording. These are controlled axes, not a separate
public general-purpose dataset.

The paper and official repository do not publish a reusable ALOHA package or the
number of demonstrations, trials per task, control frequency, action convention,
split manifest or data license. The percentages in the benchmark table do not
recover those missing denominators. [Qwen-VLA, Section 5.1.2][qwen-vla]

## Navigation datasets

### Matterport3D, R2R and VLN-CE

R2R is built in 90 Matterport3D indoor scenes. The original graph-based dataset
contains 7,189 paths and 21,567 English instructions, normally three
instructions per path. VLN-CE converts these routes into continuous Habitat
episodes; its v1.3 packaging retains 4,475 unique R2R trajectories and exposes:

| Split | Instruction episodes | Scenes |
|---|---:|---:|
| train | 10,819 | 61 |
| val-seen | 778 | 53 |
| val-unseen | 1,839 | 11 |
| test | 3,408 | 18 |

The graph-path count, unique continuous trajectory count and instruction-episode
count are three different denominators. In `val-unseen`, scenes are held out
from training. RGB/depth observations and an instruction are
mapped to continuous movement or waypoint actions. Matterport3D meshes require
separate access and terms even though episode annotations and code are public.
[R2R paper][r2r] [VLN-CE data][vln-ce-data]

### Room-across-Room

RxR contains 126,069 instructions/path pairs over about 16,500 Matterport3D
trajectories. Instructions are collected in English, Hindi and Telugu, with
dense guide/follower pose traces and word-to-pose timing. Its splits include
train, val-seen, val-unseen, test-standard and test-challenge; unseen/test scenes
are disjoint from training scenes.

RxR annotations are CC BY; Matterport3D assets have their
own terms. [RxR paper][rxr] [Official repository][rxr-repo]

## OOD and dynamic manipulation

### SimplerEnv-OOD

SimplerEnv-OOD is an author-created task collection on WidowX with six held-out
tasks across three tabletop scenes: MoveAway,
MoveRight, PlaceNear, PlaceRight, PutFront and StackYellow. The withheld axes
include spatial relations, action primitives and color-object bindings.

The task names and boundary are documented, but released episode IDs and a
standalone dataset package are not.
[Qwen-VLA, Section 5.1.4][qwen-vla]

### DOMINO

DOMINO targets dynamic rather than static manipulation. The current paper
describes 35 tasks, five robot embodiments and about 117,000 expert trajectories,
with clean and domain-randomized settings. A trajectory contains synchronized
head/wrist RGB, proprioceptive joint positions and end-effector poses, plus the
continuous robot action. Tasks involve interception, tracking and timed
interaction with moving objects.

The public code/data entry point exists, but a dataset-wide license was not
confirmed from the consulted primary sources. [DOMINO paper][domino]
[Repository][domino-repo]

## Specialized Qwen robotics suites

Later Qwen reports add many suites. The following have scale sufficiently clear
in the primary reports to be useful as dataset facts:

| Suite | Data/output type | Published scale and detail |
|---|---|---|
| EBench | Isaac Sim mobile manipulation episodes → dual-arm/mobile control | 26 task types and 794 evaluation instances; dexterous tabletop, mobile pick/place and long-horizon groups |
| RoboTwin-IF | simulated instruction-following manipulation | 5 task suites with held-out instruction templates; total episode count not reported |
| RoboCasa365 | broad simulated kitchen manipulation | 365 tasks, >2,500 scenes, 3,200 objects; distinct from RoboCasa-GR1 |
| EWMBench | action-conditioned future-video generation | 21 samples across 7 tasks; intentionally small |
| WorldModelBench | generated-video instruction/quality/physics evaluation | 350 instances; output is video, not robot action |

LIBERO-Plus, RoboTwin-Clean2Rand and RoboTwin-XE define useful perturbation or
embodiment axes, but the consulted Qwen source does not give a clear total number
of evaluation episodes. They should not be assigned an invented scale.
[Qwen-RobotManip][robotmanip] [Qwen-RobotWorld][robotworld]

## Canonical ingestion contract

Different action tensors cannot be combined safely from shape alone. A local
adapter should retain:

```text
episode_id, task_id, scene_id, split, seed
instruction text and language
camera name, RGB shape/dtype, timestamp, calibration
state fields, units, order and timestamp
action controller, frame, units, rotation and gripper convention
control frequency, chunk horizon, terminal/timeout reason
success predicate, intervention and collision events
source revision, license/terms and normalization statistics
```

## Availability and unresolved details

- Public entry points exist for LIBERO, SimplerEnv, RoboCasa, RoboTwin, VLN-CE,
  R2R/RxR and DOMINO, although simulator assets may carry separate terms.
- Qwen-VLA's ALOHA data are not published as a reusable benchmark package.
- Exact public task IDs or reusable manifests are unavailable for some
  Qwen-created configurations, especially ALOHA and SimplerEnv-OOD.
- Sample-level overlap between Qwen's large pretraining mixture and public
  evaluation suites cannot be ruled out from the paper.
- No suite was downloaded or ingested into this workspace for this report.

## Sources

- Wang et al. *Qwen-VLA*. [Paper][qwen-vla] · [Local PDF][qwen-vla-local]
- Liu et al. *LIBERO*. [Paper][libero-paper] · [Repository][libero]
- Li et al. *SimplerEnv*. [Paper][simpler-paper] · [Project][simpler]
- Nasiriany et al. *RoboCasa*. [Paper][robocasa-paper] · [Project][robocasa]
- RoboTwin Team. [Task documentation][robotwin-tasks] · [Collection guide][robotwin-data]
- Anderson et al. *R2R*. [Paper][r2r]
- Krantz et al. *VLN-CE*. [Paper][vln-ce] · [Data page][vln-ce-data]
- Ku et al. *RxR*. [Paper][rxr] · [Repository][rxr-repo]
- Fang et al. *DOMINO*. [Paper][domino] · [Repository][domino-repo]
- Qwen Team. *Qwen-RobotManip*. [Paper][robotmanip]
- Qwen Team. *Qwen-RobotWorld*. [Paper][robotworld]

[qwen-vla]: https://arxiv.org/abs/2605.30280v2
[qwen-vla-local]: ../../../papers/05-gwen/vla-specific/qwen_vla_2605.30280.pdf
[libero]: https://github.com/Lifelong-Robot-Learning/LIBERO
[libero-paper]: https://arxiv.org/abs/2306.03310
[simpler]: https://simpler-env.github.io/
[simpler-paper]: https://arxiv.org/abs/2405.05941
[simpler-repo]: https://github.com/simpler-env/SimplerEnv
[robocasa]: https://robocasa.ai/
[robocasa-paper]: https://arxiv.org/abs/2406.02523
[robotwin-tasks]: https://robotwin-platform.github.io/doc/tasks/
[robotwin-data]: https://robotwin-platform.github.io/doc/usage/collect-data.html
[r2r]: https://arxiv.org/abs/1711.07280
[vln-ce]: https://arxiv.org/abs/2004.02857
[vln-ce-data]: https://jacobkrantz.github.io/vlnce/data
[rxr]: https://arxiv.org/abs/2010.07954
[rxr-repo]: https://github.com/google-research-datasets/RxR
[domino]: https://arxiv.org/abs/2603.15620
[domino-repo]: https://github.com/H-EmbodVis/DOMINO
[robotmanip]: https://arxiv.org/abs/2606.17846
[robotworld]: https://arxiv.org/abs/2606.17030
