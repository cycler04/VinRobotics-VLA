# VLA Benchmark Metrics: Formulas, Ranges, and Interpretation

## 1. Scope

There is no single standardized list of metrics used by every Vision-Language-Action paper. Different studies evaluate manipulation, navigation, lifelong learning, sim-to-real fidelity, real-time deployment, or offline action prediction.

This report covers the recurring metric families used across VLA research and the major benchmark-specific metrics. The focus is robot manipulation, with navigation metrics included separately.

An important distinction is:

| Metric type                         | Evaluates                                           | Example            |
| ----------------------------------- | --------------------------------------------------- | ------------------ |
| **Training loss**             | Whether optimization is progressing                 | Flow-matching loss |
| **Offline prediction metric** | Whether predicted actions resemble recorded actions | Action MAE         |
| **Rollout metric**            | Whether the robot actually completes the task       | Success rate       |
| **System metric**             | Whether the model can run on a robot in time        | End-to-end latency |

A low training loss or action error does not guarantee a high rollout success rate. Rollout metrics are the strongest evidence of actual robot capability.

## 2. Notation

Assume:

- $N$: number of evaluated episodes;
- $T$: number of tasks;
- $S_i \in \{0,1\}$: whether episode $i$ succeeds;
- $p_i \in [0,1]$: partial progress in episode $i$;
- $a_{t,d}$: ground-truth action at timestep $t$, dimension $d$;
- $\hat{a}_{t,d}$: predicted action;
- $H$: action horizon or number of predicted timesteps;
- $D$: number of action dimensions.

When papers report percentages, multiply a metric in $[0,1]$ by 100.

## 3. Quick reference

| Metric                | Formula or unit                           |        Natural range | Direction         | Main meaning                      |
| --------------------- | ----------------------------------------- | -------------------: | ----------------- | --------------------------------- |
| Success Rate          | successful episodes / episodes            |              ([0,1]) | Higher            | Full task completion              |
| Mean Success Rate     | mean SR across tasks                      |              ([0,1]) | Higher            | Average multi-task performance    |
| Progress Score        | completed weighted stages                 |              ([0,1]) | Higher            | Partial completion                |
| CALVIN average length | mean completed subtasks                   |              ([0,5]) | Higher            | Long-horizon chaining             |
| Episode Return        | sum of rewards                            | Environment-specific | Higher            | Reward accumulated by RL policy   |
| Action MAE            | mean absolute action error                |       ([0,infinity)) | Lower             | Offline action closeness          |
| Action MSE            | mean squared action error                 |       ([0,infinity)) | Lower             | Penalizes large action errors     |
| Token Accuracy        | correct action tokens / tokens            |              ([0,1]) | Higher            | Discrete action prediction        |
| Cross-Entropy / NLL   | negative log probability                  |       ([0,infinity)) | Lower             | Quality of predicted distribution |
| Position Error        | metres or centimetres                     |       ([0,infinity)) | Lower             | Endpoint precision                |
| Rotation Error        | geodesic angle                            |    ([0,\pi]) radians | Lower             | Orientation precision             |
| ADE                   | mean trajectory-point distance            |       ([0,infinity)) | Lower             | Whole-trajectory accuracy         |
| FDE                   | final trajectory-point distance           |       ([0,infinity)) | Lower             | Final target accuracy             |
| Collision Rate        | collision episodes / episodes             |              ([0,1]) | Lower             | Safety                            |
| Recovery Rate         | recovered failures / recoverable failures |              ([0,1]) | Higher            | Error correction                  |
| Latency               | milliseconds                              |       ([0,infinity)) | Lower             | Reaction delay                    |
| Control Frequency     | hertz                                     |       ([0,infinity)) | Higher, if stable | Update speed                      |
| Throughput            | action steps or chunks per second         |       ([0,infinity)) | Higher            | Compute capacity                  |
| Generalization Gap    | seen SR minus unseen SR                   |             ([-1,1]) | Near 0            | Distribution-shift robustness     |
| Pearson correlation   | real vs simulated scores                  |             ([-1,1]) | Higher            | Linear sim-real agreement         |
| MMRV                  | worst ranking violation average           |             ([0,1])* | Lower             | Sim-real ranking fidelity         |
| SPL                   | success weighted by path efficiency       |              ([0,1]) | Higher            | Navigation success and efficiency |

\*MMRV lies in ([0,1]) when the underlying real-world performance values are normalized to ([0,1]).

## 4. Task-completion metrics

### 4.1. Success Rate (SR)

The most common VLA rollout metric is binary task success:

$$
SR = \frac{1}{N}\sum_{i=1}^{N} S_i
$$

| Property | Value                                                                         |
| -------- | ----------------------------------------------------------------------------- |
| Range    | ([0,1]), or 0-100%                                                            |
| Best     | 1 or 100%                                                                     |
| Worst    | 0                                                                             |
| Used by  | LIBERO VLA evaluations, RLBench, RoboCasa, SimplerEnv, real-robot evaluations |

Example: if 82 of 100 rollouts succeed, SR is (0.82 = 82%\).

Success must be defined precisely. “Put the mug on the tray” might require the mug to be released, upright, completely inside the tray boundary, and stable for a fixed time.

#### Limitations

- It gives no credit for partial completion.
- It depends heavily on the task's success predicate.
- It can hide unsafe behavior that happens before eventual success.
- A result from 10 trials is much less reliable than the same percentage from 1,000 trials.

### 4.2. Mean Success Rate across tasks

For (T) tasks:

$$
\mathrm{MeanSR} = \frac{1}{T}\sum_{j=1}^{T}SR_j
$$

Range: ([0,1]). Higher is better.

This is a **macro-average**: every task receives equal weight even if tasks contain different numbers of episodes.

A **micro-average** instead pools all episodes:

$$
\mathrm{MicroSR} =
\frac{\sum_j N_jSR_j}{\sum_j N_j}
$$

Micro- and macro-averages are identical only when each task uses the same number of trials.

### 4.3. Task Progress or Progress Score (PS)

For a task divided into (K) ordered stages:

$$
\mathrm{PS}_i = \frac{1}{K}\sum_{k=1}^{K} c_{i,k}
$$

where (c_{i,k}=1) if stage (k) was completed in episode (i).

The reported mean is:

$$
\mathrm{MeanPS} = \frac{1}{N}\sum_i \mathrm{PS}_i
$$

Range: ([0,1]). Higher is better.

VLABench uses a weighted form that gives separate credit for choosing the correct objects/receptacles and for completing task steps:

$$
\mathrm{PS} =
\alpha\frac{n_{\mathrm{correct}}}{n_{\mathrm{targets}}}
+(1-\alpha)\frac{k_{\mathrm{completed}}}{K}
$$

with default decision weight (\alpha=0.2). It is naturally in ([0,1]), although tables may report it as 0-100. [VLABench paper](https://arxiv.org/abs/2412.18194)

Progress Score is valuable for long tasks, but benchmark designers must choose meaningful stages and weights.

### 4.4. Composite-task or all-stages success

For a task requiring every stage:

$$
S_i = \prod_{k=1}^{K} c_{i,k}
$$

Range: (\{0,1\}) per episode and ([0,1]) after averaging.

This explains why long-horizon success falls rapidly: one failed stage makes the entire episode unsuccessful.

### 4.5. Completion Time

$$
\mathrm{Time}_i = t_i^{\mathrm{finish}} - t_i^{\mathrm{start}}
$$

| Property         | Value                                     |
| ---------------- | ----------------------------------------- |
| Range            | ([0,infinity)) seconds                    |
| Best             | Lower, among successful and safe episodes |
| Failure handling | Report separately or assign timeout       |

Do not average successful completion times while silently removing failures; doing so can make a weak policy appear fast.

### 4.6. Episode Return

Used mainly for reinforcement learning:

$$
G_i = \sum_{t=0}^{H_i-1}\gamma^t r_{i,t}
$$

where (r_{i,t}) is reward and (\gamma \in [0,1]) is the discount factor.

Range: environment-specific and possibly negative. Higher is better. Return values cannot be compared across benchmarks unless reward definitions and normalization are identical.

## 5. Long-horizon metrics

### 5.1. CALVIN prefix success rates

CALVIN evaluates chains of five language instructions. Define (C_i) as the number of consecutive subtasks completed from the start of sequence (i), with (C_i \in \{0,1,2,3,4,5\}).

The success rate for completing at least (k) subtasks is:

$$
SR_{\ge k} = \frac{1}{N}\sum_{i=1}^{N}\mathbb{1}[C_i \ge k]
$$

Range: ([0,1]) for each (k \in \{1,2,3,4,5\}). Higher is better.

The five numbers answer progressively harder questions:

- (SR_{\ge1}): can it complete the first task?
- (SR_{\ge5}): can it complete the entire five-instruction chain?

### 5.2. CALVIN average completed sequence length

$$
\mathrm{AvgLen} = \frac{1}{N}\sum_{i=1}^{N} C_i
$$

Its natural range is ([0,5]), and:

$$
\mathrm{AvgLen} = \sum_{k=1}^{5}SR_{\ge k}
$$

Example: prefix success rates ([0.90,0.70,0.50,0.30,0.10]) give:

$$
\mathrm{AvgLen}=0.90+0.70+0.50+0.30+0.10=2.50
$$

This is the headline number often shown in modern CALVIN VLA tables. CALVIN was designed specifically for long-horizon language-conditioned continuous control. [CALVIN paper](https://arxiv.org/abs/2112.03227)

### 5.3. Subtask transition success

For transition from stage (k) to (k+1):

$$
\mathrm{TransitionSR}_k =
\frac{\#\text{episodes completing }k+1}
{\#\text{episodes completing }k}
$$

Range: ([0,1]). Higher is better.

This helps distinguish failure to perform a skill from failure to recognize that the next skill should begin.

## 6. Offline action-prediction metrics

These metrics evaluate held-out demonstrations without running the policy in the environment.

### 6.1. Mean Absolute Error (MAE or L1)

$$
\mathrm{MAE} = \frac{1}{HD}\sum_{t=1}^{H}\sum_{d=1}^{D}
|a_{t,d}-\hat{a}_{t,d}|
$$

Range: ([0,infinity)). Lower is better.

MAE is easier to interpret if actions use physical units. If actions are normalized, the number depends on the normalization scheme.

### 6.2. Mean Squared Error (MSE or L2 loss)

$$
\mathrm{MSE} = \frac{1}{HD}\sum_{t,d}
(a_{t,d}-\hat{a}_{t,d})^2
$$

Range: ([0,infinity)). Lower is better. It penalizes large errors more strongly than MAE.

### 6.3. Root Mean Squared Error (RMSE)

$$
\mathrm{RMSE}=\sqrt{\mathrm{MSE}}
$$

Range: ([0,infinity)). Lower is better. RMSE has the same units as the action.

### 6.4. Action-token accuracy

For discrete action tokens:

$$
\mathrm{TokenAcc} = \frac{1}{M}\sum_{m=1}^{M}
\mathbb{1}[y_m=\hat{y}_m]
$$

Range: ([0,1]). Higher is better.

Token accuracy depends on tokenization. Two models with different bin sizes or FAST/VQ tokenizers cannot be compared fairly using raw token accuracy.

### 6.5. Exact action-sequence match

$$
\mathrm{ExactMatch}=
\frac{1}{N}\sum_i\mathbb{1}[y_i^{1:M}=\hat{y}_i^{1:M}]
$$

Range: ([0,1]). Higher is better. It is extremely strict and becomes less informative for long sequences.

### 6.6. Negative log-likelihood and cross-entropy

$$
\mathrm{NLL}=-\frac{1}{M}\sum_{m=1}^{M}
\log p_\theta(y_m\mid x,y_{<m})
$$

Range: ([0,infinity)) for discrete targets. Lower is better.

Perplexity is:

$$
\mathrm{PPL}=e^{\mathrm{NLL}}
$$

Range: ([1,infinity)). Lower is better. These metrics evaluate probabilistic prediction, not rollout success.

### 6.7. Flow-matching or diffusion loss

A simplified flow-matching objective is:

$$
\mathcal{L}_{\mathrm{FM}}=
\mathbb{E}_{A,\epsilon,\tau}
\left[\|v_\theta(A_\tau,\tau,c)-u(A,\epsilon)\|_2^2\right]
$$

Range: ([0,infinity)). Lower is better during training.

This is primarily an optimization loss. Its scale depends on action normalization, noise schedule, time sampling, and dimensionality, so it is not a meaningful cross-paper benchmark metric.

## 7. Geometric and trajectory metrics

### 7.1. End-effector position error

$$
e_{\mathrm{pos}} = \|\hat{p}-p^*\|_2
$$

Range: ([0,infinity)) metres. Lower is better.

This is useful for reaching, insertion, and calibration tasks. Always report units and whether the target is a demonstration pose or a task-defined goal.

### 7.2. Orientation error

For predicted and target rotation matrices (\hat{R}) and (R^*):

$$
e_{\mathrm{rot}}=
\cos^{-1}\left(\frac{\operatorname{tr}((R^*)^T\hat{R})-1}{2}\right)
$$

Range: ([0,\pi]) radians or ([0,180^ degrees]). Lower is better.

Euler-angle component errors should be avoided near wrap-around and singularities.

### 7.3. Average Displacement Error (ADE)

$$
\mathrm{ADE}=\frac{1}{H}\sum_{t=1}^{H}
\|\hat{p}_t-p_t^*\|_2
$$

Range: ([0,infinity)). Lower is better. ADE measures the whole predicted path.

### 7.4. Final Displacement Error (FDE)

$$
\mathrm{FDE}=\|\hat{p}_H-p_H^*\|_2
$$

Range: ([0,infinity)). Lower is better. FDE measures only the final point.

### 7.5. Dynamic Time Warping (DTW) distance

$$
\mathrm{DTW}(P,Q)=
\min_{\omega}\sum_{(i,j)\in\omega}d(p_i,q_j)
$$

Range: ([0,infinity)). Lower is better. DTW aligns trajectories that perform similar motions at different speeds.

### 7.6. Path length

$$
L=\sum_{t=2}^{H}\|p_t-p_{t-1}\|_2
$$

Range: ([0,infinity)). Shorter is usually more efficient, but the shortest path is not necessarily safest or smoothest.

### 7.7. Smoothness, acceleration, and jerk

Discrete velocity, acceleration, and jerk can be approximated by finite differences:

$$
v_t=\frac{p_t-p_{t-1}}{\Delta t},\qquad
a_t=\frac{v_t-v_{t-1}}{\Delta t},\qquad
j_t=\frac{a_t-a_{t-1}}{\Delta t}
$$

A common jerk score is:

$$
J=\frac{1}{H-2}\sum_t\|j_t\|_2^2
$$

Range: ([0,infinity)). Lower is smoother. Values are not comparable unless timestep, units, filtering, and trajectory duration match.

## 8. Real-time and compute metrics

### 8.1. End-to-end latency

$$
L_{\mathrm{e2e}} =
L_{\mathrm{camera}}+L_{\mathrm{preprocess}}+L_{\mathrm{VLM}}+
L_{\mathrm{action}}+L_{\mathrm{communication}}+L_{\mathrm{controller}}
$$

Range: ([0,infinity)) ms. Lower is better.

Report median, p95, and maximum latency. Mean latency alone hides missed control deadlines.

### 8.2. Control frequency

$$
f_{\mathrm{control}}=\frac{1}{\Delta t_{\mathrm{command}}}
$$

Range: ([0,infinity)) Hz. Higher can improve reactivity, but only if commands are stable and based on fresh observations.

### 8.3. Action-generation throughput

$$
\mathrm{Throughput}=\frac{\#\text{generated action steps}}{\text{second}}
$$

or chunks per second. Range: ([0,infinity)). Higher is better.

Throughput is not equal to control frequency. A model can generate a 50-step chunk quickly but only refresh its observation twice per second.

### 8.4. Observation age or staleness

$$
A_{\mathrm{obs}} = t_{\mathrm{execution}}-t_{\mathrm{capture}}
$$

Range: ([0,infinity)) ms. Lower is better. This is especially important for asynchronous action-chunk execution.

### 8.5. Deadline miss rate

$$
\mathrm{DMR}=\frac{\#\{i:L_i>B\}}{N}
$$

where (B) is the latency budget. Range: ([0,1]). Lower is better.

### 8.6. Real-time factor (RTF)

$$
\mathrm{RTF}=\frac{\text{simulated or generated duration}}
{\text{wall-clock duration}}
$$

Range: ([0,infinity)). RTF (>1) is faster than real time; RTF (<1) is slower.

### 8.7. Resource metrics

| Metric              | Unit/range                 | Direction               |
| ------------------- | -------------------------- | ----------------------- |
| Parameters          | count, ([0,infinity))      | Lower for equal quality |
| FLOPs per inference | operations, ([0,infinity)) | Lower for equal quality |
| Peak GPU memory     | GB, ([0,infinity))         | Lower                   |
| Energy per chunk    | joules, ([0,infinity))     | Lower                   |
| Training GPU-hours  | GPU-hours, ([0,infinity))  | Lower for equal quality |

Efficiency metrics must always be paired with task performance.

## 9. Robustness and generalization metrics

### 9.1. Seen and unseen success rates

Report separately:

$$
SR_{\mathrm{seen}},\qquad SR_{\mathrm{unseen}}
$$

Each lies in ([0,1]). Higher is better.

“Unseen” must specify what changed: object instance, category, instruction wording, scene, task, camera, or embodiment.

### 9.2. Generalization Gap

$$
\mathrm{Gap}=SR_{\mathrm{seen}}-
SR_{\mathrm{unseen}}
$$

Range: ([-1,1]). A value near zero is desirable only when both success rates are high. A gap of zero with both scores at zero is not useful.

### 9.3. Relative performance drop

$$
\mathrm{RelativeDrop}=
\frac{SR_{\mathrm{clean}}-SR_{\mathrm{shift}}}
{SR_{\mathrm{clean}}}
$$

Defined when (SR_{\mathrm{clean}}>0). A typical range is (( -infinity,1]); zero means no drop, positive means degradation, and negative means improvement under the shift. Lower is better.

### 9.4. Worst-group success

$$
\mathrm{WorstGroupSR}=\min_g SR_g
$$

Range: ([0,1]). Higher is better. This reveals weak conditions hidden by the overall mean.

### 9.5. Sample efficiency

Examples include:

- demonstrations required to reach a target SR;
- success per 100 demonstrations;
- learning-curve area versus number of episodes.

The natural count range is ([0,infinity)), with lower data requirements being better for a fixed performance target.

## 10. Safety, failure, and recovery metrics

### 10.1. Collision Rate

$$
\mathrm{CollisionRate}=
\frac{\#\text{episodes with collision}}{N}
$$

Range: ([0,1]). Lower is better.

Papers should distinguish object contact required by the task from unsafe collision.

### 10.2. Constraint Violation Rate

$$
\mathrm{ViolationRate}=
\frac{\#\text{violated monitored constraints}}
{\#\text{constraint opportunities}}
$$

Range: ([0,1]). Lower is better. Constraints may include joint, workspace, force, speed, or temporal safety rules.

### 10.3. Safe Success Rate

$$
\mathrm{SafeSR}=
\frac{1}{N}\sum_i
\mathbb{1}[\text{task success}_i \land \text{no safety violation}_i]
$$

Range: ([0,1]). Higher is better.

Always report ordinary SR beside SafeSR. Their difference is the rate of episodes that succeeded unsafely.

### 10.4. Intervention Rate

$$
\mathrm{InterventionRate}=
\frac{\#\text{episodes requiring human intervention}}{N}
$$

Range: ([0,1]). Lower is better.

An alternative is interventions per hour or per kilometre, with range ([0,infinity)).

### 10.5. Recovery Rate

$$
\mathrm{RecoveryRate}=
\frac{\#\text{failures successfully recovered}}
{\#\text{recoverable failures encountered}}
$$

Range: ([0,1]). Higher is better.

The denominator must be defined: all failures, injected failures, or failures judged recoverable.

### 10.6. Mean Time Between Failures (MTBF)

$$
\mathrm{MTBF}=\frac{\text{total operating time}}
{\#\text{failures}}
$$

Range: ([0,infinity)) time units. Higher is better.

## 11. Lifelong-learning metrics used by LIBERO

LIBERO originally evaluates lifelong learning, although modern VLA fine-tuning papers often use LIBERO tasks and report only mean task success. The original benchmark reports Forward Transfer, Negative Backward Transfer, and area under the success curve. [LIBERO paper](https://arxiv.org/abs/2306.03310)

Let (R_{i,j}) denote success on task (j) after learning through task (i).

### 11.1. Forward Transfer (FWT)

A common matrix-based form is:

$$
\mathrm{FWT}=\frac{1}{T-1}\sum_{j=2}^{T}
(R_{j-1,j}-b_j)
$$

where (b_j) is performance on task (j) without prior-task learning.

Range: ([-1,1]) when success is normalized. Higher is better. Positive FWT means earlier tasks help with a new task.

LIBERO's implementation emphasizes learning speed on the new task, so exact values depend on its checkpoint and learning-curve protocol.

### 11.2. Backward Transfer (BWT)

$$
\mathrm{BWT}=\frac{1}{T-1}\sum_{j=1}^{T-1}
(R_{T,j}-R_{j,j})
$$

Range: ([-1,1]). Higher is better.

- positive: later learning improves previous tasks;
- zero: previous performance is retained;
- negative: catastrophic forgetting.

### 11.3. Negative Backward Transfer (NBT)

A nonnegative forgetting form is:

$$
\mathrm{NBT}=\frac{1}{T-1}\sum_{j=1}^{T-1}
\max(0,R_{j,j}-R_{T,j})
$$

Range: ([0,1]). Lower is better. NBT is easier to read as “amount forgotten.”

### 11.4. Area Under the Success Curve (AUC)

For normalized training progress (x \in [0,1]):

$$
\mathrm{AUC}=\int_0^1 SR(x)\,dx
$$

Range: ([0,1]) when both axes are normalized. Higher is better. AUC rewards policies that learn quickly and retain performance.

## 12. Sim-to-real evaluation fidelity

SIMPLER uses rollout success rates inside simulation, but it evaluates the **quality of the simulator as a policy-ranking proxy** using correlation metrics. [SIMPLER paper](https://arxiv.org/abs/2405.05941)

Let (r_i) and (s_i) be real and simulated performance for policy (i).

### 12.1. Pearson correlation

$$
\rho =
\frac{\sum_i(r_i-\bar r)(s_i-\bar s)}
{\sqrt{\sum_i(r_i-\bar r)^2}
 \sqrt{\sum_i(s_i-\bar s)^2}}
$$

Range: ([-1,1]). Higher is better.

- (1): perfect positive linear relationship;
- (0): no linear relationship;
- (-1): perfectly reversed linear relationship.

### 12.2. Spearman rank correlation

Pearson applied to ranks:

$$
\rho_s=\mathrm{corr}(\mathrm{rank}(r),\mathrm{rank}(s))
$$

Range: ([-1,1]). Higher is better. It tests ordering rather than linear agreement.

### 12.3. Mean Maximum Rank Violation (MMRV)

Define a pairwise violation:

$$
V_{ij}=
\begin{cases}
|r_i-r_j|,&(r_i-r_j)(s_i-s_j)<0\\
0,&\text{otherwise}
\end{cases}
$$

Then:

$$
\mathrm{MMRV}=\frac{1}{M}\sum_{i=1}^{M}\max_j V_{ij}
$$

Range: ([0,1]) when real performance is in ([0,1]). Lower is better. MMRV penalizes a simulator more when it reverses policies whose real-world performance differs substantially.

These metrics evaluate the simulator, not the VLA directly.

## 13. Language, planning, and grounding metrics

These are used when a VLA or high-level module produces text, skills, parameters, boxes, or plans.

### 13.1. Skill Recall

$$
\mathrm{SkillRecall}=
\frac{|\text{predicted required skills}\cap\text{required skills}|}
{|\text{required skills}|}
$$

Range: ([0,1]). Higher is better.

### 13.2. Parameter Recall

$$
\mathrm{ParameterRecall}=
\frac{\#\text{correct required arguments}}
{\#\text{required arguments}}
$$

Range: ([0,1]). Higher is better.

### 13.3. Skill-and-parameter recall

$$
\mathrm{PairRecall}=
\frac{\#\text{correct skill-argument pairs}}
{\#\text{required skill-argument pairs}}
$$

Range: ([0,1]). Higher is better.

### 13.4. Precise or Exact Matching Rate

$$
\mathrm{PM}=
\frac{\#\text{outputs exactly matching the required program or plan}}
{N}
$$

Range: ([0,1]). Higher is better.

VLABench uses these metrics for VLM/skill-library evaluations, not as a replacement for physical rollout success.

### 13.5. Object-selection accuracy

$$
\mathrm{SelectionAcc}=
\frac{\#\text{correct target-object selections}}{N}
$$

Range: ([0,1]). Higher is better.

### 13.6. Bounding-box IoU

$$
\mathrm{IoU}=\frac{|B_{\mathrm{pred}}\cap B_{\mathrm{gt}}|}
{|B_{\mathrm{pred}}\cup B_{\mathrm{gt}}|}
$$

Range: ([0,1]). Higher is better. Used for grounding/localization modules rather than complete VLA control.

## 14. Navigation metrics sometimes used with VLAs

### 14.1. Navigation Success Rate

The fraction of episodes ending within the goal tolerance. Range: ([0,1]). Higher is better.

### 14.2. Distance to Goal

$$
d_{\mathrm{goal}}=\|p_{\mathrm{final}}-p_{\mathrm{goal}}\|
$$

Range: ([0,infinity)) metres. Lower is better.

### 14.3. Success weighted by Path Length (SPL)

$$
\mathrm{SPL}=\frac{1}{N}\sum_{i=1}^{N}
S_i\frac{L_i^*}{\max(L_i,L_i^*)}
$$

where (L_i^*) is the shortest path and (L_i) is the executed path.

Range: ([0,1]). Higher is better. SPL is zero for failed episodes and rewards successful, efficient paths.

### 14.4. Navigation Collision Rate

Collisions per episode or collision episodes divided by episodes. The first has range ([0,infinity)); the second ([0,1]). Lower is better.

## 15. Statistical reporting

### 15.1. Standard deviation

$$
s=\sqrt{\frac{1}{K-1}\sum_{k=1}^{K}(x_k-\bar{x})^2}
$$

Measures variation across seeds, tasks, or evaluation batches. Range: ([0,infinity)). Lower means more consistency, not necessarily better average performance.

### 15.2. Standard error

$$
\mathrm{SE}=\frac{s}{\sqrt{K}}
$$

Estimates uncertainty in the reported mean. Range: ([0,infinity)). It decreases with more independent runs.

### 15.3. Binomial confidence interval for success rate

An approximate 95% interval is:

$$
\hat p \pm 1.96\sqrt{\frac{\hat p(1-\hat p)}{N}}
$$

Wilson or exact intervals are preferable for small (N) or values near 0 and 1.

Example: 8 successes in 10 trials and 800 successes in 1,000 trials both give 80%, but the second estimate is much more precise.

### 15.4. Number of rollouts

Always report (N) per task, number of tasks, number of seeds, and whether resets are deterministic. A metric without its sample count is incomplete.

## 16. Major benchmark-to-metric mapping

| Benchmark                           | Main VLA metric                   | Additional metrics                                               | Important range                                 |
| ----------------------------------- | --------------------------------- | ---------------------------------------------------------------- | ----------------------------------------------- |
| **LIBERO**                    | Per-task and mean Success Rate    | Original lifelong setting: FWT, NBT, AUC                         | SR ([0,1]); FWT typically ([-1,1]); NBT ([0,1]) |
| **CALVIN**                    | Average completed sequence length | Success at chain lengths 1-5                                     | AvgLen ([0,5]); each prefix SR ([0,1])          |
| **RLBench**                   | Task Success Rate                 | Mean over tasks, few-shot/task-specific breakdown                | ([0,1])                                         |
| **RoboCasa**                  | Success Rate                      | Seen/unseen breakdown, composite-task results                    | ([0,1])                                         |
| **SimplerEnv**                | Simulated task Success Rate       | Pearson correlation and MMRV against real policy rankings        | SR ([0,1]), Pearson ([-1,1]), MMRV ([0,1])*     |
| **VLABench**                  | Progress Score                    | Seen/unseen scores; VLM skill/parameter recall and precise match | Usually ([0,1]) or 0-100                        |
| **ManiSkill / Meta-World**    | Success Rate or Return            | Per-task mean, normalized score                                  | SR ([0,1]); return environment-specific         |
| **Real-robot custom tasks**   | Success Rate                      | Completion time, interventions, collisions, precision            | Metric-specific                                 |
| **Navigation VLA benchmarks** | Success and SPL                   | Distance-to-goal, path length, collisions                        | Success/SPL ([0,1])                             |

The original benchmark papers are [LIBERO](https://arxiv.org/abs/2306.03310), [CALVIN](https://arxiv.org/abs/2112.03227), [RLBench](https://arxiv.org/abs/1909.12271), [RoboCasa](https://arxiv.org/abs/2406.02523), [SIMPLER](https://arxiv.org/abs/2405.05941), and [VLABench](https://arxiv.org/abs/2412.18194).

## 17. Recommended minimum evaluation set

A strong manipulation VLA report should include at least:

1. **Full-task Success Rate** per task and macro-average.
2. **Progress Score** for long-horizon tasks.
3. **Seen and unseen Success Rates**, with the distribution shift defined.
4. **Collision or safety-violation rate**.
5. **Recovery rate** or intervention rate.
6. **Median and p95 end-to-end latency** on the deployment hardware.
7. **Control frequency**, action-chunk length, and number of executed actions per chunk.
8. **Number of rollouts, seeds, and confidence intervals**.
9. **Offline action error or likelihood**, only as a diagnostic rather than the main result.
10. **Compute and memory use** when claiming efficiency.

## 18. How to read a VLA result table

Suppose a paper reports:

```text
LIBERO mean success:        92%
Unseen-scene success:       61%
Collision rate:              8%
Median / p95 latency:       45 / 92 ms
Control frequency:          20 Hz
Evaluation:                 20 rollouts per task, 10 tasks, 3 seeds
```

Interpretation:

- The policy is strong on its main benchmark.
- Its generalization gap is (0.92-0.61=0.31), which is substantial.
- Eight percent of episodes contain a collision even if some eventually succeed.
- A 20 Hz loop has a nominal 50 ms period. Median latency fits, but p95 latency misses that deadline, so asynchronous chunk execution or a lower refresh rate is required.
- The sample count is large enough to be more credible than a result based on a handful of hand-selected trials.

No single metric answers whether a VLA is good. The most informative combination is:

> **task success + partial progress + generalization + safety + real-time performance + statistical uncertainty.**
