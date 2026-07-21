# VLA Basics: From Seeing and Language to Physical Action

## 1. The main idea

A **Vision-Language-Action (VLA)** model is a VLM that does not stop at understanding or answering. It uses what the robot sees and what a person asks, then decides **how the robot should move**.

The simplest comparison is:

| Model | Input                                         | Output       |
| ----- | --------------------------------------------- | ------------ |
| VLM   | Image + text                                  | Text         |
| VLA   | Image + instruction + current robot condition | Robot action |

A useful one-line definition is:

> **VLA turns “I see the situation and understand the goal” into “this is the movement I should make next.”**

## 2. Inputs

At one moment, the model commonly receives:

- **What the robot sees:** one or more camera images.
- **What the human wants:** for example, “put the red cup in the sink.”
- **The robot's current condition:** where its arm and gripper are, whether the gripper is open, and sometimes recent images or movements.

Some systems also use touch, depth, or sound, but these are additions rather than the core idea.

## 3. Runtime dataflow

The robot repeatedly follows this loop:

1. **Observe:** look at the current scene.
2. **Connect the request to the scene:** identify the relevant cup, sink, obstacles, and robot position.
3. **Choose the next movement:** reach, grasp, lift, move, release, or another small step.
4. **Act:** send the movement to the robot.
5. **Look again:** check what changed and choose the next movement.

This repetition is essential. The model does not merely create one answer and finish. Its action changes the world, so it must observe the result and continue from the new situation.

Some VLAs output one small movement at a time. Others output a short group of movements, execute part of it, then look again.

## 4. Outputs

The final output must be something the robot can execute. For a robot arm, it may describe:

- how the hand should move in 3D;
- how the wrist should rotate;
- whether the gripper should open or close.

For a mobile robot, it may instead describe wheel movement, direction, or speed. Therefore, the exact output depends on the robot body.

Not every VLA goes directly from input to motor commands. A system may first produce an intermediate result such as:

- a short verbal plan: “grasp the cup, then move to the sink”;
- a target point: “grasp here”;
- a path or desired future image.

That intermediate result is then converted into executable movement. The attached survey groups these different forms under the broad idea of **action tokens**: information that becomes progressively more useful for producing action.

## 5. How a VLA learns

The most direct training data is a collection of demonstrations. Each moment records:

| Instruction             | What the robot saw | Robot condition          | Movement performed       |
| ----------------------- | ------------------ | ------------------------ | ------------------------ |
| “Pick up the red cup” | Camera image       | Arm and gripper position | Move hand toward the cup |

Across many demonstrations, the model learns: **given this request and this situation, predict the movement that a successful operator made.**

The VLM foundation contributes broad visual and language knowledge. Robot demonstrations teach the missing connection between that knowledge and physical movement. This robot data is much harder and more expensive to collect than ordinary image-text data, which is one of the main limits of current VLA research.

## 6. Concrete example

Task: **“Put the red cup in the sink.”**

| Moment | Model sees               | Model outputs             |
| ------ | ------------------------ | ------------------------- |
| 1      | Cup is far from the hand | Move toward the cup       |
| 2      | Hand is beside the cup   | Align fingers around it   |
| 3      | Fingers surround the cup | Close gripper             |
| 4      | Cup is held              | Lift and move toward sink |
| 5      | Cup is above sink        | Open gripper              |

The instruction remains the same, but the correct output changes after every observation.

## 7. The mental model to keep

- A **VLM** answers: “What is here, and what does the request mean?”
- A **VLA** answers: “Given that understanding, what should this body do next?”
- A VLA is a **continuous observe-act-observe loop**, not a one-shot captioning task.
- Its output is tied to a particular robot body. The same goal may require different movements on different robots.
- Its key training example is the relationship between **instruction, observation, and successful action**.

In Transformer terms, the easiest starting intuition is: a VLA keeps the visual-language understanding you already know, but extends prediction from words to actions that change the next input.

## 8. Core vocabulary used in VLA research

These terms do not appear in literally every paper, but they form the common vocabulary of the field.

| Term                                            | Plain meaning                                                                                                                                                                     | Example                                                    |
| ----------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| **Observation**                           | Everything available to the model at the current moment. It commonly includes camera images and the robot's own condition.                                                        | Current image + arm position                               |
| **Instruction**                           | The language description of the task or goal.                                                                                                                                     | “Put the cup in the sink.”                               |
| **State**                                 | A description of the current situation. The full state of the real world is usually unknown, so the model acts from its limited observation.                                      | Cup position, arm position, gripper condition              |
| **Proprioception**                        | The robot's sense of its own body, rather than the outside world.                                                                                                                 | Joint angles, hand position, gripper opening               |
| **Action**                                | A command that the robot can execute.                                                                                                                                             | Move the hand 2 cm forward and close the gripper           |
| **Action space**                          | The complete set and format of actions available to a robot.                                                                                                                      | Arm movement, wrist rotation, and gripper control          |
| **Policy**                                | The learned decision-maker that maps the current observation and instruction to an action. In most VLA papers, “the policy” means the part that controls behavior.              | `(image, instruction, robot condition) → next movement` |
| **Embodiment**                            | The particular body through which the model acts. Different arms, grippers, and mobile robots have different abilities and action spaces.                                         | A two-finger arm versus a humanoid robot                   |
| **Timestep**                              | One moment in the repeated observe-and-act process.                                                                                                                               | Observe at time`t`, then predict action `t`            |
| **Control frequency**                     | How often the system produces or updates robot commands, measured in times per second (Hz). Higher is not automatically better, but fine movement usually needs frequent updates. | 10 Hz means ten updates per second                         |
| **Trajectory**                            | An ordered sequence of observations and actions through time. It describes how behavior unfolds, not only the final result.                                                       | Reach → grasp → lift → move → release                  |
| **Episode**                               | One complete attempt at a task, from its initial situation until success, failure, or a time limit.                                                                               | One attempt to put the cup in the sink                     |
| **Rollout**                               | An episode produced by running the policy. Researchers inspect rollouts to see what the model actually does.                                                                      | Run the trained model on a robot for one attempt           |
| **Demonstration**                         | An example of successful or useful behavior, usually collected from a human controlling a robot or from another controller.                                                       | A human teleoperates the arm to pick up a cup              |
| **Imitation learning / behavior cloning** | Training the model to predict the actions found in demonstrations. It is similar to supervised learning over observation-action examples.                                         | Learn to copy the operator's next movement                 |
| **Action token**                          | A general unit of action-related output. Depending on the paper, it can be a direct motor command, a target point, a path, a goal image, or an internal representation.           | “Close gripper,” a 3D grasp point, or a movement value   |
| **Action chunk**                          | Several future actions predicted together instead of one action at a time. This can make control faster and smoother, but later actions may become outdated if the world changes. | Predict the next 20 arm commands together                  |
| **Closed-loop control**                   | The robot repeatedly observes the result of its actions and adjusts.                                                                                                              | Move toward cup, look again, then correct alignment        |
| **Open-loop control**                     | The robot executes a prepared action sequence without using new observations to correct it during execution.                                                                      | Execute all 20 predicted commands without looking again    |
| **Generalization**                        | The ability to succeed beyond the exact demonstrations used for training.                                                                                                         | Pick up a new cup in a new kitchen                         |
| **Success rate**                          | The fraction of task attempts completed successfully. This is the most common top-level evaluation measure.                                                                       | 82 successful episodes out of 100 = 82%                    |

### 8.1. Robot-arm terms

| Term                               | Plain meaning                                                                                                                              |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| **Joint**                    | A movable connection in the robot arm. Joint-based control directly specifies how these joints should move.                                |
| **End effector**             | The working part at the end of the arm, usually a hand, gripper, or tool.                                                                  |
| **Pose**                     | Position plus orientation. An end-effector pose says both where the hand is and which way it points.                                       |
| **Degrees of freedom (DoF)** | The number of independently controllable movement dimensions. A 6-DoF hand pose normally has three position and three rotation dimensions. |
| **Gripper action**           | A command controlling the robot's grasping tool, commonly open/close or a continuous opening value.                                        |

### 8.2. The vocabulary in one sentence

> At every **timestep**, the VLA **policy** receives an **observation**, language **instruction**, and **proprioception**, then predicts an **action** or **action chunk** within the robot's **action space**; running this repeatedly produces a **rollout trajectory**, whose result is measured by **success rate**.

## 9. Core challenges and constraints

A VLA must combine semantic understanding with precise, timely, and safe physical control. A model can understand the instruction correctly and still fail because its movement is late, imprecise, unsafe, incompatible with the robot, or unable to recover from a small mistake.

The main challenge groups are:

| Area                  | Central question                                                              |
| --------------------- | ----------------------------------------------------------------------------- |
| Data                  | Does the training data cover the tasks, scenes, mistakes, and robot body?     |
| Perception            | Can the model identify the correct object and its usable 3D location?         |
| Language grounding    | Does it connect the instruction to the right object, relation, and behavior?  |
| Action generation     | Are the commands smooth, precise, and valid for this robot?                   |
| Real-time control     | Are actions produced before the observation becomes outdated?                 |
| Long-horizon behavior | Can the model track progress, sequence subtasks, and recover from errors?     |
| Generalization        | Does it work outside the exact conditions represented in training?            |
| Safety                | Can failures be detected and contained before causing damage?                 |
| Evaluation            | Are we measuring real task success rather than only offline prediction error? |

### 9.1. Data and learning constraints

#### 9.1.1. Robot data is scarce and expensive

Image-text data can be collected from the internet, but useful robot training data normally requires a real robot, a simulator, or a human operator. Real-world demonstrations are slow to collect and may include hardware wear, operator inconsistency, sensor errors, and failed episodes.

The important constraint is that a VLA cannot learn physical skills from VLM knowledge alone. A VLM may know that a mug has a handle, but robot data must teach how this embodiment should approach, grasp, lift, and recover if the mug slips.

Common responses include:

- pooling data from multiple robots;
- generating simulation or synthetic data;
- learning from human videos;
- pretraining broadly, then fine-tuning on a smaller robot-specific dataset;
- collecting corrective and recovery behavior, not only perfect demonstrations.

#### 9.1.2. Data quality and coverage matter as much as volume

A dataset containing only successful, clean demonstrations teaches the policy what ideal behavior looks like, but not how to recover after drifting away from that behavior. At deployment time, small errors move the robot into situations that may not exist in the demonstrations. This is called **distribution shift**.

Training data should cover:

- different object positions, appearances, and backgrounds;
- variations in lighting, camera view, and clutter;
- multiple valid ways of completing the task;
- partial failures and corrective movements;
- task beginnings, transitions, and endings;
- safe stopping behavior when the task becomes impossible.

#### 9.1.3. Cross-embodiment data is not automatically compatible

Different robots have different cameras, joint counts, grippers, coordinate systems, and action meanings. Combining their data requires a shared representation or robot-specific adapters.

For example, the numeric action `[0.1, 0, 0]` might mean a 10 cm Cartesian movement in one dataset, a normalized joint command in another, and a velocity in a third. Mixing them without correct metadata and conversion makes the training target meaningless.

#### 9.1.4. VLM knowledge can be forgotten

Fine-tuning heavily on robot trajectories can damage the general visual-language capabilities inherited from the original VLM. The model may improve on one robot benchmark while becoming worse at recognizing unusual objects, reading labels, or interpreting new instructions.

Recent training recipes therefore mix vision-language data with robot data, freeze or isolate parts of the VLM, or use a separate action expert so that motor learning does not overwrite all semantic knowledge.

### 9.2. Perception and grounding challenges

#### 9.2.1. 2D understanding is not enough for precise manipulation

An RGB image provides strong semantic information but does not directly provide exact depth, contact geometry, or object mass. A VLA must infer where to touch, how far to move, and whether the gripper can reach without collision.

This becomes difficult with:

- transparent, reflective, deformable, or very small objects;
- clutter and occlusion;
- similar-looking objects;
- poor lighting or motion blur;
- tasks requiring millimetre-level alignment;
- camera views that do not show the contact area.

Depth cameras, wrist cameras, point clouds, force sensing, and calibrated geometry can help, but each adds hardware and synchronization constraints.

#### 9.2.2. Language must be grounded to the current scene

The model must resolve which physical object and relation an instruction refers to. “Put it there,” “use the clean cup,” or “move the nearest block” require context, memory, comparison, or clarification.

A VLA can fail even when it recognizes every object if it selects the wrong instance, misunderstands left/right from the user's viewpoint, or interprets “on” differently from the task evaluator.

Instructions used during deployment should match the capabilities and language diversity represented during training. Safety-critical ambiguity should trigger clarification or refusal rather than a confident movement.

#### 9.2.3. Partial observability and temporal memory

The current camera image may not reveal everything needed for the task. An object can move behind the arm, a drawer can hide its contents, or the robot may need to remember which items were already handled.

A single-frame VLA can repeatedly open the same drawer or forget that it has already completed a subtask. Longer image history, explicit memory, object tracking, and high-level state tracking can help, but increase computation and context length.

### 9.3. Action-generation and control constraints

#### 9.3.1. Action representation must match the controller

Before training or deployment, the action interface must be defined exactly:

| Choice          | Examples                                                                    |
| --------------- | --------------------------------------------------------------------------- |
| Control target  | Joint position, joint velocity, end-effector pose, or end-effector movement |
| Reference frame | Robot base, world, camera, or end-effector frame                            |
| Time meaning    | Absolute target, change from current value, or velocity per second          |
| Rotation format | Euler angles, quaternion, axis-angle, or 6D rotation                        |
| Gripper format  | Binary open/close, continuous width, force, or velocity                     |
| Units           | Metres versus millimetres; radians versus degrees                           |

A mismatch can make a good model appear completely broken and can also damage hardware. Action normalization statistics must come from the correct dataset and must be reversed correctly before sending commands to the robot.

#### 9.3.2. Smoothness versus responsiveness

Predicting an action chunk produces smoother motion and reduces how often a large VLA must run. However, a long chunk may continue toward an old target after the object or robot has moved.

The trade-off is:

- **long chunks:** faster average execution and smoother trajectories, but less reactive;
- **short chunks:** more reactive, but require more frequent inference and may produce discontinuities between chunks.

Most systems predict a chunk, execute only part of it, observe again, and replace the unexecuted remainder.

#### 9.3.3. Several actions may all be correct

There may be multiple valid grasps and paths for the same observation. A simple average of those demonstrations can create an invalid movement between them. For example, averaging a left-side approach and a right-side approach may drive the hand directly into the object.

Diffusion and flow-matching action generators are popular partly because they can represent complex continuous action distributions. Even then, generated chunks must remain consistent across time.

#### 9.3.4. Small errors accumulate

A 2 mm error may be harmless during an approach but critical when inserting a plug or closing a zipper. Repeating slightly incorrect predictions can gradually move the robot far outside the training distribution.

Closed-loop observation, visual correction, force feedback, precise calibration, and recovery data are therefore as important as the initial action prediction.

### 9.4. Real-time performance

Real-time performance is not simply “high GPU utilization” or a large number of generated actions per second. The complete delay from sensing to useful robot command matters:

```text
camera exposure and transfer
  + image preprocessing
  + VLM inference
  + planning, if used
  + action generation
  + network or process communication
  + controller delay
  = end-to-end control latency
```

The control period gives the available time budget:

| Desired update rate | Maximum period per update |
| ------------------: | ------------------------: |
|               10 Hz |                    100 ms |
|               20 Hz |                     50 ms |
|               50 Hz |                     20 ms |
|              120 Hz |              about 8.3 ms |

These are timing budgets, not universal VLA requirements. A large VLA may run at a lower rate while a lightweight low-level controller interpolates or executes a predicted action chunk at a higher rate.

#### 9.4.1. Why latency causes physical failure

If the camera image was captured at time `t` but the command reaches the robot much later, it is based on a stale observation. Meanwhile, the arm, target object, or human may have moved. This can cause oscillation, overshoot, collisions, jerky transitions, or repeated corrections.

Important measurements include:

- median and worst-case sensing-to-action latency;
- VLM and action-head latency separately;
- effective control frequency on the actual robot;
- missed deadlines and latency variation, also called jitter;
- time spent transferring images and tensors;
- how old the observation is when its action is executed.

#### 9.4.2. Common real-time strategies

- Cache instruction and image features that do not need recomputation.
- Predict several actions in parallel as a chunk.
- Use a smaller action expert after a larger VLM context pass.
- Quantize or compile the model when validation shows acceptable behavior.
- Reduce image count or resolution carefully.
- Run inference asynchronously while the robot executes the current chunk.
- Reuse part of the previous chunk to preserve continuity.
- Keep fast safety and motor-control loops outside the large VLA.

Asynchronous execution improves throughput but makes timing alignment harder. The system must know which predicted timestep corresponds to the robot's actual state when the new chunk begins.

### 9.5. Long-horizon planning and recovery

#### 9.5.1. Long tasks amplify every weakness

If an individual subtask succeeds 95% of the time, a task requiring 20 independent successful subtasks would have an idealized total success probability of approximately:

$$
0.95^{20} \approx 0.36
$$

Real subtasks are not independent, but the example shows why a strong short-horizon policy can still perform poorly on room cleaning or meal preparation.

Long-horizon systems need to determine:

- which subtask should happen next;
- whether the current subtask is complete;
- what has already been completed;
- when to retry, choose another strategy, ask for help, or stop.

An explicit planner can improve progress tracking but adds another source of errors and latency. An implicit end-to-end policy is simpler but harder to inspect and may repeat behavior.

#### 9.5.2. Recovery must be learned or designed

Robots should detect and recover from cases such as:

- failed grasp;
- dropped or moved object;
- blocked path;
- drawer that did not open;
- object hidden by the robot's arm;
- controller timeout;
- unexpected human interaction.

A model trained only on successful trajectories often continues as though the failed action succeeded. Recovery data, success detectors, retry limits, progress checks, and safe fallback behavior should be part of the system design.

### 9.6. Generalization and deployment constraints

#### 9.6.1. Generalization has several levels

A model may generalize to a new position but not a new object, or to a new object but not a new robot. Papers should specify which variation is unseen:

- new object instances;
- new object categories;
- new backgrounds or lighting;
- new camera positions;
- new instruction wording;
- new task combinations;
- new environments;
- new robot embodiments.

Randomly splitting nearby frames from the same trajectory between training and evaluation can give an unrealistically high estimate of generalization. Evaluation should separate entire scenes, tasks, objects, or robot runs.

#### 9.6.2. Simulation does not perfectly transfer to reality

Simulation may have different textures, lighting, friction, contact behavior, camera noise, and actuator delay. A policy can score highly in simulation yet fail on real hardware.

Domain randomization, real-world fine-tuning, accurate sensor models, and conservative controllers reduce this gap but do not eliminate it.

#### 9.6.3. Hardware and calibration are part of the model system

VLA performance depends on factors outside the neural network:

- camera calibration and mounting stability;
- timestamp synchronization;
- robot kinematics and controller tuning;
- gripper force and mechanical wear;
- network reliability;
- available GPU memory and thermal limits;
- emergency stop and collision detection.

A camera moved after calibration or a different gripper can invalidate the relationship learned between pixels and actions.

### 9.7. Safety and reliability

A VLA is probabilistic and may produce an unexpected command even for a familiar input. A deployed system should not send unconstrained predictions directly to motors.

Common safety layers include:

- joint, velocity, acceleration, and force limits;
- workspace and self-collision constraints;
- human detection and protected zones;
- action validity checks;
- watchdog timeouts;
- emergency stop;
- confidence or uncertainty checks;
- human approval for high-risk actions;
- logs that connect observations, predictions, and executed commands.

The safety layer should run independently and faster than the VLA. Language instructions must not be allowed to override physical safety limits.

### 9.8. What does “accuracy” mean for a VLA?

There is no single VLA accuracy number comparable to image-classification accuracy.

#### 9.8.1. Offline action-prediction error

During training, researchers can measure how closely a predicted action matches a recorded demonstration using L1, L2, or token accuracy. This is useful for optimization but is not sufficient for judging the robot.

Low offline error may still give poor rollouts because:

- several different actions may all be valid;
- a small error can compound after many steps;
- the model is evaluated on demonstration states rather than states created by its own mistakes;
- an action close to the operator's action may still cross a contact or safety boundary.

#### 9.8.2. Task success rate

The main top-level metric is normally:

$$
\text{Success rate} = \frac{\text{successful episodes}}{\text{total evaluated episodes}}
$$

Success criteria must be defined precisely. “Place the mug on the tray” may require the mug to be fully inside a marked area, upright, released, and stable for several seconds.

Success rate alone can also hide important differences. A complete evaluation should include:

| Metric                                   | What it reveals                             |
| ---------------------------------------- | ------------------------------------------- |
| Full-task success                        | Whether the final goal was completed        |
| Subtask or stage completion              | Where long tasks fail                       |
| Position and rotation error              | Physical precision at the goal              |
| Completion time                          | Efficiency                                  |
| Path length or motion smoothness         | Wasteful or jerky behavior                  |
| Collision and safety-violation rate      | Risk                                        |
| Intervention rate                        | How often a human must rescue the robot     |
| Recovery success                         | Whether the model can correct a failed step |
| Unseen-condition success                 | Generalization                              |
| End-to-end latency and control frequency | Real-time feasibility                       |

#### 9.8.3. Precision is task-dependent

The required spatial accuracy differs greatly by task. Moving a towel into a basket may tolerate centimetres of error. Inserting a connector may require millimetre-level position accuracy and tight rotation alignment.

Therefore, a paper's success rate should always be interpreted together with:

- the difficulty and tolerance of the task;
- whether evaluation is in simulation or on a real robot;
- how many episodes and random seeds were used;
- whether scenes, objects, and instructions were truly unseen;
- whether humans reset, assist, or select favorable trials.

### 9.9. Practical constraints checklist

Before training or running a VLA, verify:

- **Input contract:** camera count, order, resolution, color format, crop, and history length.
- **Timing contract:** camera timestamps, model rate, controller rate, chunk length, and execution delay.
- **State contract:** joint ordering, missing sensors, units, normalization, and coordinate frames.
- **Action contract:** absolute versus relative commands, control mode, rotation format, units, and gripper semantics.
- **Embodiment contract:** robot ID, kinematics, workspace, and embodiment-specific adapters.
- **Language contract:** prompt template, supported tasks, ambiguity handling, and stop commands.
- **Compute contract:** GPU memory, worst-case latency, numerical precision, and thermal stability.
- **Safety contract:** limits, collision checking, watchdog, emergency stop, and human supervision.
- **Evaluation contract:** exact success definition, unseen split, number of rollouts, failure categories, and latency measurement.
- **Logging contract:** save the input observation, model output, decoded command, executed command, timestamps, and outcome for every failure analysis.

## 10. Sources

Based primarily on *A Survey on Vision-Language-Action Models: An Action Tokenization Perspective*, especially the unified framework on page 1, the overview of action tokens on page 12, and the discussion of direct robot actions on pages 31-36. The real-time and modern action-generation discussion is also informed by the original [π0](https://arxiv.org/abs/2410.24164), [GR00T N1](https://arxiv.org/abs/2503.14734), [OpenVLA-OFT](https://arxiv.org/abs/2502.19645), and [Xiaomi-Robotics-0](https://arxiv.org/abs/2602.12684) reports.
