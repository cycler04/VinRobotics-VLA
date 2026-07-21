# Modern Action Space Representations in Vision-Language-Action (VLA) Models

## Overview

An **action space** defines **what the policy predicts** as the output of a Vision-Language-Action (VLA) model.

It is important to distinguish **task type** from **action space**.

```text
Instruction
        ↓
Task
        ↓
Action Space
        ↓
Robot Controller
        ↓
Robot Motion
```

Example:

```text
Instruction:
"Pick up the red mug."

Task:
Manipulation

Action Space:
[
 Δx,
 Δy,
 Δz,
 Δroll,
 Δpitch,
 Δyaw,
 gripper
]
```

Different VLA models may solve the same task while using completely different action spaces.

---

# Taxonomy of Modern Action Spaces

```text
Action Spaces
│
├── Joint-space
│     ├── Joint Position
│     ├── Joint Velocity
│     └── Joint Torque
│
├── Cartesian-space
│     ├── End-Effector Pose
│     ├── End-Effector Delta
│     ├── Bimanual Pose
│     └── Dexterous Hand
│
├── Mobile-space
│     ├── Base Velocity
│     ├── Navigation Waypoint
│     └── Driving Controls
│
├── Trajectory-space
│     ├── Action Chunk
│     ├── Human Trajectory
│     └── Future Path
│
└── Latent-space
      └── Learned Motion Embedding
```

---

# 1. Joint Position Space

The policy predicts desired joint positions directly.

```text
[
 joint1,
 joint2,
 joint3,
 ...
 jointN
]
```

Example

```text
[
0.31,
-1.08,
0.72,
...
]
```

The controller simply moves each motor to the desired angle.

### Advantages

- Simple
- Accurate
- Direct hardware control

### Disadvantages

- Robot-specific
- Difficult to transfer across embodiments

Typical robots

- Franka
- UR5
- xArm
- Many teleoperation datasets

---

# 2. Joint Velocity Space

Instead of positions, predict joint velocities.

```text
[
 joint1_velocity,
 joint2_velocity,
 ...
]
```

Example

```text
[
0.3,
-0.2,
0.4
]
```

Controller integrates velocity into future joint positions.

### Advantages

- Smooth control
- Continuous servoing

### Disadvantages

- Integration drift
- Requires current robot state

---

# 3. Joint Torque Space

Lowest-level action representation.

```text
[
 torque1,
 torque2,
 ...
]
```

Example

```text
[
2.1,
-0.8,
4.3
]
```

Directly controls motor torques.

### Advantages

- Maximum control authority
- Suitable for dynamic behaviors

### Disadvantages

- Extremely robot-dependent
- Difficult to train
- Sensitive to latency

Mostly used in

- Reinforcement Learning
- Humanoid locomotion

---

# 4. End-Effector Pose Space

Predict the robot hand pose instead of joints.

```text
[
 x,
 y,
 z,
 roll,
 pitch,
 yaw,
 gripper
]
```

Sometimes orientation uses quaternion

```text
[
 x,
 y,
 z,
 qx,
 qy,
 qz,
 qw,
 gripper
]
```

The robot solves inverse kinematics afterwards.

### Advantages

- More transferable
- Easier across robot arms

### Disadvantages

- Requires IK solver

---

# 5. End-Effector Delta Space

The most common action representation in modern VLAs.

Instead of absolute position:

```text
[
 Δx,
 Δy,
 Δz,
 Δroll,
 Δpitch,
 Δyaw,
 gripper
]
```

Example

```text
[
+2 cm,
-1 cm,
0,
0,
0,
5°,
close
]
```

Each action is a small correction from the current pose.

### Advantages

- Stable
- Reactive
- Easy to learn
- Better transferability

Used by

- π0
- π0.5
- OpenVLA
- RT-2 (discretized)
- DiffusionVLA

---

# 6. Bimanual Action Space

For robots with two arms.

```text
[
 left arm pose,

 right arm pose,

 left gripper,

 right gripper
]
```

Example

```text
[
Left EE,

Right EE,

Open,

Close
]
```

Common for

- ALOHA
- Mobile ALOHA
- Bimanual humanoids

---

# 7. Dexterous Hand Space

Instead of one gripper value, every finger is controlled.

```text
[
 arm,

 thumb joints,

 index joints,

 middle joints,

 ring joints,

 little finger joints
]
```

May contain

- 20
- 30
- 40+
  dimensions

Typical robots

- Shadow Hand
- Inspire Hand
- Allegro Hand

---

# 8. Navigation Waypoint Space

Common for mobile robots.

```text
[
 Δx,
 Δy,
 Δheading
]
```

Example

```text
[
1.2 m,
0.4 m,
20°
]
```

Represents the next waypoint rather than wheel commands.

### Advantages

- High-level
- Robot-independent

Used by

- Mobile robots
- Indoor navigation
- Qwen-VLA navigation

---

# 9. Base Velocity Space

Lower-level mobile control.

```text
[
 linear velocity,

 angular velocity
]
```

Usually

```text
[
v,
ω
]
```

The robot converts these into wheel speeds.

Typical robots

- Differential drive
- ROS navigation stack

---

# 10. Driving Control Space

Autonomous vehicles.

```text
[
 steering,

 throttle,

 brake
]
```

or

```text
[
 steering,

 acceleration
]
```

Examples

- End-to-end driving
- Autonomous cars

---

# 11. Human Trajectory Space

Instead of robot commands, predict future human motion.

```text
[
 wrist pose,

 hand pose,

 body pose
]
```

Often represented as

- wrist translation
- wrist rotation
- hand articulation
- body joints

Applications

- Imitation learning
- Human motion prediction
- EgoVLA
- Qwen-VLA

---

# 12. Action Chunk Space

Instead of predicting one action,

predict multiple future actions simultaneously.

Instead of

```text
Action
```

predict

```text
[
Action_t,

Action_t+1,

Action_t+2,

...

Action_t+15
]
```

Mathematically

$$
A\in\mathbb{R}^{H\times D}
$$

where

- H = prediction horizon
- D = action dimension

Example

```text
[
 [Δx Δy Δz g],
 [Δx Δy Δz g],
 [Δx Δy Δz g],
 ...
]
```

Advantages

- Lower inference frequency
- Smoother motion
- Better temporal consistency

This is now the dominant output format for diffusion and flow-based VLAs.

---

# 13. Latent Action Space

Instead of predicting physical commands,

the policy predicts a learned embedding.

```text
[
 z1,
 z2,
 ...
 z128
]
```

A separate controller decodes

```text
Latent

↓

Robot actions
```

Advantages

- Compact
- Robot-independent
- Hierarchical control

Disadvantages

- Hard to interpret
- Requires decoder

---

# Comparison

| Action Space        | Typical Dimension | Controller Needed        | Cross-Robot Transfer     |
| ------------------- | ----------------: | ------------------------ | ------------------------ |
| Joint Position      |             6–40 | No                       | Poor                     |
| Joint Velocity      |             6–40 | Small                    | Poor                     |
| Joint Torque        |             6–40 | Minimal                  | Very Poor                |
| End-Effector Pose   |              7–8 | IK                       | Good                     |
| End-Effector Delta  |                 7 | IK                       | Excellent                |
| Bimanual            |            14–20 | IK                       | Good                     |
| Dexterous Hand      |            20–50 | IK                       | Moderate                 |
| Navigation Waypoint |                 3 | Navigation Controller    | Excellent                |
| Base Velocity       |              2–3 | Mobile Controller        | Moderate                 |
| Driving Controls    |              2–3 | Vehicle Controller       | Moderate                 |
| Human Trajectory    |            20–50 | Retargeting              | N/A                      |
| Action Chunk        |              H×D | Same as underlying space | Same as underlying space |
| Latent Action       |           32–512 | Decoder Policy           | Excellent                |

---

# Which Action Spaces Do Popular VLAs Use?

| Model        | Primary Action Space                                                           |
| ------------ | ------------------------------------------------------------------------------ |
| RT-2         | Discretized End-Effector Delta                                                 |
| OpenVLA      | Discretized End-Effector Delta                                                 |
| π0          | Continuous End-Effector Delta Action Chunk                                     |
| π0.5        | Continuous End-Effector Delta Action Chunk                                     |
| DiffusionVLA | Continuous Action Chunk                                                        |
| DexVLA       | Arm + Dexterous Hand Action Chunk                                              |
| Qwen-VLA     | Unified Multi-Action Space (Joint, End-Effector, Navigation, Human Trajectory) |

Notes on Continuous, Discretized:

| Aspect         | Continuous Action                             | Discretized Action                                       |
| -------------- | --------------------------------------------- | -------------------------------------------------------- |
| Output         | Real numbers. Direct value of action control. | Discrete tokens/classes, each token is a defined action. |
| .Example       | `[0.021, -0.14, 0.003, 0.8]`                | `[523, 112, 901, 45]`                                  |
| Prediction     | Regression                                    | Classification / next-token prediction                   |
| Loss           | MSE, L1, Flow Matching                        | Cross-Entropy                                            |
| Typical models | π0, Diffusion Policy, ACT, OpenVLA           | RT-2, Qwen-VLA, RoboFlamingo variants                    |
| Precision      | Very high                                     | Limited by quantization                                  |
| Fits LLM       | Less naturally                                | Very naturally                                           |

---

# Trends

## Early VLA (2022–2024)

Mostly

- Joint Position
- End-Effector Delta
- Action Tokens

---

## Modern Foundation VLA (2025–2026)

Moving toward

- Continuous Action Chunks
- Multi-step prediction
- Diffusion / Flow Matching
- Multi-embodiment action spaces

---

## Future Direction

Research is increasingly exploring

- Latent action representations
- Universal action spaces
- Whole-body humanoid control
- Unified robot + human trajectory prediction
- Cross-embodiment transferable actions

The trend is moving away from robot-specific joint commands toward higher-level, transferable continuous action representations while preserving the ability to execute on many different robot embodiments.
