
# Qwen Research Reading Roadmap

From Overview → Foundation → Embodied AI → Specialized Robotics

This roadmap is ordered by dependency. Each stage builds on the previous one.

---

# Part I — Overview & Foundation

These papers explain the Qwen architecture before introducing robotics.

---

## 1. Qwen3 Technical Report ⭐⭐⭐⭐⭐

**Purpose**

The foundation of the modern Qwen family.

Topics

- Dense vs MoE architecture
- Thinking / Non-thinking modes
- Long context
- Scaling laws
- Training pipeline
- Benchmark results

Paper

https://arxiv.org/abs/2505.09388

---

## 2. Qwen3.5 — Towards Native Multimodal Agents ⭐⭐⭐⭐⭐

**Purpose**

Introduces Qwen's transition from an LLM into a native multimodal agent.

Topics

- Native multimodal architecture
- Agent capabilities
- Tool use
- Planning
- Long-horizon reasoning
- Vision-language foundation
- Reinforcement learning

Official Blog

https://qwen.ai/blog?id=qwen3.5

GitHub

https://github.com/QwenLM/Qwen3.6

---

## 3. Qwen3.6 ⭐⭐⭐⭐☆

**Purpose**

Evolution of Qwen3.5 with stronger coding, reasoning and agent abilities.

Focus on

- Agentic Coding
- Repository reasoning
- Thinking preservation
- Efficient MoE
- Hybrid Attention
- Gated DeltaNet

Official Blog

https://qwen.ai/blog?id=qwen3.6-35b-a3b

GitHub

https://github.com/QwenLM/Qwen3.6

---

# Part II — Embodied AI Overview

These papers bridge multimodal foundation models and robotics.

---

## 4. Qwen-VLA ⭐⭐⭐⭐⭐

**Purpose**

General Vision-Language-Action foundation model.

This is the bridge between Qwen-VL and robotics.

Topics

- Vision-Language-Action (VLA)
- Continuous action generation
- Trajectory prediction
- Cross-embodiment learning
- Multi-task learning
- Spatial reasoning

Paper

https://arxiv.org/abs/2605.18409
(if the latest arXiv identifier changes, use the official blog below)

Official Blog

https://qwen.ai/blog?id=qwenvla

GitHub

https://github.com/QwenLM/Qwen-VLA

---

# Part III — Specialized Robot Foundation Models

After understanding Qwen-VLA, these papers focus on three different robotics domains.

---

## 5. Qwen-RobotManip ⭐⭐⭐⭐⭐

**Purpose**

General robotic manipulation foundation model.

Topics

- Manipulation pretraining
- Cross embodiment learning
- Motion alignment
- Representation alignment
- Behavior alignment
- Human-to-robot data conversion
- Large-scale imitation learning

Highlights

- ~38,100 hours training data
- 15 robot embodiments
- Open-source datasets only

Paper

https://arxiv.org/abs/2606.17846

Project

https://qwen.ai

---

## 6. Qwen-RobotNav ⭐⭐⭐⭐⭐

**Purpose**

Navigation foundation model.

Topics

- Indoor navigation
- Autonomous driving
- Object search
- Waypoint following
- Agentic navigation
- Observation strategy
- Planning

Highlights

- 15.6 million training samples

Paper

https://arxiv.org/abs/2606.18112

Project Page

https://yhqpkueecs.github.io/

---

## 7. Qwen-RobotWorld ⭐⭐⭐⭐⭐

**Purpose**

Language-conditioned robot world model.

Instead of predicting actions, it predicts future observations.

Topics

- Video diffusion
- World models
- Future prediction
- Language-conditioned simulation
- Synthetic data generation
- Planning

Highlights

- 8.6M videos
- 200M+ frames
- 20+ embodiments
- 500+ action categories

Paper

https://arxiv.org/abs/2606.17030

---

# Part IV — Niche / Advanced Papers

These papers dive deeper into specific components used by the robot models.

---

## Native Active Perception as Reasoning

Topics

- Active camera control
- Observation planning
- Perception as reasoning

Paper

https://huggingface.co/Qwen/papers

---

## Unified Multimodal Autoregressive Modeling with Shared Context-Visual Tokenizer

Topics

- Unified tokenizer
- Shared visual tokens
- Autoregressive multimodal modeling

Paper

https://huggingface.co/Qwen/papers

---

# Suggested Reading Order

1. Qwen3 Technical Report
2. Qwen3.5 — Towards Native Multimodal Agents
3. Qwen3.6
4. Qwen-VLA
5. Qwen-RobotManip
6. Qwen-RobotNav
7. Qwen-RobotWorld
8. Native Active Perception as Reasoning
9. Shared Context-Visual Tokenizer

---

# Recommended Goal for Robotics Researchers

Overview
├── Qwen3
├── Qwen3.5
└── Qwen3.6

↓

Embodied AI
└── Qwen-VLA

↓

Robot Foundation Models
├── RobotManip
├── RobotNav
└── RobotWorld

↓

Advanced Research
├── Active Perception
└── Unified Visual Tokenization
