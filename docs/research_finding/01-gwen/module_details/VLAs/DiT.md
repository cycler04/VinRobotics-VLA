
# Diffusion Transformer (DiT)

## What is DiT?

**Diffusion Transformer (DiT)** is a **Transformer-based backbone** for generative models. It replaces the traditional U-Net used in diffusion models.

Unlike **Flow Matching**, **DDPM**, or **Rectified Flow**, DiT is **not a training algorithm**. It is simply the neural network that learns the mapping:

$$
f_\theta(x_t, t, c)
$$

where:

- \(x_t\): noisy latent
- \(t\): timestep
- \(c\): conditioning information (text, image, robot state, etc.)

The model predicts a target determined by the training objective.

---

## Relationship with Flow Matching

DiT and Flow Matching solve different problems.

| Component               | Role                                                        |
| ----------------------- | ----------------------------------------------------------- |
| **DiT**           | Neural network architecture (Transformer backbone)          |
| **Flow Matching** | Training objective that teaches the network what to predict |

The same DiT architecture can also be trained using:

- DDPM (noise prediction)
- Velocity prediction
- Score Matching
- Flow Matching
- Consistency Models

Only the prediction target changes.

---

## General Architecture

```text
Noisy latent + timestep + conditions
                │
                ▼
        Diffusion Transformer
                │
                ▼
      Predicted target (noise, flow,
      velocity, action, etc.)
```

---

## Common Applications

### 1. Image Generation

The most common use.

```text
Text
  │
Text Encoder
  │
Latent + Noise
  │
 DiT
  │
Image
```

Examples:

- FLUX
- Stable Diffusion 3
- Qwen-Image

---

### 2. Video Generation

Instead of image patches, the model processes **spatio-temporal patches** (time × height × width).

Examples:

- Sora
- Movie Gen

---

### 3. Audio Generation

DiT can generate or enhance audio from latent representations.

Applications include:

- Speech synthesis
- Music generation
- Audio editing

---

### 4. 3D Generation

The input may be:

- Point clouds
- Voxels
- Latent 3D representations

Applications include:

- 3D object generation
- CAD generation
- Neural rendering

---

### 5. Robotics

Modern robot foundation models use DiT to predict continuous robot actions.

```text
Image + Robot State + Instruction
                │
                ▼
               DiT
                │
                ▼
      Future action trajectory
```

Outputs may include:

- End-effector pose
- Joint commands
- Navigation actions
- Manipulation trajectories

#### DiT in Qwen-VLA

Qwen-VLA places a **separate 16-block DiT action decoder** after its
Qwen3.5-4B vision-language backbone. The VLM first produces hidden states from
images, instructions, and the embodiment prompt. After projection to the DiT
width, those context tokens are concatenated with projected noisy-action tokens
and processed using joint self-attention, flow-timestep AdaLN conditioning, and
multi-section RoPE.

```text
Qwen3.5 VLM hidden states + noisy action chunk + flow timestep
                              │
                              ▼
                   16-block DiT decoder
                              │
                              ▼
                    action velocity field
                              │
                    repeated Euler updates
                              ▼
                  continuous action chunk
```

One DiT pass predicts a **flow velocity**, not the final command. Starting from
Gaussian action noise, several Euler steps repeatedly call the approximately
1.15B-parameter DiT until it produces the final `H × K` action or trajectory
tensor. Thus, “DiT” identifies the decoder architecture, while conditional flow
matching defines its learning and generation process. [Qwen-VLA, §§2.2–2.5](https://arxiv.org/abs/2605.30280)
See [the detailed action-decoder report](action_generation/05_large_diffusion_transformer.md#qwen-vla-the-dit-action-decoder)
for the tensor mask, parameter breakdown, and multi-embodiment interface.

---

### 6. Motion Generation

DiT can predict future human or robot motion.

Examples:

- Human pose prediction
- Robot trajectory planning
- Animation generation

---

### 7. Scientific Modeling

Researchers also apply DiT to continuous scientific data such as:

- Molecular generation
- Protein structures
- Material design
- Physics simulations

---

## Why DiT Is Popular

Compared with U-Nets, Transformers provide:

| Feature                       | U-Net    | DiT       |
| ----------------------------- | -------- | --------- |
| Global attention              | Limited  | ✓        |
| Large-scale model scaling     | Moderate | Excellent |
| Multimodal conditioning       | Moderate | Excellent |
| Variable-length token support | Limited  | ✓        |

These advantages have made DiT the preferred backbone for many modern generative foundation models.

---

## Key Takeaways

- **DiT is a Transformer architecture, not a training algorithm.**
- It can be paired with **Flow Matching, DDPM, Score Matching, Velocity Prediction,** and other objectives.
- The same architecture is used across **image, video, audio, 3D, robotics, motion, and scientific generation**.
- Its flexibility and scalability have made it the dominant backbone for many recent generative AI models.
