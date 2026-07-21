# Vision Transformer (ViT) in Qwen VLMs

> **Question:** What does the vision transformer contribute to a VLM, and how
> did Qwen change it from Qwen2-VL through Qwen3-VL/Qwen3.5?
>
> **Scope:** the visual path from image/video patches to contextual visual
> features. Position encoding and token compression are expanded separately in [pos_encode.md](pos_encode.md) and [patch_merger.md](patch_merger.md).
> Research date: 2026-07-21.

## Short answer

A ViT turns an image or video into a **sequence of contextual feature vectors**.
It does not itself produce words. In Qwen, the next module groups and projects
those vectors into the language model's hidden width, after which they replace
visual placeholder tokens in the multimodal sequence.

The original ViT recipe is `patchify -> linear projection -> position -> Transformer encoder -> classifier`. A Qwen VLM keeps the patch sequence instead
of reducing it to one classification token, supports variable image sizes and
videos, and trains the visual features for language-grounded perception. The
original ViT applies global multi-head self-attention to the complete patch
sequence, including its classification token, in every encoder block. The
Transformer blocks are also not identical across Qwen generations: Qwen2.5-VL,
for example, uses window attention in most visual layers and full attention in
four layers. [Original ViT, §3][vit] [Qwen2.5-VL, §2.1][qwen25]

```mermaid
flowchart LR
    A[Image or sampled video] --> B[Dynamic-resolution preprocessing]
    B --> C[Conv3D patch or tubelet embedding]
    C --> D[Visual position information]
    D --> E[ViT blocks]
    E --> F[Dense contextual visual features]
    F --> G[2 x 2 patch merger + projection]
    G --> H[LLM-width visual tokens]
```

## Core mechanism

Let the processed input have temporal length `T`, height `H`, width `W`, spatial
patch size `P`, and temporal patch size `P_t`. Before merging, the number of
visual features is

$$
N_{\text{patch}} = \frac{T}{P_t}\frac{H}{P}\frac{W}{P}.
$$

The divisions are exact because Qwen's processor resizes or pads the grid to
compatible multiples. Qwen uses a 3D convolution whose kernel and stride equal
`(P_t, P, P)`. It is therefore both a non-overlapping patch extractor and a
learned linear projection:

$$
X_0 = \operatorname{flatten}\left(
\operatorname{Conv3D}_{(P_t,P,P)}(I)
\right) \in \mathbb{R}^{N_{\text{patch}}\times d_v}.
$$

For a static image, Qwen2/2.5 duplicates the image into two identical frames and uses `P_t=2`, so the temporal output grid still has length one. For video, two
consecutive frames form one tubelet. [Qwen2-VL, §2.1][qwen2]
[Qwen2.5-VL, §2.1.1][qwen25]

Each pre-normalized ViT block then performs

$$
\begin{aligned}
Y_l &= X_l + \operatorname{Attention}(\operatorname{Norm}(X_l)),\\
X_{l+1} &= Y_l + \operatorname{MLP}(\operatorname{Norm}(Y_l)).
\end{aligned}
$$

Attention makes each patch representation depend on other patches allowed by
the layer's attention boundary. Full attention can connect the complete visual
grid; window attention restricts a layer to local windows. The result remains a
dense sequence—there is no Qwen-VL equivalent of using only the original ViT
`[CLS]` vector for classification.

## Attention: what is different from an LLM?

A ViT normally reuses the same multi-head scaled dot-product attention as a text
Transformer: patches are projected to queries, keys, and values, and each output
is a weighted mixture of values. The Q/K/V derivation, head concatenation,
residual path, and per-token MLP therefore are not repeated here. They follow the
same principle as text attention. [Attention Is All You Need, §3.2][transformer]

What changes is the **input topology and attention boundary**:

| Concern | Decoder-only LLM attention | ViT attention |
| --- | --- | --- |
| Input unit | A text token in a 1D sequence | An image patch or video tubelet on a 2D/3D grid |
| Position | Token order | Spatial or spatiotemporal coordinates |
| Visibility | Usually causal: token \(i\) cannot read future tokens | Usually bidirectional: a patch can read any permitted patch |
| Natural locality | Nearby sequence positions | Nearby regions in the image grid or nearby video frames |
| Output role in a VLM | Builds the generated text representation | Contextualizes vision features before merging and insertion into the LLM |

The ViT tower attends among visual patches; it does not yet mix those patches
with prompt words. In Qwen, vision-language interaction occurs after the merger
places LLM-width visual vectors into the multimodal sequence.

Position is especially important because flattening a grid into a sequence does
not by itself preserve “above,” “below,” or frame time. Qwen injects visual
coordinates into attention through visual RoPE; the exact coordinate flow is in
[pos_encode.md](pos_encode.md).

## Vision-specific attention variance

### Global visual attention

The original ViT uses full bidirectional attention over the complete patch
sequence, including its classification token. Unlike causal LLM attention, the
mask is not triangular: every patch can directly exchange information with
every other patch in one layer. [Original ViT, §3.1][vit]

This becomes expensive because image resolution controls sequence length. With
Qwen2.5's 14-pixel patches:

| Processed image | Patch grid | Patches | Score pairs per head |
| --- | ---: | ---: | ---: |
| \(224\times224\) | \(16\times16\) | 256 | 65,536 |
| \(448\times448\) | \(32\times32\) | 1,024 | 1,048,576 |

Doubling both image dimensions creates four times as many patches and sixteen
times as many query-key pairs. This spatial scaling pressure is why window
attention is particularly useful in vision models.

### Window attention

Window attention preserves ordinary Q/K/V attention but changes **which patches
may interact**. The 2D patch grid is partitioned into local windows, and
attention is computed independently inside each window:

```text
Global attention              Window attention

A A A A                       A A | B B
A A A A                       A A | B B
A A A A                       ----+----
A A A A                       C C | D D
                              C C | D D

one 16-patch region           four independent 4-patch regions
```

For \(N\) patches divided into fixed windows of \(K\) patches, the pair count is
\(N K\) instead of \(N^2\). The trade-off is that distant patches cannot
communicate directly in that layer.

Swin Transformer solves the fixed-boundary problem by alternating regular and
shifted window partitions. Two patches separated by one layer's boundary can
fall into the same window in the next layer, allowing information to propagate
across the image through depth. Swin also merges patches between stages to build
a spatial hierarchy. [Swin Transformer, §§1 and 3.2][swin]

```text
Layer L: regular windows  ->  Layer L+1: shifted windows
local mixing only             previous window boundaries are crossed
```

Window attention is best understood as a vision-shaped sparse pattern: its
regular rectangular blocks match image geometry and are efficient to batch.

### Hybrid window and global attention

Another solution keeps cheap window attention in most layers and inserts a few
global layers to reconnect the complete image:

```text
window -> window -> window -> global -> repeat
```

Qwen2.5-VL uses this design. Of its 32 visual blocks, blocks
`{7,15,23,31}` use full attention and the other 28 use 112 x 112-pixel windows.
For a `224 x 224` image, each local layer processes four windows of 64 patches:

```text
window layer: 4 * (64 * 64) = 16,384 score pairs/head
global layer:     256 * 256  = 65,536 score pairs/head
```

The window layers reduce cost, while the periodic global layers provide direct
whole-image exchange. This is not Swin's shifted-window mechanism: Qwen2.5 uses
ordinary windows plus explicit global-attention layers.
[Qwen2.5-VL, Table 1 and §2.1.1][qwen25]
[Pinned Qwen2.5-VL implementation][qwen25-code]

### Other variants, briefly

Generic sparse attention, linear attention, and optimized kernels such as
FlashAttention are not intrinsically ViT mechanisms; the same ideas also apply
to text Transformers. They are therefore out of scope here. The relevant
vision-specific choice is how attention connectivity follows the image/video
grid: global, fixed window, shifted window, or a window/global schedule.

In the documented Qwen towers, Qwen2.5-VL uses the hybrid window/global scheme.
The pinned Qwen3-VL implementation instead uses full attention within each
packed image or video segment, and Qwen3.5 inherits that vision-block family.
[Pinned Qwen3-VL implementation][qwen3-code]
[Pinned Qwen3.5 vision path][qwen35-code]

## Example dataflow: one 224 x 224 image

Use **Qwen2.5-VL-7B** as a concrete example. The paper/config specify a
14-pixel patch, temporal patch size 2, visual width 1,280, 32 ViT blocks, and an
output width of 3,584. The following shows tensor shapes; numeric feature values
are learned and therefore omitted. [Qwen2.5-VL, Table 1 and §2.1][qwen25]
[Pinned Qwen2.5-VL-7B config][qwen25-config]

```text
Input RGB image
  logical shape: 224 x 224 x 3
        |
        | treat image as two identical frames
        v
Logical video-shaped input
  2 frames x 224 x 224 x 3
        |
        | Conv3D kernel=stride=(2, 14, 14)
        v
Patch grid
  T=1, H=16, W=16, width=1280
  flattened shape: 256 x 1280
        |
        | visual 2D RoPE on the 16 x 16 grid
        v
32 ViT blocks
  blocks 7, 15, 23, 31: full attention over 256 patches
  other 28 blocks: window attention
        |
        v
Contextual ViT features
  shape: 256 x 1280
        |
        | 2 x 2 merger; four neighboring vectors become one
        v
Merged visual payload
  grid: 1 x 8 x 8
  shape: 64 x 3584
        |
        | replace 64 visual placeholders in the prompt
        v
Qwen2.5 language decoder
```

For this image, a 112 x 112-pixel attention window corresponds to `8 x 8`
pre-merge patches. On the `16 x 16` patch grid, a window-attention layer therefore
processes four spatial windows of 64 patches each, while a full-attention layer
can connect all 256 patches. This is why the final feature at one location is no
longer just a flattened local patch: it contains context accumulated across the
32-block attention graph.

The final `64 x 3584` sequence is the merger output, not the raw ViT output. The
next two reports zoom into the two omitted transformations: the exact
[position-coordinate flow](pos_encode.md) and one worked
[2 x 2 merge operation](patch_merger.md).

## Classification ViT versus Qwen's VLM ViT

| Aspect          | Original classification ViT                      | Qwen vision tower                                                                                      |
| --------------- | ------------------------------------------------ | ------------------------------------------------------------------------------------------------------ |
| Input shape     | Usually one fixed resolution                     | Dynamic/native-resolution image and variable video grids                                               |
| Patch embedding | 2D linear projection or equivalent Conv2D        | Shared Conv3D patch/tubelet projection for image and video                                             |
| Position        | Learned 1D absolute table in the original design | 2D visual RoPE; later Qwen also interpolates learned absolute embeddings                               |
| Attention       | Global attention in every encoder block          | Generation-dependent: Qwen2.5 mixes window and global attention; Qwen3-VL uses packed visual attention |
| Output          | Commonly one class representation                | All spatial features, then a 2 x 2 merger/projector                                                    |
| Training target | Image classification                             | Vision-language alignment and end-to-end multimodal objectives                                         |

The ViT paper itself emphasizes that useful performance required large-scale
pretraining. Architecture alone does not explain OCR, grounding, or visual
reasoning quality. [Original ViT][vit]

## Evolution in Qwen

| Model      | Verified visual-backbone change                                                                                                                                                      | Practical effect                                                                                                               |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------ |
| Qwen2-VL   | Approximately 675M-parameter ViT; patch size 14; removes the old absolute table, adds 2D RoPE, dynamic resolution, and depth-2 3D convolution                                        | One tower handles images and videos while preserving variable spatial detail                                                   |
| Qwen2.5-VL | New 32-layer, width-1280 ViT trained from scratch; 16 heads; 14-pixel patches; 28 window-attention layers and full attention at blocks`{7,15,23,31}`; RMSNorm and SwiGLU-style FFN | Most visual mixing scales approximately linearly with patch count while periodic global layers restore whole-image interaction |
| Qwen3-VL   | Continues from pretrained SigLIP-2, trains at dynamic resolutions, combines interpolated learned positions with 2D RoPE, and exports three intermediate levels for DeepStack         | Stronger pretrained visual initialization and multi-level fusion with the LLM                                                  |
| Qwen3.5    | Inherits Qwen3-VL visual building blocks but the released implementation deletes DeepStack and passes only the final merged features at decoder input                                | Simpler fusion path; do not assume every Qwen3-VL feature survives unchanged                                                   |

### Qwen2-VL

**Verified.** Qwen2-VL uses a single roughly 675M-parameter ViT for all LLM
sizes. A 14-pixel spatial patch and two-frame temporal kernel create image/video
features; dynamic resolution makes the sequence length depend on the input
grid. The model removes the prior absolute position embeddings and uses 2D RoPE
inside the tower. [Qwen2-VL, §2.1][qwen2]

This design improves flexibility, but higher resolution still produces more
patches. Qwen2's own ablation also shows that merely enlarging every image is not
always better; inappropriate upscaling can move inputs away from the training
distribution. [Qwen2-VL, §3.3.1][qwen2]

### Qwen2.5-VL

**Verified.** The Qwen2.5-VL paper reports a 32-layer ViT with hidden width
1,280, 16 heads, and patch size 14. Its window/global attention schedule is
detailed above. Smaller regions are processed without padding.
[Qwen2.5-VL, Table 1 and §2.1.1][qwen25]

The paper reports an FFN intermediate size of 3,456, while the currently
released 7B checkpoint config records 3,420. This is a source discrepancy, not a
value to silently normalize away. Use the checkpoint config when reproducing
that checkpoint and the paper value when describing the paper experiment.
[Pinned Qwen2.5-VL-7B config][qwen25-config]

### Qwen3-VL and Qwen3.5

**Verified.** Qwen3-VL initializes its encoder from SigLIP-2 and continuously
trains it at dynamic resolutions. The paper calls the result Qwen3-ViT and uses
SigLIP2-SO-400M by default, with SigLIP2-Large for the 2B and 4B variants. Its
ablation compares this trained encoder with the original SigLIP-2 under the
reported setup; the result should not be read as a universal ranking of vision
encoders. [Qwen3-VL, §2 and §5.12.1][qwen3]

The pinned Qwen3-VL-8B config has 27 visual blocks, width 1,152, 16 heads,
`patch_size=16`, `temporal_patch_size=2`, and DeepStack taps at blocks 8, 16, and
24. [Pinned Qwen3-VL-8B config][qwen3-config] Qwen3.5's 27B config keeps the same
patch/tower shape but has an empty DeepStack index list; its reference forward
path explicitly removes the DeepStack modules. These exact numbers are
checkpoint-specific, not a promise for every family member.
[Pinned Qwen3.5-27B config][qwen35-config]
[Pinned Qwen3.5 vision path][qwen35-code]

## Cost, information, and common mistakes

- Visual self-attention is quadratic in the number of patches when it is global.
  Qwen2.5's window layers reduce this part to roughly linear scaling for a fixed window size, but its four full-attention layers remain quadratic.
- Patch features are not raw local pixels after the ViT. Each feature has already accumulated context through the allowed attention graph before merging.
- Dynamic resolution preserves more source detail only within the processor's
  pixel/token budget. It does not imply lossless native pixels.
- A larger image creates more visual prefill work and, after merging, more tokens for the LLM. The [patch merger](patch_merger.md) reduces but does not eliminate
  that growth.
- Qwen2.5 window attention and Qwen3-VL DeepStack belong to different releases.
  Listing both as generic properties of every “recent Qwen” model is incorrect.
- Qwen-VLA confirms a Qwen3.5 VLM backbone with ViT and spatial merging, but its
  paper does not disclose the exact 4B vision configuration. Do not substitute
  the 27B config without marking it as an inference.

## Sources

All online sources were accessed on 2026-07-21.

- Dosovitskiy et al. *An Image Is Worth 16x16 Words: Transformers for Image
  Recognition at Scale*. ICLR 2021. [arXiv][vit]
- Vaswani et al. *Attention Is All You Need*. NeurIPS 2017.
  [arXiv][transformer]
- Liu et al. *Swin Transformer: Hierarchical Vision Transformer using Shifted
  Windows*. ICCV 2021. [arXiv][swin]
- Wang et al. *Qwen2-VL: Enhancing Vision-Language Model's Perception of the
  World at Any Resolution*. 2024. [Local PDF][qwen2-local] · [arXiv][qwen2]
- Bai et al. *Qwen2.5-VL Technical Report*. 2025.
  [Local PDF][qwen25-local] · [arXiv][qwen25]
- Bai et al. *Qwen3-VL Technical Report*. 2025. [arXiv][qwen3]
- Qwen and Hugging Face. Pinned checkpoint configs and reference implementations
  linked below.

[vit]: https://arxiv.org/abs/2010.11929
[transformer]: https://arxiv.org/abs/1706.03762
[swin]: https://arxiv.org/abs/2103.14030
[qwen2]: https://arxiv.org/abs/2409.12191
[qwen25]: https://arxiv.org/abs/2502.13923
[qwen3]: https://arxiv.org/abs/2511.21631
[qwen2-local]: ../../../gwen-overview/qwen2_vl_2409.12191.pdf
[qwen25-local]: ../../../gwen-overview/qwen2.5_vl_2502.13923.pdf
[qwen25-config]: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/blob/cc594898137f460bfe9f0759e9844b3ce807cfb5/config.json
[qwen25-code]: https://github.com/huggingface/transformers/blob/29985e67cccdddef7e336d7e53840500359d30a3/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py
[qwen3-config]: https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/blob/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b/config.json
[qwen3-code]: https://github.com/huggingface/transformers/blob/29985e67cccdddef7e336d7e53840500359d30a3/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py
[qwen35-config]: https://huggingface.co/Qwen/Qwen3.5-27B/blob/fc05daec18b0a78c049392ed2e771dde82bdf654/config.json
[qwen35-code]: https://github.com/huggingface/transformers/blob/7ea2320c76117e6742364808a666ef6f2fb40a67/src/transformers/models/qwen3_5/modular_qwen3_5.py
