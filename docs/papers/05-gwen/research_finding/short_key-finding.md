**Modern module differents in original LLM.**

- RoPE (Rotary Position Embedding) - Position Embedding
- Attentions - Decrease memory:
  - Multi Query Attention (MQA)
  - Grouped Query Attention (GQA)
  - FlashAttention.
- Feed Forward Layer: ReLU -> SwiGLU
- LayerNorm: -> RMSNorm
- MoE (Mixture of Experts) Reduce token use in FFN, Using a router to route to smaller FFN work on a specific knowledge (call expert):

## What is architecturally distinctive in Qwen?

- **Hybrid sequence layers:** Qwen3.5/3.6 repeat three Gated DeltaNet
  (linear/recurrent attention) layers and one gated full-attention layer. This
  lowers long-context cost while periodically restoring global token mixing.
- **Hybrid attention + sparse MoE + MTP:** for example, Qwen3.6-35B-A3B has 35B
  total parameters but activates about 3B per token, and adds multi-token
  prediction for faster decoding.
- **Explicit multimodal design:** Qwen3-VL adds Interleaved-MRoPE, DeepStack
  multi-level vision features, and text timestamps for spatial-temporal
  image/video grounding.

The individual parts are **not unique to Qwen**. Its distinction is their
openly documented combination. GQA, RoPE, RMSNorm, SwiGLU, and MoE are common in
other LLMs; thinking/non-thinking control is mainly post-training and inference
behavior, not a new neural block.

Sources: [Qwen3.6 model card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B),
[Qwen3-VL technical report](https://arxiv.org/abs/2511.21631), and
[Qwen3 technical report](https://arxiv.org/abs/2505.09388). Accessed 2026-07-20.
