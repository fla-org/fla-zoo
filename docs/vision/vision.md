# FLA for Vision 🖼️

<div align="right">
<em>Last updated: 2025-02-26</em>
</div>

> **Note:** `vision` here refers specifically to image classification. Currently, only classification models are implemented to serve as general vision encoders. See training scripts at [SFT](examples/vision/mim.py) and [MIM](examples/vision/mim.pyy) for more details.

## Overview 📋

This module implements image classification models based on FLA to:
- Simplify their application as general vision encoders
- Enable easier adoption and comparisons
- Provide a foundation for vision understanding tasks

The architecture is primarily based on Hugging Face's Vision Transformer (ViT) implementation with several FLA-specific customizations.

### Available Model Types

- `xxxForImageClassification` - For image classification tasks
- `xxxForMaskedImageModeling` - For masked image modeling pre-training
- `xxxVisionModel` - Base vision models for custom downstream tasks

## Implementation 🛠️

### 1. Code Structure

- Maintains consistency with existing language models in [FLA](https://github.com/fla-org/flash-linear-attention)
- Follows similar patterns and conventions for easier understanding

### 2. Scanning Options

| Scan Type | Operation Flow |
|-----------|---------------|
| **Uni-scan** | `[B, L, D] → FLA → [B, L, D]` |
| **Random-scan** | `[B, L, D] → random shuffle → [B, L, D] → FLA → [B, L, D]` |
| **Flip-scan** | `[B, L, D] → flip → [B, L, D] → FLA → [B, L, D]` |
| **Bi-scan** | `[B, L, D] → flip → [2*B, L, D] → FLA → [2*B, L, D] → combine → [B, L, D]` |
| **Cross-scan** | `[B, L, D] → cross-scan → [4*B, L, D] → FLA → [4*B, L, D] → cross-merge → [B, L, D]` |

> ⚠️ **Warning:** The latter two options (Bi-scan and Cross-scan) are design choices adopted by some SSM-based vision models. **Enabling them does not guarantee better performance and will reduce hardware efficiency.**

### 3. Technical Details

- Uses mean pooling exclusively for sequence aggregation
- Adapts common components from Hugging Face's ViT implementation:
  - `Embedding` - For patch and position embeddings
  - `Pooler` - For sequence pooling
  - Initialization code for pretrained models

> 🔜 **Coming Soon:** Mamba, Mamba2, and Samba models will be implemented in future versions due to their structural differences.

## Model Compatibility Tests 🧪

### Test Configuration

| Parameter | Value |
|-----------|-------|
| **Total layers** | 6 |
| **Hybrid setting** | Attention layers at indices 1,3,5 |
| **Default attention mode** | chunk (except for rwkv6) |

### Test Results

| Model | Pure FLA | Hybrid |
|-------|----------|--------|
| **abc** | ❌ CompilationError | ❌ CompilationError |
| **bitnet** | ❌ AttributeError | ❌ AttributeError |
| **deltanet** | ✅ | ✅ |
| **gated_deltanet** | RTX 4060: ❌<br>A100: ✅ | RTX 4060: ❌<br>A100: ✅ |
| **gla** | ❌ CompilationError | ❌ CompilationError |
| **gsa** | ✅ | ✅ |
| **hgrn** | ✅ | ✅ |
| **hgrn2** | ✅ | ✅ |
| **linear_attn** | ❌ Matmul Shape error | ❌ Matmul Shape error |
| **retnet** | ✅ | ✅ |
| **rwkv6** | chunk: ❌<br>fused_recurrent: ✅ | chunk: ❌<br>fused_recurrent: ✅ |
| **transformer** | ✅ | ✅ |
| **lightnet** | ✅ | ✅ |

> **Note:** Errors primarily stem from respective attention implementations from FLA.