# FLA for vision

- last updated: 2025-01-26
- `vision` here specifies only image. Also, only classification models are implemented to obtain general vision encoders.

## Overview
This part implements image classification models based on FLA to simplify their application as general vision encoders and enable easier adoption and comparisons. The architecture is primarily based on Hugging Face's Vision Transformer (ViT) implementation with several customizations. 

Models include `xxxForImageClassification` and `xxxForMaskedImageModeling` and `xxxVisionModel`.

## Implementation

1. **Code Structure**
   - Maintains consistency with existing language models in FLA, see [FLA](https://github.com/fla-org/flash-linear-attention)

2. **Scanning Options**
   - Uni-scan: `[B, L, D] -> FLA -> [B, L, D]`
   - Random-scan: `[B, L, D] -> random shuffle -> [B, L, D] -> FLA -> [B, L, D]`
   - Bi-scan: `[B, L, D] -> flip -> [2 * B, L, D] -> FLA -> [2 * B, L, D] -> combine -> [B, L, D]`
   - Cross-scan: `[B, L, D] -> cross-scan -> [4 * B, L, D] -> FLA -> [4 * B, L, D] -> cross-merge -> [B, L, D]`

    The latter two are design choices adopted by some SSM-based vision models, **enabling them does not garantee better performance and will damage hardware efficiency.**
   
3. **Technical Details**
   - Uses mean pooling only.
   - Adapted common components (`Embedding`, `Pooler`) and initialization code for pretrained models from Hugging Face's ViT implementation.

Note: Currently, Mamba, Mamba2, and Samba models will be implemented in future versions due to their structural differences.

## Model Compatibility Tests

Test Configuration:
- Total layers: 6
- Hybrid setting: Attention layers at indices 1,3,5
- Default attention mode: chunk (except for rwkv6)

Test Results:

| Model          | Pure FLA                                          | Hybrid                                            |
| -------------- | ------------------------------------------------- | ------------------------------------------------- |
| abc            | ❌ CompilationError                                | ❌ CompilationError                                |
| bitnet         | ❌ AttributeError                                  | ❌ AttributeError                                  |
| deltanet       | ✅                                                 | ✅                                                 |
| gated_deltanet | RTX 4060:❌<br>A100: ✅                           | RTX 4060:❌<br>A100: ✅                           |
| gla            | ❌ CompilationError                                | ❌ CompilationError                                |
| gsa            | ✅                                                 | ✅                                                 |
| hgrn           | ✅                                                 | ✅                                                 |
| hgrn2          | ✅                                                 | ✅                                                 |
| linear_attn    | ❌ Matmul Shape error                              | ❌ Matmul Shape error                              |
| retnet         | ✅                                                 | ✅                                                 |
| rwkv6          | chunk:❌<br>fused_recurrent:✅                     | chunk:❌<br>fused_recurrent:✅                     |
| transformer    | ✅                                                 | ✅                                                 |

**Note: Errors primarily stem from respective attention implementations from FLA.**

