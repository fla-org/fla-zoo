<div align="center">

# FLA-Zoo: FLA models beyond language

</div>
<div align="center">
This repo implements a collection of FLA models that extend beyond language, supporting vision, video, and more. Under the hood are efficient kernels from <strong>fla-org</strong>. Plus, specific tools for implementing FLA models beyond language are also provided.
</div>

<div align="center">
  <br/>
  <img width="400" alt="diagram" src="assets/flazoo.png">
  <!-- <br/>
  <em>[ai generated image with modifications]</em> -->
</div>
<br/>

* [News](#news)
* [Features](#features)
* [Installation](#installation)
<!-- * [Citation](#citation) -->

## News

- **[2025-04-03]** MoBA and X-Attention are included as part of the collection for sparse attention. You can use them in specific layers of FLA models or directly use their full-blown models.

- **[2025-03-16]** Native Sparse Attention (NSA) for vision is now added. See the triton implementation under the hood [here](https://github.com/fla-org/native-sparse-attention) and its visual variant [here](https://github.com/fla-org/fla-zoo/blob/main/flazoo/models/attentions.py).

- **[2025-03-02]** A pilot version of Native Sparse Attention (NSA) is added. More experiments should be conducted to test its performance.

- **[2025-02-23]** Add LightNet for classification. Also, a pilot SFT training script for vision models is added, check it out in [here](examples/vision/sft.py).

- **[2025-02-20]** Experiments evaluating the performance of vision models are in progress.

- **[2025-01-25]** This repo is created with some vision encoders.

## Features

- **`vision:`** `fla-zoo` currently supports vision encoders. A simple documentation is in [here](docs/vision/vision.md).
- **`video:`** `fla-zoo` currently supports video understanding models. Documentation is in progress.

## Installation

Requirements:
- All the dependencies shown [here](https://github.com/fla-org/flash-linear-attention?tab=readme-ov-file#installation)
- [torchvision](https://github.com/pytorch/vision)
- [diffusers](https://github.com/huggingface/diffusers)

For example, you can install all the dependencies using the following command:
```bash
conda create -n flazoo python=3.12
conda activate flazoo
pip install torch torchvision accelerate diffusers timm
pip install transformers datasets evaluate causal_conv1d einops scikit-learn wandb
pip install flash-attn --no-build-isolation
pip install -U "huggingface_hub[cli]"
```
Now we can start cooking! 🚀

Note that as an actively developed repo, currently no released packages of `fla-zoo` are provided. Use `pip install -e .` to install the package in development mode.
