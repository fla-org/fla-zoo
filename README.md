<div align="center">

# FLA-Zoo: Flash-Linear-Attention models beyond language

</div>
<div align="center">
This repository implements a collection of FLA models that extend beyond language, supporting vision, video, and more. Meanwhile, popular sparse attention models like NSA will also be included.
</div>

<div align="center">
  <br/>
  <img width="400" alt="diagram" src="assets/flazoo.png">
  <!-- <br/>
  <em>[ai generated image with modifications]</em> -->
</div>
<br/>

* [Features](#features)
* [Installation](#installation)
* [News](#news)
* [TODO](#todo)
<!-- * [Citation](#citation) -->

## Features

- **`vision`:** `fla-zoo` currently supports vision models. A simple documentation is in [here](docs/vision/vision.md). TL;DR: use hybrid model for better performance and efficiency. "Vision" here refers to image classification tasks.
- **`video`:** `fla-zoo` currently supports certain video models. Documentation is in progress.

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

Note that as an actively developed repo, currently no released packages of `fla-zoo` are provided.

## News

- **$\texttt{[2025-03-02]}$:** A pilot version of Native Sparse Attention (NSA) is added. More experiments will be conducted to test its performance.

- **$\texttt{[2025-02-23]}$:** Add LightNet for classification. Also, SFT training script for vision models is added, check it out in [here](examples/vision/sft.py).

- **$\texttt{[2025-02-20]}$:** I'm currently conducting many experiments to comprehsnively evaluate the performance of FLA models on vision tasks and some critical design aspects. Stay tuned!

- **$\texttt{[2025-01-25]}$:** This repo is created with some vision models (I mean classification model).

## TODO

- [x] Write documentation for vision models.
- [ ] Write documentation for video models.
- [ ] Release training scripts for vision models.
- [ ] Add diffusion models to support image/video generation.

