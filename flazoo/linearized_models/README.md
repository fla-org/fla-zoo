# Linearized Models 🚀

*Last updated 4.22*

## Overview

This directory contains a set of linearized version of some transformers that everybody likes.

## Purpose

The `models` directory in this repo is to directly implement some linear models from scratch. They can serve as useful tools when exploring training linear model from scratch or compare the pro and cons of different linear models.

Compared to the `models` directory, the models in this directory are initially trained using full attention with a time complexity of O(n^2). They are awesome but inefficient (often impractical) when processing long sequences.

We ideally want linear time complexity models, but the weights of these full attention models are too good to be thrown away. So this directory is created to store the linearized versions of these models. We retain most of the weights and only replace the attention layers with linear attention with potentially some new parameters.

## Note

It should be noted that after converting them into linear models, we should probably finetune them using some datasets, ideally with long sequence data.