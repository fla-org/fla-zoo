import os
import torch
import logging
import random
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from PIL import Image

from transformers import (
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    set_seed,
    AutoImageProcessor,
    DefaultDataCollator,
)
from transformers.utils.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from datasets import Dataset, load_dataset
from torchvision import transforms

from flazoo import ABCVisionConfig, ABCForImageClassification
from flazoo import BitNetVisionConfig, BitNetForImageClassification
from flazoo import DeltaNetVisionConfig, DeltaNetForImageClassification
from flazoo import GatedDeltaNetVisionConfig, GatedDeltaNetForImageClassification
from flazoo import GLAVisionConfig, GLAForImageClassification
from flazoo import GSAVisionConfig, GSAForImageClassification
from flazoo import HGRNVisionConfig, HGRNForImageClassification
from flazoo import HGRN2VisionConfig, HGRN2ForImageClassification
from flazoo import LightNetVisionConfig, LightNetForImageClassification
from flazoo import LinearAttentionVisionConfig, LinearAttentionForImageClassification
from flazoo import RetNetVisionConfig, RetNetForImageClassification
from flazoo import RWKV6VisionConfig, RWKV6ForImageClassification
from flazoo import TransformerVisionConfig, TransformerForImageClassification
from flazoo import NSAVisionConfig, NSAForImageClassification

import evaluate
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


def get_wandb_run_name(model_args, data_args, training_args) -> str:
    """
    Generate wandb run name from arguments
    Format: {model_type}_{dataset}_{train_bs}_{eval_bs}_{epochs}
    """
    dataset = data_args.dataset_name.split('/')[-1]
    return f"{model_args.model_type}_{dataset.split('/')[-1]}_{model_args.scan_type}_tr{training_args.per_device_train_batch_size}_ev{training_args.per_device_eval_batch_size}_e{int(training_args.num_train_epochs)}_lr{training_args.learning_rate}{"_nchybrid_" + model_args.attn_layers.replace(",", "") if model_args.use_attn else ''}"

@dataclass
class ModelArguments:
    """
    Arguments for constructing the FLA-vision models
    """
    model_type: str = "deltanet"
    num_hidden_layers: int = 6
    hidden_size: int = 256
    num_heads: int = 16
    mlp_dim: Optional[int] = None # default to 4 * hidden_size
    attn_mode: str = "chunk"
    head_dim: int = 64 # For gated deltanet
    fuse_cross_entropy: bool = False
    scan_type: str = "uni-scan"
    use_attn: bool = False
    attn_layers: str = "0,1"
    hidden_dropout_prob: float = 0.5
    attn_num_heads: int = 16
    attn_num_kv_heads: Optional[int] = None
    attn_window_size: Optional[int] = None
    dtype: str = "float32"  # Model precision type: float32, float16, or bfloat16

@dataclass
class DataArguments:
    """
    Arguments for dataset preparation
    """
    dataset_name: str = "uoft-cs/cifar100" 
    image_size: int = 224
    patch_size: int = 16

@dataclass
class FLATrainingArguments(TrainingArguments):
    """
    Arguments for training the model with configurable wandb settings
    """
    output_dir: str = "output"
    do_train: bool = True
    do_eval: bool = True
    per_device_train_batch_size: int = 64
    per_device_eval_batch_size: int = 64
    num_train_epochs: int = 10
    logging_steps: int = 100
    save_steps: int = 500
    eval_steps: int = 1
    save_total_limit: int = 1
    seed: int = 42
    lr_scheduler_type: str = "constant"
    warmup_ratio: float = 0.2
    
    report_to: str = "none" 
    logging_dir: str = "logs" 
    logging_strategy: str = "steps" 
    logging_steps: int = 10 
    save_strategy: str = "epoch"

    dataloader_num_workers: int = 32
    dataloader_pin_memory: bool = True 
    persistent_workers: bool = True

    adam_beta1: float = 0.9  
    adam_beta2: float = 0.999
    weight_decay: float = 0.05
    label_smoothing_factor: float = 0.1
    # max_grad_norm: float = 1.0
    
    def __post_init__(self):
        super().__post_init__()
        if "wandb" in self.report_to:
            os.environ["WANDB_PROJECT"] = "fla-vision"
            
        os.makedirs(self.logging_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

def setup_logging(training_args):
    if not os.path.exists('logs'):
        os.makedirs('logs')
    log_filename = f'logs/training_{training_args.output_dir.split('/')[-1]}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging to {log_filename}")

def get_datasets(data_args, model_args):
    """
    Load and process datasets using standard HuggingFace image processing pipeline
    """
    dataset2class = {
        'cifar10': 10,
        'cifar100': 100,
        'slegroux/tiny-imagenet-200-clean': 200,
        'ILSVRC/imagenet-1k': 1000 
    }
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16
    }
    dtype = dtype_map[model_args.dtype]
    
    dataset = load_dataset(data_args.dataset_name)
    image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

    _transforms = transforms.Compose([
        transforms.Resize((data_args.image_size, data_args.image_size)),
        transforms.RandomHorizontalFlip(p=0.5),  
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=image_processor.image_mean,
            std=image_processor.image_std
        ),
        transforms.Lambda(lambda x: x.to(dtype))
    ])

    def transform_images_cifar100(examples):
        """Apply transforms to images"""
        examples["pixel_values"] = [
            _transforms(img)
            for img in examples["img"]
        ]
        examples['labels'] = examples['fine_label']
        del examples["img"]
        del examples["fine_label"]
        del examples["coarse_label"]
        return examples
    
    def transform_images_cifar10(examples):
        """Apply transforms to images"""
        examples["pixel_values"] = [
            _transforms(img)
            for img in examples["img"]
        ]
        examples['labels'] = examples['label']
        del examples["img"]
        del examples["label"]
        return examples

    def transform_tinyimagenet(examples):
        """Apply transforms to images"""
        examples["pixel_values"] = [
            _transforms(img)
            for img in examples["image"]
        ]
        examples['labels'] = examples['label']
        del examples["image"]
        del examples["label"]
        return examples

    def transform_imagenet(examples):
        """Apply transforms to ImageNet images with grayscale handling"""
        examples["pixel_values"] = []
        for img in examples["image"]:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            examples["pixel_values"].append(_transforms(img))
            
        examples['labels'] = examples['label']
        del examples["image"]
        del examples["label"]
        return examples

    if data_args.dataset_name == 'ILSVRC/imagenet-1k':
        transformed_dataset = dataset.with_transform(transform_imagenet)
    elif 'cifar100' in data_args.dataset_name:
        transformed_dataset = dataset.with_transform(transform_images_cifar100)
    elif 'cifar10' in data_args.dataset_name:
        transformed_dataset = dataset.with_transform(transform_images_cifar10)
    else:
        transformed_dataset = dataset.with_transform(transform_tinyimagenet)
    
    eval_split = 'validation' if 'validation' in transformed_dataset else 'test'

    return (
        transformed_dataset['train'],
        transformed_dataset[eval_split],
        dataset2class[data_args.dataset_name]
    )

def get_model(model_args, data_args, num_labels):
    """
    Initialize model with proper configuration
    """
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16
    }
    if model_args.dtype not in dtype_map:
        raise ValueError(f"Unsupported dtype: {model_args.dtype}")
    dtype = dtype_map[model_args.dtype]
    
    attn_config = None
    if model_args.use_attn:
        attn_config = {
            'layers': [int(i) for i in model_args.attn_layers.split(',')],
            'num_heads': model_args.attn_num_heads,
            'num_kv_heads': model_args.attn_num_kv_heads,
            'window_size': model_args.attn_window_size
        }

    model_classes = {
        'deltanet': (DeltaNetVisionConfig, DeltaNetForImageClassification),
        'abc': (ABCVisionConfig, ABCForImageClassification),
        'gated_deltanet': (GatedDeltaNetVisionConfig, GatedDeltaNetForImageClassification),
        'bitnet': (BitNetVisionConfig, BitNetForImageClassification),
        'gla': (GLAVisionConfig, GLAForImageClassification),
        'gsa': (GSAVisionConfig, GSAForImageClassification),
        'hgrn': (HGRNVisionConfig, HGRNForImageClassification),
        'hgrn2': (HGRN2VisionConfig, HGRN2ForImageClassification),
        "lightnet": (LightNetVisionConfig, LightNetForImageClassification),
        'linear_attn': (LinearAttentionVisionConfig, LinearAttentionForImageClassification),
        'retnet': (RetNetVisionConfig, RetNetForImageClassification),
        'rwkv6': (RWKV6VisionConfig, RWKV6ForImageClassification),
        'transformer': (TransformerVisionConfig, TransformerForImageClassification),
        'nsa': (NSAVisionConfig, NSAForImageClassification)
    }

    if model_args.model_type not in model_classes:
        raise ValueError(f"Unsupported model type: {model_args.model_type}")

    ConfigClass, ModelClass = model_classes[model_args.model_type]
    config = ConfigClass(
        num_hidden_layers=model_args.num_hidden_layers,
        hidden_size=model_args.hidden_size,
        num_heads=model_args.num_heads,
        head_dim=model_args.head_dim if "gated_deltanet" in model_args.model_type else None,
        patch_size=data_args.patch_size,
        image_size=data_args.image_size, 
        num_classes=num_labels,
        attn_mode=model_args.attn_mode,
        fuse_cross_entropy=model_args.fuse_cross_entropy,
        attn=attn_config,
        scan_type=model_args.scan_type,
        hidden_dropout_prob=model_args.hidden_dropout_prob,
    )
    
    model = ModelClass(config)
    return model.to(dtype=dtype)

def print_model_info(model, model_args):
    """
    Print model information in a formatted way
    """
    logging.info("\n" + "="*80)
    logging.info("Model Configuration:")
    logging.info("-"*40)
    
    # Print model parameters statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logging.info(f"{'Model Type:':<25} {model_args.model_type}")
    logging.info(f"{'Total Parameters:':<25} {total_params:,}")
    logging.info(f"{'Trainable Parameters:':<25} {trainable_params:,}")
    logging.info(f"{'Parameter Efficiency:':<25} {trainable_params/total_params*100:.2f}%")
    
    if model_args.use_attn:
        logging.info("\nAttention Configuration:")
        logging.info("-"*40)
        logging.info(f"{'Attention Layers:':<25} {model_args.attn_layers}")
        logging.info(f"{'Number of Heads:':<25} {model_args.attn_num_heads}")
        if model_args.attn_num_kv_heads:
            logging.info(f"{'Number of KV Heads:':<25} {model_args.attn_num_kv_heads}")
        if model_args.attn_window_size:
            logging.info(f"{'Window Size:':<25} {model_args.attn_window_size}")
    
    logging.info("="*40 + "\n")

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, FLATrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if "wandb" in training_args.report_to:
        training_args.run_name = get_wandb_run_name(model_args, data_args, training_args)
    setup_logging(training_args)
    
    set_seed(training_args.seed)

    train_dataset, eval_dataset, num_labels = get_datasets(data_args, model_args)
    
    model = get_model(model_args, data_args, num_labels)
    
    print_model_info(model, model_args)

    data_collator = DefaultDataCollator()
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator, 
        compute_metrics=compute_metrics,
    )

    if training_args.do_train:
        train_result = trainer.train()
        trainer.save_model()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

if __name__ == "__main__":
    main()