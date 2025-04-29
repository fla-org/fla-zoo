import os
import torch
import logging
import argparse
import numpy as np
import random
import time
import json
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
from PIL import Image

# Add accelerate import
from accelerate import Accelerator

# timm related imports
import timm
import timm.data
from timm.data import create_transform, Mixup, AugMixDataset
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from torch import optim as optim
from timm.scheduler import create_scheduler_v2
from timm.utils import ModelEma, accuracy, AverageMeter, CheckpointSaver, update_summary
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.scheduler.scheduler import Scheduler
import torch.nn.functional as F
from torch import nn

# Import datasets for loading data
import datasets
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import math
# Import fla-zoo models
from flazoo import ABCVisionConfig, ABCForImageClassification
from flazoo import BitNetVisionConfig, BitNetForImageClassification
from flazoo import DeltaNetVisionConfig, DeltaNetForImageClassification
from flazoo import GatedDeltaNetVisionConfig, GatedDeltaNetForImageClassification
from flazoo import GLAVisionConfig, GLAForImageClassification
from flazoo import GSAVisionConfig, GSAForImageClassification
from flazoo import HGRNVisionConfig, HGRNForImageClassification
from flazoo import HGRN2VisionConfig, HGRN2ForImageClassification
from flazoo import LinearAttentionVisionConfig, LinearAttentionForImageClassification
from flazoo import RetNetVisionConfig, RetNetForImageClassification
from flazoo import RWKV6VisionConfig, RWKV6ForImageClassification
from flazoo import RWKV7VisionConfig, RWKV7ForImageClassification
from flazoo import TransformerVisionConfig, TransformerForImageClassification
from flazoo import NSAVisionConfig, NSAForImageClassification
from flazoo import LightNetVisionConfig, LightNetForImageClassification

# for init from dino
from flazoo.helpers.initializer import (
    initialize_custom_mapping
)

from flazoo.helpers.linearizer import init_from_dino2_base_p14, init_from_siglip2_base_p16_224

class CosineLRScheduler(Scheduler):
    """
    Cosine decay with restarts.
    This is described in the paper https://arxiv.org/abs/1608.03983.

    Inspiration from
    https://github.com/allenai/allennlp/blob/master/allennlp/training/learning_rate_schedulers/cosine.py
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 t_initial: int,
                 t_mul: float = 1.,
                 lr_min: float = 0.,
                 decay_rate: float = 1.,
                 warmup_t=0,
                 warmup_lr_init=0,
                 warmup_prefix=False,
                 cycle_limit=0,
                 t_in_epochs=True,
                 noise_range_t=None,
                 noise_pct=0.67,
                 noise_std=1.0,
                 noise_seed=42,
                 initialize=True) -> None:
        super().__init__(
            optimizer, param_group_field="lr",
            noise_range_t=noise_range_t, noise_pct=noise_pct, noise_std=noise_std, noise_seed=noise_seed,
            initialize=initialize)

        assert t_initial > 0
        assert lr_min >= 0
        if t_initial == 1 and t_mul == 1 and decay_rate == 1:
            print("Cosine annealing scheduler will have no effect on the learning "
                           "rate since t_initial = t_mul = eta_mul = 1.")
        self.t_initial = t_initial
        self.t_mul = t_mul
        self.lr_min = lr_min
        self.decay_rate = decay_rate
        self.cycle_limit = cycle_limit
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.warmup_prefix = warmup_prefix
        self.t_in_epochs = t_in_epochs
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

    def _get_lr(self, t):
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            if self.warmup_prefix:
                t = t - self.warmup_t

            if self.t_mul != 1:
                i = math.floor(math.log(1 - t / self.t_initial * (1 - self.t_mul), self.t_mul))
                t_i = self.t_mul ** i * self.t_initial
                t_curr = t - (1 - self.t_mul ** i) / (1 - self.t_mul) * self.t_initial
            else:
                i = t // self.t_initial
                t_i = self.t_initial
                t_curr = t - (self.t_initial * i)

            gamma = self.decay_rate ** i
            lr_min = self.lr_min * gamma
            lr_max_values = [v * gamma for v in self.base_values]

            if self.cycle_limit == 0 or (self.cycle_limit > 0 and i < self.cycle_limit):
                lrs = [
                    lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * t_curr / t_i)) for lr_max in lr_max_values
                ]
            else:
                lrs = [self.lr_min for _ in self.base_values]

        return lrs

    def get_epoch_values(self, epoch: int):
        if self.t_in_epochs:
            return self._get_lr(epoch)
        else:
            return None

    def get_update_values(self, num_updates: int):
        if not self.t_in_epochs:
            return self._get_lr(num_updates)
        else:
            return None

    def get_cycle_length(self, cycles=0):
        if not cycles:
            cycles = self.cycle_limit
        cycles = max(1, cycles)
        if self.t_mul == 1.0:
            return self.t_initial * cycles
        else:
            return int(math.floor(-self.t_initial * (self.t_mul ** cycles - 1) / (1 - self.t_mul)))

# Parameter definitions
@dataclass
class ModelArguments:
    """FLA-vision model parameters"""
    model_type: str = field(default="deltanet", metadata={"help": "Model type"})
    compress_attention: bool = field(default=False, metadata={"help": "Whether to compress attention"})
    num_hidden_layers: int = field(default=6, metadata={"help": "Number of hidden layers"})
    hidden_size: int = field(default=256, metadata={"help": "Hidden dimension size"})
    num_heads: int = field(default=16, metadata={"help": "Number of attention heads"})
    channel_mixer_dim: Optional[int] = field(default=None, metadata={"help": "channel mixer dimension, defaults to 4*hidden_size"})
    attn_mode: str = field(default="chunk", metadata={"help": "Attention mode"})
    head_dim: int = field(default=64, metadata={"help": "Head dimension, used for gated deltanet"})
    fuse_cross_entropy: bool = field(default=False, metadata={"help": "Whether to use fused cross entropy"})
    scan_type: str = field(default="uni-scan", metadata={"help": "Scan type"})
    use_attn: bool = field(default=False, metadata={"help": "Whether to use attention"})
    attn_layers: str = field(default="0,1", metadata={"help": "Layers using attention"})
    hidden_dropout_prob: float = field(default=0.2, metadata={"help": "Hidden layer dropout probability"})
    attn_num_heads: int = field(default=16, metadata={"help": "Number of attention heads"})
    attn_num_kv_heads: int = field(default=None, metadata={"help": "Number of KV attention heads"})
    attn_window_size: int = field(default=None, metadata={"help": "Attention window size"})
    attn_block_size: int = field(default=None, metadata={"help": "Attention block size"})
    attn_topk: int = field(default=None, metadata={"help": "Attention topk"})
    attn_block_counts: int = field(default=None, metadata={"help": "Attention block counts"})
    attn_stride: int = field(default=None, metadata={"help": "Attention stride"})
    attn_chunk_size: int = field(default=None, metadata={"help": "Attention chunk size"})
    attn_type: str = field(default="full_attn", metadata={"help": "Attention type used in hybrid model"})
    dtype: str = field(default="float32", metadata={"help": "Model precision type"})

@dataclass
class DataArguments:
    """Dataset parameters"""
    dataset_name: str = field(default="cifar100", metadata={"help": "Dataset name"})
    image_size: int = field(default=224, metadata={"help": "Image size"})
    patch_size: int = field(default=16, metadata={"help": "Patch size"})

@dataclass
class TrainingArguments:
    """Training parameters"""
    output_dir: str = field(default="output", metadata={"help": "Output directory"})
    num_epochs: int = field(default=10, metadata={"help": "Number of training epochs"})
    batch_size: int = field(default=64, metadata={"help": "Training batch size"})
    val_batch_size: int = field(default=64, metadata={"help": "Validation batch size"})
    learning_rate: float = field(default=1e-3, metadata={"help": "Initial learning rate"})
    weight_decay: float = field(default=0.05, metadata={"help": "Weight decay"})
    momentum: float = field(default=0.9, metadata={"help": "Momentum"})
    optimizer: str = field(default="adamw", metadata={"help": "Optimizer type"})

    model: str = field(default="deltanet", metadata={"help": "Model type"})
    
    # Learning rate scheduler parameters
    sched: str = field(default="cosine", metadata={"help": "LR scheduler type"})
    warmup_epochs: int = field(default=5, metadata={"help": "Warmup epochs"})
    cooldown_epochs: int = field(default=0, metadata={"help": "Cooldown epochs"})
    min_lr: float = field(default=5e-6, metadata={"help": "Minimum learning rate"})
    
    # Mixup parameters
    mixup: float = field(default=0.8, metadata={"help": "Mixup alpha value"})
    cutmix: float = field(default=1.0, metadata={"help": "Cutmix alpha value"})
    mixup_prob: float = field(default=1.0, metadata={"help": "Probability of applying mixup"})
    mixup_switch_prob: float = field(default=0.5, metadata={"help": "Probability of switching from mixup to cutmix"})
    mixup_mode: str = field(default="batch", metadata={"help": "How to apply mixup/cutmix ('batch', 'pair', 'elem')"})
    
    # EMA parameters
    use_ema: bool = field(default=True, metadata={"help": "Whether to use EMA"})
    ema_decay: float = field(default=0.9999, metadata={"help": "EMA decay rate"})
    ema_force_cpu: bool = field(default=False, metadata={"help": "Force EMA to be stored on CPU"})
    
    # Other training parameters
    seed: int = field(default=0, metadata={"help": "Random seed"})
    workers: int = field(default=4, metadata={"help": "Data loading workers"})
    pin_memory: bool = field(default=True, metadata={"help": "Whether to use pin_memory"})
    log_interval: int = field(default=50, metadata={"help": "Logging interval"})
    eval_metric: str = field(default="top1", metadata={"help": "Evaluation metric"})
    
    # Wandb parameters
    report_to_wandb: bool = field(default=False, metadata={"help": "Whether to use wandb logging"})
    wandb_project: str = field(default="fla-vision-timm", metadata={"help": "Wandb project name"})
    wandb_run_name: str = field(default="", metadata={"help": "Wandb run name"})
    
    # Label smoothing parameters
    label_smoothing: float = field(default=0.1, metadata={"help": "Label smoothing factor"})
    
    # Augmentation parameters
    use_augmix: bool = field(default=False, metadata={"help": "Whether to use AugMix"})
    ra_magnitude: int = field(default=9, metadata={"help": "RandAugment magnitude"})
    ra_num_ops: int = field(default=2, metadata={"help": "RandAugment num operations"})
    
    # Save and evaluation parameters
    save_freq: int = field(default=1, metadata={"help": "Save frequency (epochs)"})
    eval_freq: int = field(default=1, metadata={"help": "Evaluation frequency (epochs)"})
    save_checkpoint: bool = field(default=True, metadata={"help": "Whether to save checkpoints"})
    save_total_limit: int = field(default=3, metadata={"help": "Maximum number of checkpoints to keep"})

    # init from pretrained
    init_from_pretrained: bool = field(default=False, metadata={"help": "Whether to initialize from pretrained model"})
    # which model
    init_model: str = field(default="dino", metadata={"help": "Which model to initialize from"})

    # training mode: distill, label, hybrid
    training_mode: str = field(default="label", metadata={"help": "Training mode: distill, label, hybrid"})

def setup_logging(training_args):
    """Set up logging"""
    if not os.path.exists('logs'):
        os.makedirs('logs')
    log_filename = f'logs/training_timm_{os.path.basename(training_args.output_dir)}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging to {log_filename}")

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_wandb_run_name(model_args, data_args, training_args):
    """Generate wandb run name"""
    dataset = data_args.dataset_name if '/' not in data_args.dataset_name else data_args.dataset_name.split('/')[-1]
    return f"{model_args.model_type}_{dataset}_{model_args.scan_type}_b{training_args.batch_size}_e{training_args.num_epochs}_lr{training_args.learning_rate}{'_mixup' if training_args.mixup > 0 else ''}{'_ema' if training_args.use_ema else ''}"

class FlaDatasetWrapper(Dataset):
    """Wrapper to convert HuggingFace Dataset to PyTorch Dataset"""
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        
        # Determine image and label keys
        if 'img' in self.dataset.column_names:
            self.img_key = 'img'
            self.label_key = 'fine_label' if 'fine_label' in self.dataset.column_names else 'label'
        elif 'image' in self.dataset.column_names:
            self.img_key = 'image'
            self.label_key = 'label'
        else:
            raise ValueError("Cannot determine image and label columns in dataset")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item[self.img_key]
        label = item[self.label_key]
        
        # 确保图像是RGB格式
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        if self.transform is not None:
            image = self.transform(image)
            
        return image, label

from torchvision import transforms
from torchvision.transforms import InterpolationMode

def get_datasets(data_args, training_args, model_args):
    """Load and process datasets"""
    dataset_map = {
        'cifar10': {'name': 'cifar10', 'num_classes': 10},
        'cifar100': {'name': 'cifar100', 'num_classes': 100},
        'tiny-imagenet': {'name': 'slegroux/tiny-imagenet-200-clean', 'num_classes': 200},
        'imagenet': {'name': 'ILSVRC/imagenet-1k', 'num_classes': 1000}
    }
    
    # Get dataset info
    dataset_key = data_args.dataset_name.lower()
    if (dataset_key in dataset_map):
        dataset_info = dataset_map[dataset_key]
        dataset_name = dataset_info['name']
        num_classes = dataset_info['num_classes']
    else:
        dataset_name = data_args.dataset_name
        # For custom datasets, try to infer number of classes
        if dataset_name.endswith("cifar10"):
            num_classes = 10
        elif dataset_name.endswith("cifar100"):
            num_classes = 100
        else:
            logging.warn(f"Cannot determine number of classes for {dataset_name} defaulting to 100")
            num_classes = 100
    
    # Load dataset
    try:
        dataset = load_dataset(dataset_name, trust_remote_code=True, cache_dir='data')
    except Exception as e:
        logging.error(f"Failed to load dataset {dataset_name}: {e}")
        raise
    
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16
    }
    dtype = dtype_map.get(model_args.dtype, torch.float32)

    from timm.data import create_transform

    train_transforms = create_transform(
            input_size=data_args.image_size,
            is_training=True,
            color_jitter=0.4,
            auto_augment="rand-m9-mstd0.5-inc1",
            re_prob=0.25,
            re_mode='pixel',
            re_count=1,
            interpolation='bicubic',
        )
    
    train_transforms = list(train_transforms.transforms)

    train_transforms.insert(0, transforms.Lambda(lambda x: x.convert('RGB') if x.mode != 'RGB' else x))

    train_transforms.append(transforms.Lambda(lambda x: x.to(dtype)))

    train_transforms = transforms.Compose(train_transforms)

    # log seq len, patch size and image size

    logging.info(f"Image size: {data_args.image_size}, Patch size: {data_args.patch_size}, Sequence length: {int((data_args.image_size / data_args.patch_size) ** 2)}")
    
    size = int((256 / 224) * data_args.image_size)

    val_transforms = transforms.Compose([
        transforms.Lambda(lambda x: x.convert('RGB') if x.mode != 'RGB' else x),
        transforms.Resize(size),
        transforms.CenterCrop(data_args.image_size), 
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD
        ),
        transforms.Lambda(lambda x: x.to(dtype))
    ])
    
    if 'train' in dataset:
        train_dataset = FlaDatasetWrapper(dataset['train'], train_transforms)
        eval_split = 'validation' if 'validation' in dataset else 'test'
        eval_dataset = FlaDatasetWrapper(dataset[eval_split], val_transforms)
    else:
        # If dataset doesn't have splits, split manually
        train_size = int(0.8 * len(dataset['train']))
        val_size = len(dataset['train']) - train_size
        train_subset, val_subset = torch.utils.data.random_split(
            dataset['train'], [train_size, val_size]
        )
        train_dataset = FlaDatasetWrapper(train_subset, train_transforms)
        eval_dataset = FlaDatasetWrapper(val_subset, val_transforms)
    
    logging.info(f"Loaded dataset {dataset_name} with {len(train_dataset)} training and {len(eval_dataset)} validation samples")
    logging.info(f"Using custom transforms for training and validation")
    
    return train_dataset, eval_dataset, num_classes

def get_model(model_args, data_args, num_classes):
    """Initialize model"""
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
            'window_size': model_args.attn_window_size,
            'block_size': model_args.attn_block_size,
            'topk': model_args.attn_topk,
            'block_counts': model_args.attn_block_counts,
            'stride': model_args.attn_stride,
            'chunk_size': model_args.attn_chunk_size,
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
        'rwkv7': (RWKV7VisionConfig, RWKV7ForImageClassification),
        'transformer': (TransformerVisionConfig, TransformerForImageClassification),
        'nsa' : (NSAVisionConfig, NSAForImageClassification)
    }

    if model_args.model_type not in model_classes:
        raise ValueError(f"Unsupported model type: {model_args.model_type}")

    ConfigClass, ModelClass = model_classes[model_args.model_type]
    config = ConfigClass(
        compress_attention=model_args.compress_attention,
        num_hidden_layers=model_args.num_hidden_layers,
        hidden_size=model_args.hidden_size,
        num_heads=model_args.num_heads,
        head_dim=model_args.head_dim if "gated_deltanet" in model_args.model_type else None,
        patch_size=data_args.patch_size,
        image_size=data_args.image_size, 
        num_classes=num_classes,
        attn_mode=model_args.attn_mode,
        fuse_cross_entropy=model_args.fuse_cross_entropy,
        attn=attn_config,
        train_scan_type=model_args.scan_type,
        attn_type=model_args.attn_type,
        hidden_dropout_prob=model_args.hidden_dropout_prob,
    )
    
    model = ModelClass(config)

    return model.to(dtype=dtype)

def print_model_info(model, model_args):
    """Print model information"""
    logging.info("\n" + "="*80)
    logging.info("Model Configuration:")
    logging.info("-"*40)
    
    # Print model parameter statistics
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

def train_one_epoch(
    epoch, model, loader, optimizer, loss_fn, device, 
    mixup_fn=None, model_ema=None, training_args=None, accelerator=None, lr_scheduler=None, teacher=None, temperature=1.0
):
    """Train for one epoch"""
    model.train()
    
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    
    end = time.time()
    last_idx = len(loader) - 1
    
    for batch_idx, (inputs, targets) in enumerate(loader):
        data_time_m.update(time.time() - end)
        
        # No need to move tensors to device when using accelerator
        if accelerator is None:
            inputs, targets = inputs.to(device), targets.to(device)
        
        if mixup_fn is not None:
            inputs, targets = mixup_fn(inputs, targets)
        
        outputs = model(pixel_values=inputs, output_hidden_states=(teacher != None))

        if teacher != None:
            with torch.no_grad():
                teacher_hidden_states = teacher(pixel_values=inputs, output_hidden_states=True).hidden_states

            student_hidden_states = outputs.hidden_states

            # calculate loss in each hiddenm_states using mse loss
            loss_distill = 0
            for i in range(len(student_hidden_states)):
                loss_distill += F.mse_loss(student_hidden_states[i], teacher_hidden_states[i])


        logits = outputs.logits

        loss = loss_fn(logits, targets)

        if teacher != None:
            loss = loss * 1e-9 + loss_distill
                
        # Use accelerator for backward pass
        if accelerator is not None:
            accelerator.backward(loss)
        else:
            loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()

        # lr_scheduler.step_update(epoch * len(loader) + batch_idx)
        lr_scheduler.step(epoch + (batch_idx / len(loader)))
        
        # Update EMA model
        if model_ema is not None:
            model_ema.update(model)
        
        # Record loss
        losses_m.update(loss.item(), inputs.size(0))
        
        batch_time_m.update(time.time() - end)
        
        # Log progress only on main process
        if (accelerator is None or accelerator.is_local_main_process) and (batch_idx % training_args.log_interval == 0 or batch_idx == last_idx):
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            logging.info(
                'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                'Loss: {loss.val:#.4g} ({loss.avg:#.3g})  '
                'LR: {lr:.3e}  '
                'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                'Data: {data_time.val:.3f}s ({data_time.avg:.3f}s)'.format(
                    epoch,
                    batch_idx, len(loader),
                    100. * batch_idx / last_idx,
                    loss=losses_m,
                    lr=current_lr,
                    batch_time=batch_time_m,
                    rate=inputs.size(0) / batch_time_m.val,
                    rate_avg=inputs.size(0) / batch_time_m.avg,
                    data_time=data_time_m
                )
            )
            
            # Log to wandb during training
            if training_args.report_to_wandb and accelerator is not None and accelerator.is_local_main_process:
                try:
                    import wandb
                    wandb.log({
                        'train/step': epoch * len(loader) + batch_idx,
                        'train/loss': losses_m.val,
                        'train/loss_avg': losses_m.avg,
                        'train/lr': current_lr,
                        'train/batch_time': batch_time_m.val,
                        'train/data_time': data_time_m.val
                    })
                except:
                    pass
        
        end = time.time()
    
    return {'loss': losses_m.avg}

def validate(model, loader, loss_fn, device, training_args, accelerator=None):
    """Validate model performance"""
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()
    
    model.eval()
    
    end = time.time()
    last_idx = len(loader) - 1
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            if accelerator is None:
                inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(pixel_values=inputs)

            logits = outputs.logits


            loss = loss_fn(logits, targets)
            
            # Calculate accuracy
            acc1, acc5 = accuracy(logits, targets, topk=(1, 5))
            
            # Update metrics
            losses_m.update(loss.item(), inputs.size(0))
            top1_m.update(acc1.item(), inputs.size(0))
            top5_m.update(acc5.item(), inputs.size(0))
            
            batch_time_m.update(time.time() - end)
            end = time.time()
            
            # Log progress only on main process
            if (accelerator is None or accelerator.is_local_main_process) and (batch_idx % training_args.log_interval == 0 or batch_idx == last_idx):
                logging.info(
                    'Test: [{:>4d}/{} ({:>3.0f}%)]  '
                    'Loss: {loss.val:#.4g} ({loss.avg:#.3g})  '
                    'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                    'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})  '
                    'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                    '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)'.format(
                        batch_idx, len(loader),
                        100. * batch_idx / last_idx,
                        loss=losses_m,
                        top1=top1_m,
                        top5=top5_m,
                        batch_time=batch_time_m,
                        rate=inputs.size(0) / batch_time_m.val,
                        rate_avg=inputs.size(0) / batch_time_m.avg
                    )
                )
    
    metrics = {'loss': losses_m.avg, 'top1': top1_m.avg, 'top5': top5_m.avg}
    return metrics

def main():
    """Main function to run training"""
    # Add start time tracking for total training time
    start_time = time.time()
    
    parser = argparse.ArgumentParser(description='Training with timm features')
    
    # Add arguments for all dataclasses
    for dc in [ModelArguments, DataArguments, TrainingArguments]:
        for field_name, field_value in dc.__dataclass_fields__.items():
            metadata = field_value.metadata
            help_text = metadata.get('help', '')
            default = field_value.default
            
            if field_value.type in [bool, type(None)]:
                parser.add_argument(f'--{field_name}', 
                                    action='store_true', 
                                    default=default, 
                                    help=help_text)
            else:
                parser.add_argument(f'--{field_name}', 
                                    type=field_value.type, 
                                    default=default, 
                                    help=help_text)
    
    args = parser.parse_args()
    
    # Convert namespace to dataclasses
    model_args = ModelArguments(
        compress_attention=args.compress_attention,
        model_type=args.model_type,
        num_hidden_layers=args.num_hidden_layers,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        channel_mixer_dim=args.channel_mixer_dim,
        attn_mode=args.attn_mode,
        head_dim=args.head_dim,
        fuse_cross_entropy=args.fuse_cross_entropy,
        scan_type=args.scan_type,
        use_attn=args.use_attn,
        attn_layers=args.attn_layers,
        hidden_dropout_prob=args.hidden_dropout_prob,
        attn_num_heads=args.attn_num_heads,
        attn_num_kv_heads=args.attn_num_kv_heads,
        attn_window_size=args.attn_window_size,
        attn_block_size=args.attn_block_size,
        attn_topk=args.attn_topk,
        attn_block_counts=args.attn_block_counts,
        attn_stride=args.attn_stride,
        attn_chunk_size=args.attn_chunk_size,
        attn_type=args.attn_type,
        dtype=args.dtype
    )
    
    data_args = DataArguments(
        dataset_name=args.dataset_name,
        image_size=args.image_size,
        patch_size=args.patch_size
    )
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        optimizer=args.optimizer,
        sched=args.sched,
        warmup_epochs=args.warmup_epochs,
        cooldown_epochs=args.cooldown_epochs,
        min_lr=args.min_lr,
        mixup=args.mixup,
        cutmix=args.cutmix,
        mixup_prob=args.mixup_prob,
        mixup_switch_prob=args.mixup_switch_prob,
        mixup_mode=args.mixup_mode,
        use_ema=args.use_ema,
        ema_decay=args.ema_decay,
        ema_force_cpu=args.ema_force_cpu if hasattr(args, 'ema_force_cpu') else False,
        seed=args.seed,
        workers=args.workers,
        pin_memory=args.pin_memory,
        log_interval=args.log_interval,
        eval_metric=args.eval_metric,
        report_to_wandb=args.report_to_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        label_smoothing=args.label_smoothing,
        use_augmix=args.use_augmix,
        ra_magnitude=args.ra_magnitude,
        ra_num_ops=args.ra_num_ops,
        save_freq=args.save_freq,
        eval_freq=args.eval_freq,
        save_checkpoint=args.save_checkpoint,
        save_total_limit=args.save_total_limit if hasattr(args, 'save_total_limit') else 3,
        init_from_pretrained=args.init_from_pretrained,
        init_model=args.init_model,
        training_mode=args.training_mode
    )
    
    # Initialize accelerator
    accelerator = Accelerator(mixed_precision="bf16")
    
    # Create output directory - only on main process
    if accelerator.is_local_main_process:
        os.makedirs(training_args.output_dir, exist_ok=True)
    
    # Setup logging and random seed
    if accelerator.is_local_main_process:
        setup_logging(training_args)
    set_seed(training_args.seed)
    
    # Generate wandb run name if needed
    if training_args.report_to_wandb and not training_args.wandb_run_name:
        training_args.wandb_run_name = get_wandb_run_name(model_args, data_args, training_args)
    
    # Initialize wandb if enabled - only on main process
    if training_args.report_to_wandb and accelerator.is_local_main_process:
        try:
            import wandb
            wandb.init(project=training_args.wandb_project, name=training_args.wandb_run_name)
            logging.info(f"Wandb initialized with run name: {training_args.wandb_run_name}")
        except ImportError:
            logging.warning("Wandb not installed, skipping wandb initialization")
            training_args.report_to_wandb = False
    
    # Get datasets and dataloaders
    train_dataset, eval_dataset, num_classes = get_datasets(data_args, training_args, model_args)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_args.batch_size,
        shuffle=True,
        num_workers=training_args.workers,
        pin_memory=training_args.pin_memory,
        drop_last=True
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=training_args.val_batch_size,
        shuffle=False,
        num_workers=training_args.workers,
        pin_memory=training_args.pin_memory,
        drop_last=False
    )
    
    # Create model
    model = get_model(model_args, data_args, num_classes)

    teacher = None

    # init if needed
    if training_args.init_from_pretrained:
        if training_args.init_model == "dino":
            # nice format, some lines
            logging.info("="*80)
            logging.info("Initializing from DINO")
            logging.info("="*80)
            model.backbone = init_from_dino2_base_p14(model.backbone, train_mlp=True, return_pretrained=False)
        elif training_args.init_model == "siglip":
            logging.info("="*80)
            logging.info("Initializing from SigLIP")
            logging.info("="*80)
            if training_args.training_mode == "label":
                model.backbone = init_from_siglip2_base_p16_224(model.backbone, train_mlp=True, init_embedding=True, return_pretrained=False)
            else:
                logging.info("Retaining SigLIP for distillation and only training the backbone")
                model.backbone, teacher = init_from_siglip2_base_p16_224(model.backbone, train_mlp=True, init_embedding=True, return_pretrained=True)
        else:
            raise ValueError(f"Unknown init model: {training_args.init_model}")
    
    
    if teacher != None:
        # eval
        teacher.eval()
    
    # Print model info only on main process
    if accelerator.is_local_main_process:
        print_model_info(model, model_args)
    
    # Setup mixup if enabled
    mixup_fn = None
    if training_args.mixup > 0 or training_args.cutmix > 0:
        logging.info("Using mixup/cutmix data augmentation")
        mixup_fn = Mixup(
            mixup_alpha=training_args.mixup,
            cutmix_alpha=training_args.cutmix,
            prob=training_args.mixup_prob,
            switch_prob=training_args.mixup_switch_prob,
            mode=training_args.mixup_mode,
            label_smoothing=training_args.label_smoothing,
            num_classes=num_classes
        )
    
    # Setup EMA model if enabled
    model_ema = None
    if training_args.use_ema:
        model_ema = ModelEma(
            model,
            decay=training_args.ema_decay,
            device='cpu' if training_args.ema_force_cpu else '',
            resume=''
        )
        logging.info(f"Using EMA with decay = {training_args.ema_decay}")
    
    # Setup loss function
    if training_args.mixup > 0.0:
        # Smoothed loss for mixup
        train_loss_fn = SoftTargetCrossEntropy()
    elif training_args.label_smoothing > 0.0:
        # Smoothed loss for label smoothing
        train_loss_fn = LabelSmoothingCrossEntropy(smoothing=training_args.label_smoothing)
    else:
        # Standard cross entropy
        train_loss_fn = torch.nn.CrossEntropyLoss()
    
    # Always use standard loss for validation
    validate_loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=training_args.learning_rate,
        weight_decay=training_args.weight_decay
    )

    
    # # Setup LR scheduler
    lr_scheduler, num_epochs = create_scheduler_v2(
        min_lr=5e-6,
        warmup_epochs=training_args.warmup_epochs,
        num_epochs=training_args.num_epochs,
        warmup_lr=5e-7,
        optimizer=optimizer,
        updates_per_epoch=len(train_loader)
    )

    logging.info(f"Using LR scheduler: {training_args.num_epochs} epochs, {len(train_loader)} updates per epoch")
    logging.info(f"Training for {num_epochs} epochs")
    
    # Setup checkpoint saver
    saver = None
    if training_args.save_checkpoint:
        checkpoint_dir = os.path.join(training_args.output_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        saver = CheckpointSaver(
            model=model,
            optimizer=optimizer,
            args=training_args,
            model_ema=model_ema,
            amp_scaler=None,
            checkpoint_dir=checkpoint_dir,
            recovery_dir=checkpoint_dir,
            decreasing=False,
            max_history=training_args.save_total_limit
        )
    
    # Prepare with accelerator
    model, optimizer, train_loader, eval_loader, lr_scheduler, teacher = accelerator.prepare(
        model, optimizer, train_loader, eval_loader, lr_scheduler, teacher
    )
    
    # Device is managed by accelerator
    device = accelerator.device
    if accelerator.is_local_main_process:
        logging.info(f"Using device: {device}")
    
    # Setup EMA model if enabled - after accelerator prepare
    model_ema = None
    if training_args.use_ema:
        # EMA doesn't need to be wrapped by accelerator
        model_ema = ModelEma(
            accelerator.unwrap_model(model),
            decay=training_args.ema_decay,
            device='cpu' if training_args.ema_force_cpu else None,
            resume=''
        )
        if accelerator.is_local_main_process:
            logging.info(f"Using EMA with decay = {training_args.ema_decay}")
    
    # Setup checkpoint saver - only on main process
    saver = None
    if training_args.save_checkpoint and accelerator.is_local_main_process:
        checkpoint_dir = os.path.join(training_args.output_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        saver = CheckpointSaver(
            model=accelerator.unwrap_model(model),  # Use unwrapped model for saving
            optimizer=optimizer,
            args=training_args,
            model_ema=model_ema,
            amp_scaler=None,
            checkpoint_dir=checkpoint_dir,
            recovery_dir=checkpoint_dir,
            decreasing=False,
            max_history=training_args.save_total_limit
        )
    
    # Ensure all processes are synced before starting training
    accelerator.wait_for_everyone()
    
    # Training loop
    best_metric = None
    best_epoch = None
    
    if accelerator.is_local_main_process:
        logging.info(f"Starting training for {num_epochs} epochs")
    
    for epoch in range(num_epochs):
        # Train for one epoch with accelerator
        train_metrics = train_one_epoch(
            epoch=epoch,
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=train_loss_fn,
            device=device,
            mixup_fn=mixup_fn,
            model_ema=model_ema,
            training_args=training_args,
            accelerator=accelerator,
            lr_scheduler=lr_scheduler,
            teacher=teacher
        )
        
        # Update learning rate
        # lr_scheduler.step(epoch + 1)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log LR - only on main process
        if accelerator.is_local_main_process:
            logging.info(f"Epoch {epoch+1}/{num_epochs} completed. LR: {current_lr:.3e}")
        
        # Evaluate on validation set
        if (epoch % training_args.eval_freq == 0 or epoch == num_epochs - 1) and training_args.training_mode == 'label':
            eval_metrics = validate(
                model=model,
                loader=eval_loader,
                loss_fn=validate_loss_fn,
                device=device,
                training_args=training_args,
                accelerator=accelerator,
            )
            
            # Log validation metrics - only on main process
            if accelerator.is_local_main_process:
                logging.info(
                    f"Validation: "
                    f"Loss: {eval_metrics['loss']:.4f}, "
                    f"Top-1: {eval_metrics['top1']:.2f}%, "
                    f"Top-5: {eval_metrics['top5']:.2f}%, "
                    f"LR: {current_lr:.3e}"
                )
                
                # Comprehensive wandb logging after each epoch - only on main process
                if training_args.report_to_wandb:
                    try:
                        import wandb
                        wandb_metrics = {
                            'epoch': epoch,
                            'train/epoch_loss': train_metrics['loss'],
                            'val/loss': eval_metrics['loss'],
                            'val/top1': eval_metrics['top1'],
                            'val/top5': eval_metrics['top5'],
                            'lr': current_lr,
                            'epoch_progress': (epoch + 1) / num_epochs
                        }
                        wandb.log(wandb_metrics)
                    except:
                        pass
            
                # Evaluate EMA model if available - only on main process
                if model_ema is not None:
                    ema_eval_metrics = validate(
                        model=model_ema.ema,
                        loader=eval_loader,
                        loss_fn=validate_loss_fn,
                        device=device,
                        training_args=training_args,
                    )
                    logging.info(
                        f"EMA Validation: "
                        f"Loss: {ema_eval_metrics['loss']:.4f}, "
                        f"Top-1: {ema_eval_metrics['top1']:.2f}%, "
                        f"Top-5: {ema_eval_metrics['top5']:.2f}%"
                    )
                    
                    # Log EMA metrics to wandb
                    if training_args.report_to_wandb:
                        try:
                            import wandb
                            wandb.log({
                                'epoch': epoch,
                                'ema/val_loss': ema_eval_metrics['loss'],
                                'ema/val_top1': ema_eval_metrics['top1'],
                                'ema/val_top5': ema_eval_metrics['top5']
                            })
                        except:
                            pass
                    
                    # Use EMA metrics for tracking best model if better
                    if ema_eval_metrics['top1'] > eval_metrics['top1']:
                        eval_metrics = ema_eval_metrics
                        logging.info("Using EMA metrics as they're better")
            
                # Track best metric - only on main process
                eval_metric = eval_metrics['top1']
                if best_metric is None or eval_metric > best_metric:
                    best_metric = eval_metric
                    best_epoch = epoch
                    
                    # Log new best model to wandb
                    if training_args.report_to_wandb:
                        try:
                            import wandb
                            wandb.log({
                                'best/epoch': best_epoch,
                                'best/metric': best_metric
                            })
                            # Also update summary for quick reference
                            wandb.run.summary["best_accuracy"] = best_metric
                            wandb.run.summary["best_epoch"] = best_epoch
                        except:
                            pass
                    
                    # Save best model - only on main process
                    if training_args.save_checkpoint:
                        save_path = os.path.join(training_args.output_dir, 'best_model.pth')
                        # save hf model
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.save_pretrained(os.path.join(training_args.output_dir, 'best_model_hf'))
                        to_save = {
                            'model': unwrapped_model.state_dict(),
                            'model_ema': model_ema.ema.state_dict() if model_ema is not None else None,
                            'optimizer': optimizer.state_dict(),
                            'epoch': epoch,
                            'args': vars(args),
                            'model_args': vars(model_args),
                            'data_args': vars(data_args),
                            'training_args': vars(training_args),
                            'best_metric': best_metric,
                        }
                        torch.save(to_save, save_path)
                        logging.info(f"Saved best model to {save_path}")
        
            # Save checkpoint if needed - only on main process
            if training_args.save_checkpoint and epoch % training_args.save_freq == 0 and accelerator.is_local_main_process:
                # Use saver to keep track of top models
                save_metric = eval_metrics.get(training_args.eval_metric, eval_metrics['top1'])
                saver.save_checkpoint(epoch, metric=save_metric)
        
        # Ensure all processes are synced before next epoch
        accelerator.wait_for_everyone()
    
    # Ensure all processes are synced before final steps
    accelerator.wait_for_everyone()
    
    # Calculate total training time
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    # save last model
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(os.path.join(training_args.output_dir, 'last_model_hf'))
    
    # Print final best results - only on main process
    if accelerator.is_local_main_process and training_args.training_mode == "label":
        logging.info(f"Training completed in {int(hours)}h {int(minutes)}m {seconds:.2f}s")
        logging.info(f"Best metric: {best_metric:.2f} (epoch {best_epoch})")

        # Log final stats to wandb
        if training_args.report_to_wandb:
            try:
                import wandb
                wandb.log({
                    'final/best_metric': best_metric,
                    'final/best_epoch': best_epoch,
                    'final/total_epochs': num_epochs,
                    'final/training_time_seconds': total_time,
                    'final/training_time': f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
                })
                
                # Update run summary with final stats
                wandb.run.summary["total_time"] = total_time
                wandb.run.summary["training_time"] = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
                wandb.run.summary["best_accuracy"] = best_metric
                wandb.run.summary["best_epoch"] = best_epoch
            except:
                logging.warning("Failed to log final stats to wandb")
        
        # Final save of the model - only on main process
        if training_args.save_checkpoint:
            save_path = os.path.join(training_args.output_dir, 'final_model.pth')
            unwrapped_model = accelerator.unwrap_model(model)
            to_save = {
                'model': unwrapped_model.state_dict(),
                'model_ema': model_ema.ema.state_dict() if model_ema is not None else None,
                'epoch': num_epochs - 1,
                'args': vars(args),
                'best_metric': best_metric,
                'best_epoch': best_epoch,
                'training_time': total_time
            }
            torch.save(to_save, save_path)
            logging.info(f"Saved final model to {save_path}")

if __name__ == "__main__":
    main()