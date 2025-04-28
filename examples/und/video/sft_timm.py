import pathlib
import pytorchvideo.data
import os
import argparse
import time
import logging
import random
import math
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    UniformTemporalSubsample,
)
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pytorchvideo.transforms import create_video_transform
from timm.scheduler import create_scheduler_v2
import torch
import numpy as np
from torch import optim
from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
    Resize,
    ColorJitter,
    RandomGrayscale,
)
import evaluate
from accelerate import Accelerator

# Import utility classes for training
from timm.utils import AverageMeter, accuracy, ModelEma, CheckpointSaver

from flazoo.helpers.linearizer import init_from_siglip2_base_p16_224, init_from_dino2_small_p14
metric = evaluate.load("accuracy")

# Import model classes
from flazoo.models.und.delta_net import DeltaNetForVideoClassification
from flazoo.models.und.delta_net import DeltaNetVideoConfig
from transformers import VideoMAEImageProcessor

from training.datasets.video_sm import build_dataset
from training.datasets.video_sm.mixup import Mixup
from training.datasets.video_sm.utils import multiple_samples_collate
from functools import partial

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
# Define model arguments class
@dataclass
class ModelArguments:
    """Arguments for video model configuration"""
    model_type: str = field(default="deltanet", metadata={"help": "Model type: deltanet, gated_deltanet, videomae"})
    num_hidden_layers: int = field(default=3, metadata={"help": "Number of hidden layers"})
    hidden_size: int = field(default=256, metadata={"help": "Hidden dimension size"})
    num_heads: int = field(default=16, metadata={"help": "Number of attention heads"})
    channel_mixer_dim: int = field(default=None, metadata={"help": "Channel mixer dimension, defaults to 4*hidden_size"})
    attn_mode: str = field(default="chunk", metadata={"help": "Attention mode"})
    head_dim: int = field(default=64, metadata={"help": "Head dimension, used for gated deltanet"})
    fuse_cross_entropy: bool = field(default=False, metadata={"help": "Whether to fuse cross entropy"})
    scan_type: str = field(default="uni-scan", metadata={"help": "Scan type: uni-scan, bi-scan, cross-scan, random-scan"})
    use_attn: bool = field(default=False, metadata={"help": "Whether to use attention"})
    attn_layers: str = field(default="0,1", metadata={"help": "Comma-separated list of layer indices to apply attention"})
    hidden_dropout_prob: float = field(default=0.0, metadata={"help": "Hidden dropout probability"})
    attn_num_heads: int = field(default=16, metadata={"help": "Number of attention heads"})
    attn_num_kv_heads: int = field(default=None, metadata={"help": "Number of key/value heads for attention"})
    attn_window_size: int = field(default=None, metadata={"help": "Window size for attention"})
    attn_block_size: int = field(default=None, metadata={"help": "Block size for attention"})
    attn_stride: int = field(default=None, metadata={"help": "Stride for attention"})
    attn_chunk_size: int = field(default=None, metadata={"help": "Chunk size for attention"})
    attn_block_counts: int = field(default=None, metadata={"help": "Block counts for attention"})
    attn_topk: int = field(default=None, metadata={"help": "Top-k for attention"})
    attn_type: str = field(default=None, metadata={"help": "Attention type"})
    decoder_hidden_size: int = field(default=256, metadata={"help": "Decoder hidden size"})
    decoder_num_hidden_layers: int = field(default=1, metadata={"help": "Number of decoder hidden layers"})
    dtype: str = field(default="bfloat16", metadata={"help": "Model precision type: float32, float16, or bfloat16"})
    pretrained_model_name: str = field(default=None, metadata={"help": "Pretrained model name or path"})

# Define data arguments class
@dataclass
class DataArguments:
    # Dataset parameters
    prefix: str = field(default='', metadata={"help": "prefix for data"})
    split: str = field(default=' ', metadata={"help": "split for metadata"})
    patch_size: int = field(default=16, metadata={"help": "patch size"})
    filename_tmpl: str = field(default='img_{:05}.jpg', metadata={"help": "file template"})
    data_path: str = field(default='you_data_path', metadata={"help": "dataset path"})
    input_size: int = field(default=224, metadata={"help": "images input size for backbone"})
    eval_data_path: Optional[str] = field(default=None, metadata={"help": "dataset path for evaluation"})
    nb_classes: int = field(default=400, metadata={"help": "number of the classification types"})
    imagenet_default_mean_and_std: bool = field(default=True, metadata={"help": "use ImageNet default mean and std"})
    use_decord: bool = field(default=True, metadata={"help": "whether use decord to load video, otherwise load image"})
    num_segments: int = field(default=1, metadata={"help": "number of segments"})
    num_frames: int = field(default=16, metadata={"help": "number of frames"})
    sampling_rate: int = field(default=4, metadata={"help": "sampling rate"})
    trimmed: int = field(default=60, metadata={"help": "trimmed length"})
    time_stride: int = field(default=16, metadata={"help": "time stride"})
    data_set: str = field(default='Kinetics', metadata={"help": "dataset"})
    log_dir: Optional[str] = field(default=None, metadata={"help": "path where to tensorboard log"})
    device: str = field(default='cuda', metadata={"help": "device to use for training / testing"})
    resume: str = field(default='', metadata={"help": "resume from checkpoint"})
    auto_resume: bool = field(default=True, metadata={"help": "auto resume from checkpoint"})
    save_ckpt: bool = field(default=True, metadata={"help": "save checkpoint"})
    start_epoch: int = field(default=0, metadata={"help": "start epoch"})
    test_best: bool = field(default=False, metadata={"help": "Whether test the best model"})
    eval: bool = field(default=False, metadata={"help": "Perform evaluation only"})
    dist_eval: bool = field(default=False, metadata={"help": "Enabling distributed evaluation"})
    num_workers: int = field(default=10, metadata={"help": "number of workers"})
    pin_mem: bool = field(default=True, metadata={"help": "Pin CPU memory in DataLoader for more efficient transfer to GPU"})
    no_amp: bool = field(default=False, metadata={"help": "disable mixed precision"})

    # Augmentation parameters
    color_jitter: float = field(default=0.4, metadata={"help": "Color jitter factor (default: 0.4)"})
    num_sample: int = field(default=2, metadata={"help": "Repeated_aug (default: 2)"})
    aa: str = field(default='rand-m7-n4-mstd0.5-inc1', metadata={"help": "Use AutoAugment policy. \"v0\" or \"original\". \" + \"(default: rand-m7-n4-mstd0.5-inc1)"})
    smoothing: float = field(default=0.1, metadata={"help": "Label smoothing (default: 0.1)"})
    train_interpolation: str = field(default='bicubic', metadata={"help": "Training interpolation (random, bilinear, bicubic default: \"bicubic\")"})

    # Evaluation parameters
    crop_pct: Optional[float] = field(default=None, metadata={"help": "Crop percentage"})
    short_side_size: int = field(default=224, metadata={"help": "Short side size"})
    test_num_segment: int = field(default=5, metadata={"help": "Test number of segments"})
    test_num_crop: int = field(default=3, metadata={"help": "Test number of crops"})

    # Random Erase params
    reprob: float = field(default=0.25, metadata={"help": "Random erase prob (default: 0.25)"})
    remode: str = field(default='pixel', metadata={"help": "Random erase mode (default: \"pixel\")"})
    recount: int = field(default=1, metadata={"help": "Random erase count (default: 1)"})
    resplit: bool = field(default=False, metadata={"help": "Do not random erase first (clean) augmentation split"})

    # Mixup params
    mixup: float = field(default=0.8, metadata={"help": "mixup alpha, mixup enabled if > 0."})
    cutmix: float = field(default=1.0, metadata={"help": "cutmix alpha, cutmix enabled if > 0."})
    cutmix_minmax: Optional[List[float]] = field(default=None, metadata={"help": "cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)"})
    mixup_prob: float = field(default=1.0, metadata={"help": "Probability of performing mixup or cutmix when either/both is enabled"})
    mixup_switch_prob: float = field(default=0.5, metadata={"help": "Probability of switching to cutmix when both mixup and cutmix enabled"})
    mixup_mode: str = field(default='batch', metadata={"help": "How to apply mixup/cutmix params. Per \"batch\", \"pair\", or \"elem\""})


# Define training arguments class
@dataclass
class TrainingArguments:
    """Arguments for training configuration"""
    output_dir: str = field(default="output", metadata={"help": "Output directory"})
    num_epochs: int = field(default=1, metadata={"help": "Number of training epochs"})
    batch_size: int = field(default=2, metadata={"help": "Training batch size"})
    val_batch_size: int = field(default=2, metadata={"help": "Validation batch size"})
    learning_rate: float = field(default=5e-5, metadata={"help": "Initial learning rate"})
    weight_decay: float = field(default=0.01, metadata={"help": "Weight decay"})
    momentum: float = field(default=0.9, metadata={"help": "Momentum for SGD optimizer"})
    optimizer: str = field(default="adamw", metadata={"help": "Optimizer type: adamw, sgd"})

    model: str = field(default="deltanet", metadata={"help": "Model type"})

    # Learning rate scheduler parameters
    sched: str = field(default="cosine", metadata={"help": "LR scheduler type"})
    warmup_epochs: int = field(default=1, metadata={"help": "Warmup epochs"})
    cooldown_epochs: int = field(default=0, metadata={"help": "Cooldown epochs"})
    min_lr: float = field(default=5e-6, metadata={"help": "Minimum learning rate"})

    # EMA parameters
    use_ema: bool = field(default=False, metadata={"help": "Whether to use EMA"})
    ema_decay: float = field(default=0.9999, metadata={"help": "EMA decay rate"})
    ema_force_cpu: bool = field(default=False, metadata={"help": "Force EMA to be stored on CPU"})

    # Logging parameters
    logging_steps: int = field(default=10, metadata={"help": "Logging steps"})
    log_interval: int = field(default=10, metadata={"help": "Logging interval in batches"})
    eval_freq: int = field(default=1, metadata={"help": "Evaluation frequency in epochs"})
    save_freq: int = field(default=1, metadata={"help": "Save frequency in epochs"})
    log_to_file: bool = field(default=False, metadata={"help": "Whether to log to a file"})
    log_file: str = field(default="logs/training_log.log", metadata={"help": "Path to log file"})

    # Checkpoint parameters
    save_checkpoint: bool = field(default=False, metadata={"help": "Whether to save checkpoints"})
    save_total_limit: int = field(default=3, metadata={"help": "Maximum number of checkpoints to keep"})
    load_best_model_at_end: bool = field(default=True, metadata={"help": "Whether to load the best model at the end"})
    metric_for_best_model: str = field(default="accuracy", metadata={"help": "Metric for best model"})

    # Other parameters
    seed: int = field(default=42, metadata={"help": "Random seed"})
    workers: int = field(default=4, metadata={"help": "Number of data loading workers"})
    pin_memory: bool = field(default=True, metadata={"help": "Whether to use pin memory in data loaders"})
    eval_metric: str = field(default="top1", metadata={"help": "Evaluation metric"})

# Model mapping dictionary
model_classes = {
    "deltanet": (DeltaNetVideoConfig, DeltaNetForVideoClassification),
}

def get_model(model_args, data_args, num_classes):
    """Initialize model based on configuration"""
    # Set up dtype
    global dtype_map
    if model_args.dtype not in dtype_map:
        raise ValueError(f"Unsupported dtype: {model_args.dtype}")
    dtype = dtype_map[model_args.dtype]

    # Set up attention configuration if needed
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

    # Initialize model based on model type
    if model_args.model_type in ["deltanet", "gated_deltanet"]:
        ConfigClass, ModelClass = model_classes[model_args.model_type]
        config = ConfigClass(
            num_hidden_layers=model_args.num_hidden_layers,
            hidden_size=model_args.hidden_size,
            num_heads=model_args.num_heads,
            head_dim=model_args.head_dim if "gated_deltanet" in model_args.model_type else None,
            patch_size=data_args.patch_size,
            image_size=data_args.input_size,
            num_classes=num_classes,
            attn_mode=model_args.attn_mode,
            fuse_cross_entropy=model_args.fuse_cross_entropy,
            attn=attn_config,
            train_scan_type=model_args.scan_type,
            attn_type=model_args.attn_type,
            hidden_dropout_prob=model_args.hidden_dropout_prob,
            decoder_hidden_size=model_args.decoder_hidden_size,
            decoder_num_hidden_layers=model_args.decoder_num_hidden_layers,
            num_frames=data_args.num_frames,
        )
        model = ModelClass(config)
    else:
        raise ValueError(f"Unsupported model type: {model_args.model_type}")

    return model.to(dtype=dtype)

# Define dtype mapping
dtype_map = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16
}

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_one_epoch(epoch, model, loader, optimizer, device, training_args, lr_scheduler=None, accelerator=None, model_ema=None, mixup_fn=None):
    """Train for one epoch"""
    model.train()

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()

    end = time.time()
    last_idx = len(loader) - 1

    for batch_idx, batch in enumerate(loader):
        data_time_m.update(time.time() - end)

        # Process batch data
        pixel_values = batch[0] # B, C, F, H, W
        # make it B, F, C, H, W
        pixel_values = pixel_values.permute(0, 2, 1, 3, 4)
        labels = batch[1]

        # Move to device if not using accelerator
        if accelerator is None:
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)

        # Forward pass
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss

        # Backward pass
        if accelerator is not None:
            accelerator.backward(loss)
        else:
            loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        # Update learning rate
        if lr_scheduler is not None:
            lr_scheduler.step(epoch + (batch_idx / len(loader)))

        # Update EMA model if available
        if model_ema is not None:
            model_ema.update(model)

        # Record loss
        losses_m.update(loss.item(), pixel_values.size(0))

        batch_time_m.update(time.time() - end)

        # Log progress
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
                    rate=pixel_values.size(0) / batch_time_m.val,
                    rate_avg=pixel_values.size(0) / batch_time_m.avg,
                    data_time=data_time_m
                )
            )

        end = time.time()

    return {'loss': losses_m.avg}

def validate(model, loader, device, training_args, accelerator=None):
    """Validate model performance"""
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):

            # Process batch data
            pixel_values = batch["pixel_values"]
            labels = batch["labels"]

            # Move to device if not using accelerator
            if accelerator is None:
                pixel_values = pixel_values.to(device)
                labels = labels.to(device)

            # Forward pass
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            # Calculate accuracy
            acc1, acc5 = accuracy(logits, labels, topk=(1, min(5, logits.size(1))))

            # Update metrics
            losses_m.update(loss.item(), pixel_values.size(0))
            top1_m.update(acc1.item(), pixel_values.size(0))
            top5_m.update(acc5.item(), pixel_values.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()

            # Log progress
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
                        rate=pixel_values.size(0) / batch_time_m.val,
                        rate_avg=pixel_values.size(0) / batch_time_m.avg
                    )
                )

    metrics = {'loss': losses_m.avg, 'top1': top1_m.avg, 'top5': top5_m.avg}
    return metrics

def main():

    # Initialize accelerator for mixed precision training
    accelerator = Accelerator(mixed_precision="bf16")
    """Main function to run video classification training"""
    # Add start time tracking for total training time
    start_time = time.time()

    parser = argparse.ArgumentParser(description='Video Classification Training')

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
            elif field_name == 'log_file':
                # Special handling for log_file parameter to ensure it's parsed as a string
                parser.add_argument(f'--{field_name}',
                                  type=str,
                                  default=default,
                                  help=help_text)
            else:
                parser.add_argument(f'--{field_name}',
                                  type=field_value.type,
                                  default=default,
                                  help=help_text)

    args = parser.parse_args()

    # Convert namespace to dataclasses
    # Convert namespace to dataclasses
    model_args = ModelArguments(
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
        attn_stride=args.attn_stride,
        attn_chunk_size=args.attn_chunk_size,
        attn_block_counts=args.attn_block_counts,
        attn_topk=args.attn_topk,
        attn_type=args.attn_type,
        decoder_hidden_size=args.decoder_hidden_size,
        decoder_num_hidden_layers=args.decoder_num_hidden_layers,
        dtype=args.dtype,
        pretrained_model_name=args.pretrained_model_name,
    )

    data_args = DataArguments(
        prefix=args.prefix,
        split=args.split,
        patch_size=args.patch_size,
        filename_tmpl=args.filename_tmpl,
        data_path=args.data_path,
        input_size=args.input_size,
        eval_data_path=args.eval_data_path,
        nb_classes=args.nb_classes,
        imagenet_default_mean_and_std=args.imagenet_default_mean_and_std,
        use_decord=args.use_decord,
        num_segments=args.num_segments,
        num_frames=args.num_frames,
        sampling_rate=args.sampling_rate,
        trimmed=args.trimmed,
        time_stride=args.time_stride,
        data_set=args.data_set,
        log_dir=args.log_dir,
        device=args.device,
        resume=args.resume,
        auto_resume=args.auto_resume,
        save_ckpt=args.save_ckpt,
        start_epoch=args.start_epoch,
        test_best=args.test_best,
        eval=args.eval,
        dist_eval=args.dist_eval,
        num_workers=args.num_workers,
        pin_mem=args.pin_mem,
        no_amp=args.no_amp,
        
        # Augmentation parameters
        color_jitter=args.color_jitter,
        num_sample=args.num_sample,
        aa=args.aa,
        smoothing=args.smoothing,
        train_interpolation=args.train_interpolation,
        
        # Evaluation parameters
        crop_pct=args.crop_pct,
        short_side_size=args.short_side_size,
        test_num_segment=args.test_num_segment,
        test_num_crop=args.test_num_crop,
        
        # Random Erase params
        reprob=args.reprob,
        remode=args.remode,
        recount=args.recount,
        resplit=args.resplit,
        
        # Mixup params
        mixup=args.mixup,
        cutmix=args.cutmix,
        cutmix_minmax=args.cutmix_minmax,
        mixup_prob=args.mixup_prob,
        mixup_switch_prob=args.mixup_switch_prob,
        mixup_mode=args.mixup_mode,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        momentum=args.momentum if hasattr(args, 'momentum') else 0.9,
        optimizer=args.optimizer if hasattr(args, 'optimizer') else 'adamw',
        model=args.model if hasattr(args, 'model') else 'deltanet',

        # Learning rate scheduler parameters
        sched=args.sched if hasattr(args, 'sched') else 'cosine',
        warmup_epochs=args.warmup_epochs if hasattr(args, 'warmup_epochs') else 1,
        cooldown_epochs=args.cooldown_epochs if hasattr(args, 'cooldown_epochs') else 0,
        min_lr=args.min_lr if hasattr(args, 'min_lr') else 5e-6,

        # EMA parameters
        use_ema=args.use_ema if hasattr(args, 'use_ema') else False,
        ema_decay=args.ema_decay if hasattr(args, 'ema_decay') else 0.9999,
        ema_force_cpu=args.ema_force_cpu if hasattr(args, 'ema_force_cpu') else False,

        # Logging parameters
        logging_steps=args.logging_steps,
        log_interval=args.log_interval if hasattr(args, 'log_interval') else 50,
        eval_freq=args.eval_freq if hasattr(args, 'eval_freq') else 1,
        save_freq=args.save_freq if hasattr(args, 'save_freq') else 1,
        log_to_file=args.log_to_file,
        log_file=args.log_file,

        # Checkpoint parameters
        save_checkpoint=args.save_checkpoint if hasattr(args, 'save_checkpoint') else True,
        save_total_limit=args.save_total_limit if hasattr(args, 'save_total_limit') else 3,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,

        # Other parameters
        seed=args.seed if hasattr(args, 'seed') else 42,
        workers=args.workers if hasattr(args, 'workers') else 4,
        pin_memory=args.pin_memory if hasattr(args, 'pin_memory') else True,
        eval_metric=args.eval_metric if hasattr(args, 'eval_metric') else 'top1',
    )

    # Setup logging - only on main process
    if accelerator.is_local_main_process:
        setup_logging(training_args)

    # Log arguments in detail - only on main process
    if accelerator.is_local_main_process:
        logging.info("========== CONFIGURATION ==========")
        logging.info("Model arguments:")
        for field_name, field_value in model_args.__dataclass_fields__.items():
            value = getattr(model_args, field_name)
            logging.info(f"  {field_name}: {value}")

        logging.info("\nData arguments:")
        for field_name, field_value in data_args.__dataclass_fields__.items():
            value = getattr(data_args, field_name)
            logging.info(f"  {field_name}: {value}")

        logging.info("\nTraining arguments:")
        for field_name, field_value in training_args.__dataclass_fields__.items():
            value = getattr(training_args, field_name)
            logging.info(f"  {field_name}: {value}")

    # Create model with dummy number of classes (will be updated after dataset loading)
    model = get_model(model_args, data_args, args.nb_classes)

    # model.backbone = init_from_siglip2_base_p16_224(model.backbone, train_mlp=True, init_embedding=False)
    # log about init using a nice format
    logging.info("="*80)
    logging.info("Initializing from DINO")
    model.backbone = init_from_dino2_small_p14(model.backbone, train_mlp=True, init_embedding=False)
    logging.info("="*80)

    if accelerator.is_local_main_process:
        logging.info(f"Model created: {model.__class__.__name__}")

    train_dataset, args.nb_classes = build_dataset(is_train=True, test_mode=False, args=args)
    val_dataset, _ = build_dataset(is_train=False, test_mode=False, args=args)
    test_dataset, _ = build_dataset(is_train=False, test_mode=True, args=args)

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)
        
    # Log model parameters - only on main process
    if accelerator.is_local_main_process:
        logging.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

        # Calculate and log sequence length
        # log total seq len
        total_seq_len = data_args.num_frames * (data_args.input_size // data_args.patch_size) ** 2
        logging.info(f"\nTotal sequence length: {total_seq_len} tokens")
        logging.info(f"  - Image size: {data_args.input_size}")
        logging.info(f"  - Patch size: {data_args.patch_size}")
        logging.info(f"  - Spatial tokens: {(data_args.input_size // data_args.patch_size) ** 2}")
        logging.info(f"  - Number of frames: {data_args.num_frames}")
        logging.info("===================================")

    # Set random seed for reproducibility
    set_seed(training_args.seed)

    # Create output directory
    os.makedirs(training_args.output_dir, exist_ok=True)

    num_tasks = accelerator.num_processes
    global_rank = accelerator.process_index
    sampler_train = torch.utils.data.DistributedSampler(
        train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    print("Sampler_train = %s" % str(sampler_train))
    if args.dist_eval:
        if len(val_dataset) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')
    sampler_val = torch.utils.data.DistributedSampler(
        val_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    sampler_test = torch.utils.data.DistributedSampler(
        test_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    # else:
    # sampler_val = torch.utils.data.SequentialSampler(val_dataset)
    # sampler_test = torch.utils.data.SequentialSampler(test_dataset)

    if args.num_sample > 1:
        collate_func = partial(multiple_samples_collate, fold=False)
    else:
        collate_func = None

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        collate_fn=collate_func,
        persistent_workers=True
    )

    if val_dataset is not None:
        val_loader = torch.utils.data.DataLoader(
            val_dataset, sampler=sampler_val,
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
            persistent_workers=True
        )
    else:
        val_loader = None

    if test_dataset is not None:
        test_loader = torch.utils.data.DataLoader(
            test_dataset, sampler=sampler_test,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
            persistent_workers=True
        )
    else:
        test_loader = None

    # Setup optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=training_args.learning_rate,
        weight_decay=training_args.weight_decay
    )

    len_train_loader = len(train_loader)
    len_val_loader = len(val_loader)


    lr_scheduler, _ = create_scheduler_v2(
        min_lr=5e-6,
        warmup_epochs=training_args.warmup_epochs,
        num_epochs=training_args.num_epochs,
        warmup_lr=5e-7,
        optimizer=optimizer,
        updates_per_epoch=len_train_loader
    )

    # log
    if accelerator.is_local_main_process:
        logging.info(f"Train loader length: {len_train_loader}")
        logging.info(f"Val loader length: {len_val_loader}")

    # Setup EMA model if enabled
    model_ema = None
    if training_args.use_ema:
        model_ema = ModelEma(
            model,
            decay=training_args.ema_decay,
            device='cpu' if training_args.ema_force_cpu else None
        )
        if accelerator.is_local_main_process:
            logging.info(f"Using EMA with decay = {training_args.ema_decay}")

    # Setup checkpoint saver if enabled
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
    model, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, lr_scheduler
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
        # Training loop
        logging.info("Starting training...")

    best_metric = None
    best_epoch = None
    eval_metrics = {'loss': float('inf'), 'top1': 0.0, 'top5': 0.0}

    for epoch in range(training_args.num_epochs):
        # Train for one epoch
        train_metrics = train_one_epoch(
            epoch=epoch,
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            training_args=training_args,
            lr_scheduler=lr_scheduler,
            accelerator=accelerator,
            model_ema=model_ema,
            mixup_fn=mixup_fn
        )

        # Log current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        if accelerator.is_local_main_process:
            logging.info(f"Epoch {epoch+1}/{training_args.num_epochs} completed. LR: {current_lr:.3e}")

        # Evaluate on validation set
        if epoch % training_args.eval_freq == 0 or epoch == training_args.num_epochs - 1:
            eval_metrics = validate(
                model=model,
                loader=val_loader,
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
                    f"Top-5: {eval_metrics['top5']:.2f}%"
                )

            # Evaluate EMA model if available
            if model_ema is not None:
                ema_eval_metrics = validate(
                    model=model_ema.ema,
                    loader=val_loader,
                    device=device,
                    training_args=training_args,
                )
                if accelerator.is_local_main_process:
                    logging.info(
                        f"EMA Validation: "
                        f"Loss: {ema_eval_metrics['loss']:.4f}, "
                        f"Top-1: {ema_eval_metrics['top1']:.2f}%, "
                        f"Top-5: {ema_eval_metrics['top5']:.2f}%"
                    )

                # Use EMA metrics for tracking best model if better
                if ema_eval_metrics['top1'] > eval_metrics['top1']:
                    eval_metrics = ema_eval_metrics
                    if accelerator.is_local_main_process:
                        logging.info("Using EMA metrics as they're better")

            # Track best metric
            eval_metric = eval_metrics['top1']
            if best_metric is None or eval_metric > best_metric:
                best_metric = eval_metric
                best_epoch = epoch

                # log best metric in a nice format
                if accelerator.is_local_main_process:
                    logging.info(f"\nNew best metric: {best_metric:.2f}% (epoch {best_epoch+1})")

                # Save best model
                if training_args.save_checkpoint:
                    save_path = os.path.join(training_args.output_dir, 'best_model_hf')
                    accelerator.unwrap_model(model).save_pretrained(save_path)
                    if accelerator.is_local_main_process:
                        logging.info(f"Saved best model to {save_path}")

        # Save checkpoint if needed
        if training_args.save_checkpoint and epoch % training_args.save_freq == 0:
            # Use saver to keep track of top models
            save_metric = eval_metrics.get(training_args.eval_metric, eval_metrics['top1'])
            saver.save_checkpoint(epoch, metric=save_metric)

    # Calculate total training time
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    # Log final results - only on main process
    if accelerator.is_local_main_process:
        # log best eval
        logging.info(f"Best metric: {best_metric:.2f}% (epoch {best_epoch+1})")
        # Log final results
        logging.info(f"Training completed in {int(hours)}h {int(minutes)}m {seconds:.2f}s")
        logging.info(f"Best metric: {best_metric:.2f}% (epoch {best_epoch+1})")

    if accelerator.is_local_main_process:
        logging.info(f"Model saved to {training_args.output_dir}")
    
    # load best model and test on test loader
    if training_args.test_best:
        save_path = os.path.join(training_args.output_dir, 'best_model_hf')
        model = DeltaNetForVideoClassification.from_pretrained(save_path)
        model.to(device)
        test_metrics = validate(
            model=model,
            loader=test_loader,
            device=device,
            training_args=training_args,
            accelerator=accelerator,
        )

if __name__ == "__main__":
    main()
