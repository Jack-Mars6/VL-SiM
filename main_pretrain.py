# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json

import math
import numpy as np
import os
import time
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
# import timm
# assert timm.__version__ == "0.3.2"  # version check
# import timm.optim.optim_factory as optim_factory
import torch.nn.functional as F
import util.misc as misc
from util.ctlm_dataset_monai import CTPersistentDataset
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from build import CTLMArchitecture, ContrastiveLoss
from build import get_args_parser
import os
from torch.cuda.amp import autocast, GradScaler


def main(args):
    misc.init_distributed_mode(args)
    print(args.distributed)
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    start_epoch = args.start_epoch

    dataset_train = CTPersistentDataset(
        data_folder=args.data_train,
        csv_file=args.reports_train,
        cache_dir="/home2/CT_data/CT_RATE_cache",  # 缓存目录
    )
    print("Dataset len: %.2f" % len(dataset_train))

    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        global_rank = 0
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # define the model
    contrastive_loss = ContrastiveLoss()
    model = CTLMArchitecture(align=True)
    model.to(device)

    total_iterations = args.epochs * len(data_loader_train)
    warmup_iterations = args.warmup_epochs * len(data_loader_train)

    if args.resume:
        checkpoint_path = args.resume_multimodal
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model'], strict=False)
        start_epoch = checkpoint.get('epoch', 0) + 1
        completed_iterations = start_epoch * len(data_loader_train)
        print("Resume training, start epoch: %d" % start_epoch)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params CTLMArchitecture: %.2f' % n_parameters)

    model_without_ddp = model
    eff_batch_size = args.batch_size * misc.get_world_size()

    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], broadcast_buffers=False, find_unused_parameters=True)
        model_without_ddp = model.module

    optimizer = torch.optim.AdamW(
        model_without_ddp.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay
    )
    if args.resume:
        for param_group in optimizer.param_groups:
            param_group['initial_lr'] = args.lr

    if args.warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-2,
            end_factor=1.0,
            total_iters=warmup_iterations
        )
    else:
        warmup_scheduler = None

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_iterations - warmup_iterations,
        last_epoch=completed_iterations - 1 if args.resume else -1,
        eta_min=args.min_lr
    )
    if warmup_scheduler is not None:
        from torch.optim.lr_scheduler import SequentialLR
        lr_scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, lr_scheduler],
            milestones=[warmup_iterations]
        )
    print(optimizer)

    loss_scaler = GradScaler()
    print("Using AMP mixed precision training (FP16)")
    print(f"Start training for {args.epochs} epochs with {args.warmup_epochs} epochs warmup")

    start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        # 训练过程
        model.train()
        total_loss = 0
        num_batches = 0

        for batch_idx, (images, texts) in enumerate(data_loader_train):
            images = images.to(device)

            with autocast():
                image_features, text_features = model(images, texts)
                loss = contrastive_loss(image_features, text_features)

            optimizer.zero_grad()
            loss_scaler.scale(loss).backward()

            loss_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            loss_scaler.step(optimizer)
            loss_scaler.update()
            lr_scheduler.step()

            if (batch_idx + 1) % 20 == 0:
                print(f'{time.time()}, '
                      f'Epoch: {epoch}/{args.epochs-1}, Batch: {batch_idx+1}/{len(data_loader_train)}, '
                      f'Loss: {loss.item():.6f}, '
                      f'LR: {optimizer.param_groups[0]["lr"]:.3e}'
                      f"Logit scale: {model.module.logit_scale.item():.3f}")

            total_loss += loss.item()
            num_batches += 1

        # 计算平均损失
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        train_stats = {'loss': avg_loss, 'lr': optimizer.param_groups[0]['lr']}

        if args.output_dir and ((epoch+1) % 1 == 0 or epoch + 1 == args.epochs):
        # if args.output_dir and (epoch + 1 > 10 or epoch + 1 == args.epochs):
            misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp,
                            optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch, }
        if args.output_dir and misc.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
