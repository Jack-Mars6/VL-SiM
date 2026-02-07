import argparse
import copy
from typing import Callable, Tuple

import numpy as np
import torch
from sympy.physics.units import volume
from torch import nn
from transformers import AutoModel, AutoTokenizer
from nltk.tokenize import wordpunct_tokenize
from timm.models.vision_transformer import Block
import torchvision
import models_mae
import torch.nn.functional as F
import os


def get_args_parser():
    parser = argparse.ArgumentParser('CTLM pre-training', add_help=False)
    parser.add_argument('--batch_size', default=6, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--warmup_epochs', type=int, default=0, help='epochs to warmup LR')

    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=None, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=1e-5, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=1e-7, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')


    # Dataset parameters
    parser.add_argument('--data_train', default=["/home3/CT_RATE_1_10000/", "/home3/CT_RATE_10001_20000/"], type=str, nargs='+',
                        help='dataset path')
    parser.add_argument('--reports_train', default='/home2/Reports_label/train_reports.csv', type=str, nargs='+',
                        help='dataset path')
    parser.add_argument('--labels_train', default='/home2/Reports_label/train_predicted_labels.csv', type=str,
                        nargs='+', help='dataset path')
    parser.add_argument('--data_val', default=['/home2/valid_fixed/'], type=str, nargs='+',
                        help='dataset path')
    parser.add_argument('--reports_val', default='/home2/Reports_label/validation_reports.csv', type=str, nargs='+',
                        help='dataset path')
    parser.add_argument('--labels_val', default='/home2/Reports_label/valid_predicted_labels.csv', type=str, nargs='+',
                        help='dataset path')

    # frozen按轮warmup，frozen1按迭代warmup，frozen2不warmup1e-5
    parser.add_argument('--output_dir', default='./output_dir_pre',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir_pre',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default=False, type=bool, help='resume training')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--resume_multimodal', default="",
                        help='resume from checkpoint')


    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    # parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    return parser
args = get_args_parser().parse_args()


def init_weights_vit_timm(module: nn.Module, name: str = ""):
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    if isinstance(module, nn.LayerNorm):
        module.reset_parameters()
    if isinstance(module, nn.Conv2d):
        module.reset_parameters()


def named_apply(fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(
            fn=fn,
            module=child_module,
            name=child_name,
            depth_first=depth_first,
            include_root=True,
        )
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


class ImageEncoder(nn.Module):
    def __init__(self, input_dim=768, num_blocks=2):
        super().__init__()

        # define the model
        self.image_encoder = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)
        self.num_blocks = num_blocks
        self.input_dim = input_dim

        checkpoint = torch.load("", map_location='cpu', weights_only=False)
        self.image_encoder.load_state_dict(checkpoint['model'])
        print('load image_encoder checkpoint')

        for param in self.image_encoder.parameters():
            param.requires_grad = True

        self.ln_final = nn.LayerNorm(self.input_dim)
        self.slice_pooling = nn.Linear(160, 1)

    def init_weights(self):
        if self.num_blocks > 0:
            for i in range(self.num_blocks):
                block = self.slice_transformer[i]
                named_apply(init_weights_vit_timm, block)
            self.ln_final.reset_parameters()
        if isinstance(self.slice_pooling, nn.Linear):
            nn.init.normal_(self.slice_pooling.weight, std=self.slice_pooling.in_features ** -0.5)

    def forward(self, image):
        batch_size, c, d, h, w = image.shape
        slices = image.view(batch_size * d, c, h, w)

        if slices.shape[1] == 1:
            slices = slices.repeat(1, 3, 1, 1)

        slice_features = self.image_encoder.forward_features(slices)
        transformed_features = self.ln_final(slice_features)

        transformed_features = transformed_features.view(batch_size, d, slice_features.shape[1], slice_features.shape[2])
        features_permuted = transformed_features.permute(0, 2, 3, 1)
        compressed_features = self.slice_pooling(features_permuted)
        final_features = compressed_features.squeeze(-1)
        return final_features


def sanitize_report(report):
    report = report.lower()
    report_standard =  " ".join(wordpunct_tokenize(report))
    return report_standard


class TextEncoder(nn.Module):
    def __init__(self, pooling_strategy: str = "cls"):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained('/home2/Clinical_ModernBERT')
        self.text_encoder = AutoModel.from_pretrained('/home2/Clinical_ModernBERT')
        self.linear_layer = nn.Linear(768, 512)
        self.pooling_strategy = pooling_strategy

    def forward(self, text_labels):
        text_labels = list(text_labels)
        # text_labels = [sanitize_report(text) for text in text_labels]
        inputs = self.tokenizer(
            text_labels,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(self.text_encoder.device) for k, v in inputs.items()}
        text_embeddings = self.text_encoder(**inputs).last_hidden_state

        if self.pooling_strategy == "cls":
            pooled_embeddings = text_embeddings[:, 0, :]

        elif self.pooling_strategy == "mean":
            attention_mask = inputs["attention_mask"]
            pooled_embeddings = (torch.sum(text_embeddings * attention_mask.unsqueeze(-1), dim=1) /
                                 torch.sum(attention_mask, dim=1, keepdim=True))
        else:
            pooled_embeddings = text_embeddings.sum(dim=1)

        projected_embeddings = pooled_embeddings
        return projected_embeddings


class CTLMArchitecture(nn.Module):
    def __init__(self, align: bool = False, input_dim=768, embed_dim=512,
                 init_logit_scale_align=np.log(1 / 0.07), pooling_strategy: str = "mean"):
        super().__init__()
        self.align = align
        self.pooling_strategy = pooling_strategy
        self.encode_image = ImageEncoder()
        self.encode_text = TextEncoder()
        self.logit_scale_align = nn.Parameter(torch.ones([]) * init_logit_scale_align)
        if self.align:
            self.fc_image_text = nn.Linear(input_dim * 2, embed_dim)
        else:
            self.fc_image = nn.Linear(input_dim, embed_dim)
        self.fc_text = nn.Linear(input_dim, embed_dim)
        self.contrastive_loss = ContrastiveLoss()
        self.logit_scale = self.contrastive_loss.logit_scale


    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        if self.align:
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)
            # cosine similarity
            batch_size, num_patches, _ = image_features.shape
            image_features_flat = image_features[:, 1:, :]
            text_features_expanded = text_features.unsqueeze(1).expand(-1, num_patches-1, -1)
            cosine_sim = (image_features_flat * text_features_expanded).sum(dim=-1)
            scaled_similarity = cosine_sim * self.logit_scale_align.exp()

            # Softmax
            attention_weights = torch.softmax(scaled_similarity, dim=-1)
            weighted_image_features = torch.sum(attention_weights.unsqueeze(-1) * image_features_flat,dim=1)

            cls_token = image_features[:, 0, :]
            combined_features = torch.cat((weighted_image_features, cls_token), dim=-1)

            image_features_embed = self.fc_image_text(combined_features)
            text_features_embed = self.fc_text(text_features)

        else:
            if self.pooling_strategy == "cls":
                image_features = image_features[:, 0, :]
            elif self.pooling_strategy == "mean":
                image_features = torch.mean(image_features, dim=1)
            image_features_embed = self.fc_image(image_features)
            text_features_embed = self.fc_text(text_features)

        image_features_embed = F.normalize(image_features_embed, dim=-1)
        text_features_embed = F.normalize(text_features_embed, dim=-1)
        return image_features_embed, text_features_embed


class ContrastiveLoss(nn.Module):
    def __init__(self, init_logit_scale=np.log(1 / 0.07)):
        super().__init__()

        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)

    def forward(self, image_features, text_features):
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        logits_per_image = torch.matmul(image_features, text_features.t()) * self.logit_scale.exp()
        logits_per_text = torch.matmul(text_features, image_features.t()) * self.logit_scale.exp()
        batch_size = image_features.shape[0]
        labels = torch.arange(batch_size, device=image_features.device)

        loss_image = F.cross_entropy(logits_per_image, labels)
        loss_text = F.cross_entropy(logits_per_text, labels)
        loss = (loss_image + loss_text) / 2

        return loss
