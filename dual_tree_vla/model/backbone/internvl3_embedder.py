"""
InternVL3Embedder — 视觉-语言融合嵌入器

将相机图像 + 自然语言指令联合编码为特征向量，供双树动作策略使用。
本文件从 Evo-1 移植，去掉所有对外部 Evo-1 包的依赖，
只依赖标准库（torch / torchvision / transformers / Pillow）。
"""
from __future__ import annotations

import math
import logging
from typing import List, Union

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode, to_pil_image
from transformers import AutoModel, AutoTokenizer
from PIL import Image

# ── ImageNet 标准化参数（与 InternVL3 预训练保持一致）────────────────────
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def build_transform(input_size: int) -> T.Compose:
    """图像预处理管道：RGB 转换 → Resize → Tensor → ImageNet 归一化"""
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_ar = ratio[0] / ratio[1]
        diff = abs(aspect_ratio - target_ar)
        if diff < best_ratio_diff:
            best_ratio_diff = diff
            best_ratio = ratio
        elif diff == best_ratio_diff and area > 0.5 * image_size ** 2 * ratio[0] * ratio[1]:
            best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = 1,
    image_size: int = 448,
    use_thumbnail: bool = False,
) -> List[Image.Image]:
    """
    动态图像切片：按宽高比将图像切为若干 image_size×image_size 的块。
    默认 max_num=1（不切片，直接 resize），适合机器人低延迟场景。
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width  = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width  // image_size)) * image_size,
            (i // (target_width  // image_size)) * image_size,
            ((i % (target_width  // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


class InternVL3Embedder(nn.Module):
    """
    InternVL3 视觉-语言融合嵌入器。

    主要改造（与原始 InternVL3 不同）：
      - 语言模型截断至前 14 层（降低计算量）
      - lm_head 替换为 Identity（直接输出隐藏态）
      - 关闭 ViT 梯度检查点（加速推理）

    Parameters
    ----------
    model_name   : HuggingFace model ID 或本地路径（如 "model_weights/Qwen2.5-0.5B"）
    image_size   : 输入图像分辨率（边长），默认 448
    device       : 推理设备，"cuda" 或 "cpu"
    llm_layers   : 保留的 LLM 层数，默认 14
    """

    def __init__(
        self,
        model_name: str,
        image_size: int = 448,
        device: str = "cuda",
        llm_layers: int = 14,
    ):
        super().__init__()
        self.device = device
        self.image_size = image_size
        self.max_text_length = 1024
        self.transform = build_transform(image_size)
        use_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
        model_dtype = torch.bfloat16 if use_cuda else torch.float32

        # ── Tokenizer ──────────────────────────────────────────────────────
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, use_fast=False
        )

        # ── InternVL3 主模型 ───────────────────────────────────────────────
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=model_dtype,
            trust_remote_code=True,
            use_flash_attn=use_cuda,
            low_cpu_mem_usage=use_cuda,
            _fast_init=False,
        ).to(self.device)

        # ── 截断 LLM 层数 ────────────────────────────────────────────────
        lm = self.model.language_model
        if hasattr(lm, "model"):
            lm.model.layers = torch.nn.ModuleList(list(lm.model.layers)[:llm_layers])
        else:
            lm.layers = torch.nn.ModuleList(list(lm.layers)[:llm_layers])

        # lm_head 替换为 Identity，直接透传隐藏态
        lm.lm_head = torch.nn.Identity()

        # 关闭 ViT 梯度检查点
        if hasattr(self.model, "vision_model") and hasattr(self.model.vision_model, "encoder"):
            self.model.vision_model.encoder.gradient_checkpointing = False

    # ---------------------------------------------------------------- #
    #  图像预处理                                                        #
    # ---------------------------------------------------------------- #

    def _preprocess_images(
        self,
        image_tensors: List[Union[Image.Image, torch.Tensor]],
    ):
        """
        Returns
        -------
        pixel_values  : (总切片数, 3, image_size, image_size) bfloat16
        num_tiles_list: 每张图对应的切片数
        """
        pixel_values_list = []
        for img in image_tensors:
            if img is None:
                raise ValueError("[InternVL3Embedder] image_tensors 中含 None，"
                                 "请检查 feat_cache_dir 配置和缓存提取脚本。")
            if isinstance(img, torch.Tensor):
                img = img.cpu()
                # 处理多余维度：(B,C,H,W) 或 (T,C,H,W) → 取第 0 帧
                while img.ndim > 3:
                    img = img[0]
                # (1,H,W) → 兼容，(3,H,W) 正常
                img = to_pil_image(img)
            tiles = dynamic_preprocess(img, image_size=self.image_size)
            pixel_values_list.append(
                torch.stack([self.transform(t) for t in tiles])
            )
        pixel_values = torch.cat(pixel_values_list, dim=0).to(
            dtype=torch.bfloat16, device=self.device
        )
        num_tiles_list = [pv.shape[0] for pv in pixel_values_list]
        return pixel_values, num_tiles_list

    # ---------------------------------------------------------------- #
    #  多模态 Prompt 构建                                                #
    # ---------------------------------------------------------------- #

    def _build_multimodal_prompt(
        self,
        num_tiles_list: List[int],
        text_prompt: str,
    ) -> str:
        """
        构建带图像占位符的 Prompt 字符串。
        格式：Image-1: <img><IMG_CONTEXT>×256</img>\\n...{text_prompt}
        """
        IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
        IMG_START_TOKEN   = "<img>"
        IMG_END_TOKEN     = "</img>"
        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

        prompt = ""
        for i in range(len(num_tiles_list)):
            prompt += f"Image-{i+1}: <image>\n"
        prompt += text_prompt.strip()

        for tile_count in num_tiles_list:
            token_count = self.model.num_image_token * tile_count
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * token_count + IMG_END_TOKEN
            prompt = prompt.replace("<image>", image_tokens, 1)
        return prompt

    # ---------------------------------------------------------------- #
    #  Embedding 注入                                                    #
    # ---------------------------------------------------------------- #

    def _prepare_and_fuse_embeddings(
        self,
        prompt: str,
        vit_embeds: torch.Tensor,
        image_mask: torch.Tensor,
        num_tiles_list: List[int],
    ):
        """
        将 ViT 特征注入 LLM embedding 序列。

        Returns
        -------
        input_embeds  : (1, max_text_length, hidden_dim)
        attention_mask: (1, max_text_length)
        """
        untruncated_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        true_sequence_length = untruncated_ids.shape[1]

        if true_sequence_length > self.max_text_length:
            print("\n" + "=" * 80)
            print(" WARNING: Input prompt was TRUNCATED!")
            print(f"   - Max Length Allowed    : {self.max_text_length}")
            print(f"   - Actual Length      : {true_sequence_length}")
            print(f"   - Truncated Prompt (first 100 chars): '{prompt[:100]}...'")
            print("=" * 80 + "\n")

        model_inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_text_length,
        ).to(self.device)
        input_ids      = model_inputs["input_ids"]       # (1, L)
        attention_mask = model_inputs["attention_mask"]  # (1, L)

        img_token_mask      = (input_ids == self.img_context_token_id)
        img_token_locations = torch.where(img_token_mask)[1]

        input_embeds = self.model.language_model.get_input_embeddings()(input_ids).clone()
        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)
        input_ids    = input_ids.reshape(B * N)
        selected     = (input_ids == self.img_context_token_id)

        try:
            input_embeds[selected] = (
                input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
            )
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            logging.warning(
                f"InternVL3Embedder: vit_embeds size mismatch ({e}), truncating."
            )
            n_token = selected.sum()
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]

        tokens_per_tile   = self.model.num_image_token
        current_token_idx = 0
        for i in range(len(image_mask)):
            n_tiles  = num_tiles_list[i]
            n_tokens = n_tiles * tokens_per_tile
            if not image_mask[i]:
                start = img_token_locations[current_token_idx]
                attention_mask[0, start : start + n_tokens] = 0
            current_token_idx += n_tokens

        input_embeds = input_embeds.reshape(B, N, C)
        return input_embeds, attention_mask

    # ---------------------------------------------------------------- #
    #  主入口                                                            #
    # ---------------------------------------------------------------- #

    def get_fused_image_text_embedding_from_tensor_images(
        self,
        image_tensors: List[Union[Image.Image, torch.Tensor]],
        image_mask: torch.Tensor,
        text_prompt: str,
        return_cls_only: bool = True,
    ) -> torch.Tensor:
        """
        图像 + 指令 → 融合特征向量。

        Returns
        -------
        return_cls_only=True : (1, hidden_dim)
        return_cls_only=False: (1, max_text_length, hidden_dim)
        """
        pixel_values, num_tiles_list = self._preprocess_images(image_tensors)
        if pixel_values.shape[0] == 0:
            print("Warning: No valid images to process after masking.")
        vit_embeds   = self.model.extract_feature(pixel_values)
        prompt       = self._build_multimodal_prompt(num_tiles_list, text_prompt)
        inputs_embeds, attn_mask = self._prepare_and_fuse_embeddings(
            prompt, vit_embeds, image_mask, num_tiles_list
        )
        outputs = self.model.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        fused_hidden = outputs.hidden_states[-1].to(torch.float32)
        return fused_hidden[:, 0, :] if return_cls_only else fused_hidden
