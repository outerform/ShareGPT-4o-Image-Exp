import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import gradio as gr
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

import numpy as np
import time

from attention_hooks import JanusAttentionHook


# Load model and processor
model_path = "/home/outerform/models/Janus-4o-7B"
config = AutoConfig.from_pretrained(model_path)
language_config = config.language_config
language_config._attn_implementation = 'eager'
vl_gpt = AutoModelForCausalLM.from_pretrained(model_path,
                                             language_config=language_config,
                                             trust_remote_code=True)
if torch.cuda.is_available():
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda()
else:
    vl_gpt = vl_gpt.to(torch.float16)

vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer
cuda_device = 'cuda' if torch.cuda.is_available() else 'cpu'

attention_hook = JanusAttentionHook()
attention_hook.register_hooks(vl_gpt)

@torch.inference_mode()
# @spaces.GPU(duration=120) 
# Multimodal Understanding function
def multimodal_understanding(image, question, seed, top_p, temperature):
    # Clear CUDA cache before generating
    torch.cuda.empty_cache()
    
    # set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    
    conversation = [
        {
            "role": "<|User|>",
            "content": f"<image_placeholder>\n{question}",
            "images": [image],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]
    
    pil_images = [Image.fromarray(image)]
    prepare_inputs = vl_chat_processor(
        conversations=conversation, images=pil_images, force_batchify=True
    ).to(cuda_device, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16)
    
    
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
    
    outputs = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=512,
        do_sample=False if temperature == 0 else True,
        use_cache=True,
        temperature=temperature,
        top_p=top_p,
    )
    
    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    return answer


def generate(input_ids,
             width,
             height,
             temperature: float = 1,
             parallel_size: int = 5,
             cfg_weight: float = 5,
             image_token_num_per_image: int = 576,
             patch_size: int = 16):
    # Clear CUDA cache before generating
    torch.cuda.empty_cache()
    
    tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.int).to(cuda_device)
    for i in range(parallel_size * 2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id
    inputs_embeds = vl_gpt.language_model.get_input_embeddings()(tokens)
    generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).to(cuda_device)

    pkv = None
    for i in range(image_token_num_per_image):
        with torch.no_grad():
            outputs = vl_gpt.language_model.model(inputs_embeds=inputs_embeds,
                                                use_cache=True,
                                                past_key_values=pkv)
            pkv = outputs.past_key_values
            hidden_states = outputs.last_hidden_state
            logits = vl_gpt.gen_head(hidden_states[:, -1, :])
            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]
            logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens[:, i] = next_token.squeeze(dim=-1)
            next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)

            img_embeds = vl_gpt.prepare_gen_img_embeds(next_token)
            inputs_embeds = img_embeds.unsqueeze(dim=1)

    

    patches = vl_gpt.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int),
                                                 shape=[parallel_size, 8, width // patch_size, height // patch_size])

    return generated_tokens.to(dtype=torch.int), patches

def unpack(dec, width, height, parallel_size=5):
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255)

    visual_img = np.zeros((parallel_size, width, height, 3), dtype=np.uint8)
    visual_img[:, :, :] = dec

    return visual_img



@torch.inference_mode()
# @spaces.GPU(duration=120)  # Specify a duration to avoid timeout
def generate_image(prompt,
                   seed=None,
                   guidance=5,
                   t2i_temperature=1.0):
    # Clear CUDA cache and avoid tracking gradients
    torch.cuda.empty_cache()
    # Set the seed for reproducible results
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
    width = 384
    height = 384
    parallel_size = 5
    
    with torch.no_grad():
        messages = [{'role': '<|User|>', 'content': prompt},
                    {'role': '<|Assistant|>', 'content': ''}]
        text = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(conversations=messages,
                                                                   sft_format=vl_chat_processor.sft_format,
                                                                   system_prompt='')
        text = text + vl_chat_processor.image_start_tag
        
        input_ids = torch.LongTensor(tokenizer.encode(text))
        output, patches = generate(input_ids,
                                   width // 16 * 16,
                                   height // 16 * 16,
                                   cfg_weight=guidance,
                                   parallel_size=parallel_size,
                                   temperature=t2i_temperature)
        images = unpack(patches,
                        width // 16 * 16,
                        height // 16 * 16,
                        parallel_size=parallel_size)

        return [Image.fromarray(images[i]).resize((768, 768), Image.LANCZOS) for i in range(parallel_size)]
        

# NOTE: MOD CZY BEG
from dataclasses import dataclass

@dataclass
class VLChatProcessorOutput():
    sft_format: str
    input_ids: torch.Tensor
    pixel_values: torch.Tensor
    num_image_tokens: torch.IntTensor

    def __len__(self):
        return len(self.input_ids)

def process_image(pil_images, vl_chat_processor):
    images = pil_images
    images_outputs = vl_chat_processor.image_processor(images, return_tensors="pt")
    return images_outputs['pixel_values']

def generate_image_v2v_mask_v3(question, input_image, temperature=1, parallel_size=1,
                               cfg_weight=5, cfg_weight2=5, return_attention=False):
    torch.cuda.empty_cache()

    input_img_tokens = (vl_chat_processor.image_start_tag
                        + vl_chat_processor.image_tag * vl_chat_processor.num_image_tokens
                        + vl_chat_processor.image_end_tag
                        + vl_chat_processor.image_start_tag
                        + vl_chat_processor.pad_tag * vl_chat_processor.num_image_tokens
                        + vl_chat_processor.image_end_tag)
    output_img_tokens = vl_chat_processor.image_start_tag

    pre_data = []
    input_image = Image.fromarray(input_image)
    input_images = [input_image]
    img_len = len(input_images)
    prompts = input_img_tokens * img_len + question
    conversation = [
        {"role": "<|User|>", "content": prompts},
        {"role": "<|Assistant|>", "content": ""}
    ]
    sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=vl_chat_processor.sft_format,
        system_prompt="",
    )
    sft_format = sft_format + output_img_tokens

    mmgpt = vl_gpt
    image_token_num_per_image = 576
    img_size = 384
    patch_size = 16

    with torch.inference_mode():
        input_image_pixel_values = process_image(input_images, vl_chat_processor).to(torch.bfloat16).cuda()
        quant_input, emb_loss_input, info_input = mmgpt.gen_vision_model.encode(input_image_pixel_values)
        image_tokens_input = info_input[2].detach().reshape(input_image_pixel_values.shape[0], -1)
        image_embeds_input = mmgpt.prepare_gen_img_embeds(image_tokens_input)

        input_ids = torch.LongTensor(vl_chat_processor.tokenizer.encode(sft_format))

        # --- Compute token boundaries for attention hook ---
        input_ids_list = input_ids.tolist()
        image_start_id = vl_chat_processor.image_start_id
        image_end_id = vl_chat_processor.image_end_id
        pad_id = vl_chat_processor.pad_id

        # Find all <img_end> positions to locate the VQ image token block
        img_end_positions = [i for i, t in enumerate(input_ids_list) if t == image_end_id]
        # The second <img_end> marks end of VQ image block (pad_tag block)
        # VQ block: from first <img_end>+2 (skip <img_end><img_start>) to second <img_end>
        if len(img_end_positions) >= 2:
            vq_block_start = img_end_positions[0] + 2  # skip <img_end> <img_start>
            vq_block_end = img_end_positions[1]        # exclusive: the second <img_end>
        else:
            vq_block_start = 0
            vq_block_end = 0

        # CLIP encoder block: from first <img_start>+1 to first <img_end>
        img_start_positions = [i for i, t in enumerate(input_ids_list) if t == image_start_id]
        if len(img_start_positions) >= 1 and len(img_end_positions) >= 1:
            encoder_block_start = img_start_positions[0] + 1
            encoder_block_end = img_end_positions[0]
        else:
            encoder_block_start = 0
            encoder_block_end = 0

        # Text tokens: from second <img_end>+1 to the last <img_start> (output_img_tokens)
        if len(img_start_positions) >= 3 and len(img_end_positions) >= 2:
            text_start = img_end_positions[1] + 1
            text_end = img_start_positions[-1]  # exclusive: the output <img_start>
        else:
            text_start = 0
            text_end = 0

        # --- Build token info for UI (decode each text token) ---
        text_token_ids = input_ids_list[text_start:text_end]
        text_tokens_decoded = [tokenizer.decode([tid]) for tid in text_token_ids]

        # Identify which tokens come from the actual user prompt (filter template tokens)
        prompt_only_ids = tokenizer.encode(question, add_special_tokens=False)
        n_prompt_tokens = len(prompt_only_ids)
        prompt_token_indices = list(range(min(n_prompt_tokens, len(text_token_ids))))

        encoder_pixel_values = process_image(input_images, vl_chat_processor).cuda()
        tokens = torch.zeros((parallel_size * 3, len(input_ids)), dtype=torch.long)
        for i in range(parallel_size * 3):
            tokens[i, :] = input_ids
            if i % 3 == 2:
                tokens[i, 1:-1] = pad_id
                pre_data.append(VLChatProcessorOutput(
                    sft_format=sft_format, pixel_values=encoder_pixel_values,
                    input_ids=tokens[i-2],
                    num_image_tokens=[vl_chat_processor.num_image_tokens] * img_len))
                pre_data.append(VLChatProcessorOutput(
                    sft_format=sft_format, pixel_values=encoder_pixel_values,
                    input_ids=tokens[i-1],
                    num_image_tokens=[vl_chat_processor.num_image_tokens] * img_len))
                pre_data.append(VLChatProcessorOutput(
                    sft_format=sft_format, pixel_values=None,
                    input_ids=tokens[i], num_image_tokens=[]))

        prepare_inputs = vl_chat_processor.batchify(pre_data)

        inputs_embeds = mmgpt.prepare_inputs_embeds(
            input_ids=tokens.cuda(),
            pixel_values=prepare_inputs['pixel_values'].to(torch.bfloat16).cuda(),
            images_emb_mask=prepare_inputs['images_emb_mask'].cuda(),
            images_seq_mask=prepare_inputs['images_seq_mask'].cuda()
        )

        image_gen_indices = (tokens == image_end_id).nonzero()

        for ii, ind in enumerate(image_gen_indices):
            if ii % 4 == 0:
                offset = ind[1] + 2
                inputs_embeds[ind[0], offset:offset + image_embeds_input.shape[1], :] = \
                    image_embeds_input[(ii // 2) % img_len]

        generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).cuda()

        # --- Enable attention hook ---
        if return_attention:
            attention_hook.clear()
            attention_hook.enable()
            attention_hook.set_token_ranges(
                text_range=(text_start, text_end),
                input_image_range=(vq_block_start, vq_block_end),
                encoder_image_range=(encoder_block_start, encoder_block_end),
                stream0_idx=0,
            )
            attention_hook.set_phase("prefill")

        for i in range(image_token_num_per_image):
            outputs = mmgpt.language_model.model(
                inputs_embeds=inputs_embeds, use_cache=True,
                past_key_values=outputs.past_key_values if i != 0 else None)
            hidden_states = outputs.last_hidden_state

            # Switch to generation phase after prefill
            if return_attention and i == 0:
                attention_hook.set_phase("generate")

            logits = mmgpt.gen_head(hidden_states[:, -1, :])
            logit_cond_full = logits[0::3, :]
            logit_cond_part = logits[1::3, :]
            logit_uncond = logits[2::3, :]

            logit_cond = (logit_cond_full + cfg_weight2 * logit_cond_part) / (1 + cfg_weight2)
            logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)

            probs = torch.softmax(logits / temperature, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens[:, i] = next_token.squeeze(dim=-1)

            next_token = torch.cat([
                next_token.unsqueeze(dim=1),
                next_token.unsqueeze(dim=1),
                next_token.unsqueeze(dim=1)
            ], dim=1).view(-1)
            img_embeds = mmgpt.prepare_gen_img_embeds(next_token)
            inputs_embeds = img_embeds.unsqueeze(dim=1)

        if return_attention:
            attention_hook.disable()

        dec = mmgpt.gen_vision_model.decode_code(
            generated_tokens.to(dtype=torch.int),
            shape=[parallel_size, 8, img_size // patch_size, img_size // patch_size])
        dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
        dec = np.clip((dec + 1) / 2 * 255, 0, 255)

        visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
        visual_img[:, :, :] = dec
        images = [Image.fromarray(visual_img[i]) for i in range(parallel_size)]

        if return_attention:
            attn_maps = {
                "tokens": text_tokens_decoded,
                "token_ids": text_token_ids,
                "prompt_token_indices": prompt_token_indices,
                "text_range": (text_start, text_end),
                "input_image_range": (vq_block_start, vq_block_end),
                "encoder_image_range": (encoder_block_start, encoder_block_end),
                "image_patch_size": img_size // patch_size,  # 24
                "num_layers": attention_hook.num_layers,
                "layer_indices": attention_hook.get_layer_indices(),
                "text_to_encoder_overall": attention_hook.get_text_to_encoder_image_attention(layer_idx=None),
                "text_to_image_overall": attention_hook.get_text_to_image_attention(layer_idx=None),
                "output_to_text_overall": attention_hook.get_output_to_text_attention(layer_idx=None),
            }
            for li in attention_hook.get_layer_indices():
                attn_maps[f"text_to_encoder_layer_{li}"] = attention_hook.get_text_to_encoder_image_attention(layer_idx=li)
                attn_maps[f"text_to_image_layer_{li}"] = attention_hook.get_text_to_image_attention(layer_idx=li)
                attn_maps[f"output_to_text_layer_{li}"] = attention_hook.get_output_to_text_attention(layer_idx=li)
            return images, attn_maps

        return images


def visualize_attention_map(base_image: Image.Image,
                            attention_weights,
                            spatial_shape: tuple,
                            alpha: float = 0.5) -> Image.Image:
    """Overlay an attention heatmap on a base image."""
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()

    if attention_weights.ndim == 1:
        H, W = spatial_shape
        total = H * W
        if len(attention_weights) < total:
            attention_weights = np.pad(attention_weights, (0, total - len(attention_weights)))
        attention_weights = attention_weights[:total].reshape(H, W)

    amin, amax = attention_weights.min(), attention_weights.max()
    if amax - amin > 1e-8:
        attention_weights = (attention_weights - amin) / (amax - amin)
    else:
        attention_weights = np.zeros_like(attention_weights)

    # Adaptive contrast enhancement
    for mult in range(1, 1001):
        if (attention_weights * mult > 0.5).sum() / attention_weights.size > 0.1:
            break
    attention_weights = np.clip(attention_weights * mult, 0, 1)

    from scipy.ndimage import zoom
    img_array = np.array(base_image.convert("RGB"))
    img_h, img_w = img_array.shape[:2]
    scale_h = img_h / attention_weights.shape[0]
    scale_w = img_w / attention_weights.shape[1]
    attention_resized = zoom(attention_weights, (scale_h, scale_w), order=1)

    cmap = plt.get_cmap('jet')
    norm = Normalize(vmin=0, vmax=1)
    heatmap = cmap(norm(attention_resized))
    heatmap = (heatmap[:, :, :3] * 255).astype(np.uint8)

    img_float = img_array.astype(np.float32) / 255.0
    heatmap_float = heatmap.astype(np.float32) / 255.0
    overlay = img_float * (1 - alpha) + heatmap_float * alpha
    overlay = (overlay * 255).astype(np.uint8)
    return Image.fromarray(overlay)


def get_attention_visualizations(token_idx: int,
                                 attention_maps: dict,
                                 input_image,
                                 edited_image: Image.Image,
                                 layer_idx=None):
    """
    Produce three heatmap images for a selected text token:
      viz_enc: text -> CLIP encoder image  (overlaid on input image)
      viz_vq:  text -> VQ input image      (overlaid on input image)
      viz_out: output_image -> text         (overlaid on edited image)
    """
    if not attention_maps or "tokens" not in attention_maps:
        return None, None, None
    if token_idx < 0 or token_idx >= len(attention_maps["tokens"]):
        return None, None, None

    ps = attention_maps["image_patch_size"]  # 24
    spatial = (ps, ps)

    # --- Select attention source based on layer_idx ---
    if layer_idx is not None:
        t2e_attn = attention_maps.get(f"text_to_encoder_layer_{layer_idx}")
        t2i_attn = attention_maps.get(f"text_to_image_layer_{layer_idx}")
        o2t_attn = attention_maps.get(f"output_to_text_layer_{layer_idx}")
    else:
        t2e_attn = attention_maps.get("text_to_encoder_overall")
        t2i_attn = attention_maps.get("text_to_image_overall")
        o2t_attn = attention_maps.get("output_to_text_overall")

    if isinstance(input_image, np.ndarray):
        input_pil = Image.fromarray(input_image)
    else:
        input_pil = input_image

    viz_enc = None
    viz_vq = None
    viz_out = None

    # 1. text -> CLIP encoder image: shape (num_text, num_enc_img_tokens)
    if t2e_attn is not None and t2e_attn.ndim == 2:
        if token_idx < t2e_attn.shape[0]:
            attn_row = t2e_attn[token_idx, :]  # (num_enc_img_tokens,)
            viz_enc = visualize_attention_map(input_pil, attn_row, spatial)

    # 2. text -> VQ input image: shape (num_text, num_img_tokens)
    if t2i_attn is not None and t2i_attn.ndim == 2:
        if token_idx < t2i_attn.shape[0]:
            attn_row = t2i_attn[token_idx, :]  # (num_img_tokens,)
            viz_vq = visualize_attention_map(input_pil, attn_row, spatial)

    # 3. output_image -> text: shape (num_output_tokens, num_text)
    if o2t_attn is not None and o2t_attn.ndim == 2:
        if token_idx < o2t_attn.shape[1]:
            attn_col = o2t_attn[:, token_idx]  # (num_output_tokens,)
            viz_out = visualize_attention_map(edited_image, attn_col, spatial)

    return viz_enc, viz_vq, viz_out


def compute_region_attention(attention_maps, x1, y1, x2, y2, layer_idx=None):
    """
    Given a pixel-space bounding box on the output image (384x384),
    compute average attention from patches in that region to each text token.
    Only returns attention for prompt tokens (filters out template tokens).
    Returns: (numpy array of attention per prompt token, list of prompt token labels)
    """
    x1, x2 = min(int(x1), int(x2)), max(int(x1), int(x2))
    y1, y2 = min(int(y1), int(y2)), max(int(y1), int(y2))

    ps = attention_maps["image_patch_size"]  # 24
    img_size = 384
    patch_px = img_size // ps  # 16

    pr1, pc1 = y1 // patch_px, x1 // patch_px
    pr2 = min(y2 // patch_px + 1, ps)
    pc2 = min(x2 // patch_px + 1, ps)
    pr1, pc1 = max(pr1, 0), max(pc1, 0)

    indices = [r * ps + c for r in range(pr1, pr2) for c in range(pc1, pc2)]
    if not indices:
        return None, []

    if layer_idx is not None:
        o2t = attention_maps.get(f"output_to_text_layer_{layer_idx}")
    else:
        o2t = attention_maps.get("output_to_text_overall")

    if o2t is None:
        return None, []

    if isinstance(o2t, torch.Tensor):
        o2t = o2t.detach().cpu().float()
        region_attn_full = o2t[indices, :].mean(dim=0).numpy()
    else:
        region_attn_full = np.array(o2t)[indices, :].mean(axis=0)

    prompt_indices = attention_maps.get("prompt_token_indices")
    all_tokens = attention_maps["tokens"]
    if prompt_indices is not None:
        region_attn = region_attn_full[prompt_indices]
        labels = [all_tokens[i] for i in prompt_indices]
    else:
        region_attn = region_attn_full
        labels = all_tokens

    return region_attn, labels


def compute_input_region_attention(attention_maps, x1, y1, x2, y2,
                                   attn_type="encoder", layer_idx=None):
    """
    Given a pixel-space bounding box on the input image (384x384),
    compute how much each text token attends to the selected input region.
    attn_type: "encoder" for text→CLIP, "vq" for text→VQ.
    Returns: (numpy array of attention per prompt token, list of prompt token labels)
    """
    x1, x2 = min(int(x1), int(x2)), max(int(x1), int(x2))
    y1, y2 = min(int(y1), int(y2)), max(int(y1), int(y2))

    ps = attention_maps["image_patch_size"]  # 24
    img_size = 384
    patch_px = img_size // ps  # 16

    pr1, pc1 = y1 // patch_px, x1 // patch_px
    pr2 = min(y2 // patch_px + 1, ps)
    pc2 = min(x2 // patch_px + 1, ps)
    pr1, pc1 = max(pr1, 0), max(pc1, 0)

    indices = [r * ps + c for r in range(pr1, pr2) for c in range(pc1, pc2)]
    if not indices:
        return None, []

    if attn_type == "encoder":
        key_overall, key_layer = "text_to_encoder_overall", "text_to_encoder_layer_"
    else:
        key_overall, key_layer = "text_to_image_overall", "text_to_image_layer_"

    if layer_idx is not None:
        t2i = attention_maps.get(f"{key_layer}{layer_idx}")
    else:
        t2i = attention_maps.get(key_overall)

    if t2i is None:
        return None, []

    # t2i shape: (num_text, num_img_tokens) — select columns for the region, average
    if isinstance(t2i, torch.Tensor):
        t2i = t2i.detach().cpu().float()
        region_attn_full = t2i[:, indices].mean(dim=1).numpy()  # (num_text,)
    else:
        region_attn_full = np.array(t2i)[:, indices].mean(axis=1)

    prompt_indices = attention_maps.get("prompt_token_indices")
    all_tokens = attention_maps["tokens"]
    if prompt_indices is not None:
        region_attn = region_attn_full[prompt_indices]
        labels = [all_tokens[i] for i in prompt_indices]
    else:
        region_attn = region_attn_full
        labels = all_tokens

    return region_attn, labels


def render_region_attention_chart(attn_weights, token_labels, title="Region -> Text Token Attention"):
    """Render a horizontal bar chart of attention per text token. Returns PIL Image."""
    if attn_weights is None or len(token_labels) == 0:
        return None

    n = len(token_labels)
    fig, ax = plt.subplots(figsize=(8, max(3, n * 0.35)))
    y_pos = np.arange(n)
    labels = [f"{i}: {t}" for i, t in enumerate(token_labels)]

    ax.barh(y_pos, attn_weights, color='steelblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Attention Weight")
    ax.set_title(title)
    ax.invert_yaxis()
    fig.tight_layout()

    from io import BytesIO
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert('RGB')
    return img


def draw_bbox_preview(image, x1, y1, x2, y2):
    """Draw a red bounding box on the image and return the preview."""
    if image is None:
        return None
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    bx1, bx2 = min(int(x1), int(x2)), max(int(x1), int(x2))
    by1, by2 = min(int(y1), int(y2)), max(int(y1), int(y2))
    draw.rectangle([bx1, by1, bx2, by2], outline="red", width=3)
    return img


def text_and_image_to_image(prompt, image, seed=None, guidance1=5, guidance2=5, t2i_temperature=1.0):
    torch.cuda.empty_cache()

    if seed is not None:
        seed = int(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

    output_images = generate_image_v2v_mask_v3(
        question=prompt,
        input_image=image,
        temperature=t2i_temperature,
        parallel_size=5,
        cfg_weight=guidance1,
        cfg_weight2=guidance2
    )

    return output_images


# ======================================================================
# Gradio Interface
# ======================================================================

with gr.Blocks() as demo:

    # ------------------------------------------------------------------
    # Tab 1: Text-to-Image Generation (unchanged)
    # ------------------------------------------------------------------
    gr.Markdown(value="# Text-to-Image Generation")

    with gr.Row():
        cfg_weight_input = gr.Slider(minimum=1, maximum=10, value=5, step=0.5, label="CFG Weight")
        t2i_temperature = gr.Slider(minimum=0, maximum=1, value=1.0, step=0.05, label="temperature")

    prompt_input = gr.Textbox(label="Prompt. (Prompt in more detail can help produce better images!)")
    seed_input = gr.Number(label="Seed (Optional)", precision=0, value=12345)

    generation_button = gr.Button("Generate Images")

    image_output = gr.Gallery(label="Generated Images", columns=2, rows=2, height=300)

    generation_button.click(
        fn=generate_image,
        inputs=[prompt_input, seed_input, cfg_weight_input, t2i_temperature],
        outputs=image_output
    )

    # ------------------------------------------------------------------
    # Tab 2: Image Edit with Attention Visualization
    # ------------------------------------------------------------------
    gr.Markdown(value="# Image Edit with Attention Visualization")

    attention_state = gr.State(value=None)

    with gr.Row():
        with gr.Column(scale=1):
            edit_image_input = gr.Image(label="Input Image")
            edit_prompt = gr.Textbox(label="Text Prompt")
        with gr.Column(scale=1):
            edit_image_output = gr.Image(label="Edited Result", type="pil")

    with gr.Row():
        edit_cfg1 = gr.Slider(minimum=1, maximum=10, value=5, step=0.5, label="CFG Weight 1")
        edit_cfg2 = gr.Slider(minimum=1, maximum=10, value=5, step=0.5, label="CFG Weight 2")

    with gr.Row():
        edit_temperature = gr.Slider(minimum=0, maximum=1, value=1.0, step=0.05, label="Temperature")
        edit_seed = gr.Number(label="Seed (Optional)", precision=0, value=12345)

    edit_btn = gr.Button("Generate & Capture Attention", variant="primary")

    # --- Attention Visualization Section ---
    with gr.Accordion("Attention Visualization", open=False):
        with gr.Row():
            token_selector = gr.Radio(
                label="Select Text Token",
                choices=[],
                value=None,
                interactive=True
            )

        with gr.Row():
            avg_layers_checkbox = gr.Checkbox(
                label="Average all layers",
                value=True,
                info="Uncheck to view attention at a specific model layer"
            )
            layer_slider = gr.Slider(
                minimum=0, maximum=29, step=1, value=0,
                label="Layer",
                visible=False,
                interactive=True,
                info="Select model layer"
            )

        with gr.Row():
            with gr.Column():
                text_enc_viz = gr.Image(label="Text → Encoder Image Attention", type="pil")
            with gr.Column():
                text_vq_viz = gr.Image(label="Text → VQ Image Attention", type="pil")
            with gr.Column():
                out_text_viz = gr.Image(label="Output Image → Text Attention", type="pil")

        gr.Markdown("### Output Region → Text Attention")
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("Select region on output image (pixel coordinates, 384×384):")
                with gr.Row():
                    region_x1 = gr.Slider(minimum=0, maximum=383, value=0, step=1,
                                          label="x1 (left)", interactive=True)
                    region_y1 = gr.Slider(minimum=0, maximum=383, value=0, step=1,
                                          label="y1 (top)", interactive=True)
                with gr.Row():
                    region_x2 = gr.Slider(minimum=0, maximum=383, value=383, step=1,
                                          label="x2 (right)", interactive=True)
                    region_y2 = gr.Slider(minimum=0, maximum=383, value=383, step=1,
                                          label="y2 (bottom)", interactive=True)
                region_preview = gr.Image(label="Selected Region (Output)", type="pil")
            with gr.Column(scale=1):
                region_chart = gr.Image(label="Output Region → Text Attention", type="pil")

        gr.Markdown("### Input Region ↔ Text Attention")
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("Select region on input image (coordinates mapped to 384×384):")
                with gr.Row():
                    in_region_x1 = gr.Slider(minimum=0, maximum=383, value=0, step=1,
                                             label="x1 (left)", interactive=True)
                    in_region_y1 = gr.Slider(minimum=0, maximum=383, value=0, step=1,
                                             label="y1 (top)", interactive=True)
                with gr.Row():
                    in_region_x2 = gr.Slider(minimum=0, maximum=383, value=383, step=1,
                                             label="x2 (right)", interactive=True)
                    in_region_y2 = gr.Slider(minimum=0, maximum=383, value=383, step=1,
                                             label="y2 (bottom)", interactive=True)
                in_region_preview = gr.Image(label="Selected Region (Input, 384×384)", type="pil")
            with gr.Column(scale=1):
                in_region_clip_chart = gr.Image(label="Text → CLIP Input Region Attention", type="pil")
                in_region_vq_chart = gr.Image(label="Text → VQ Input Region Attention", type="pil")

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def process_edit(image, prompt, cfg1, cfg2, temperature, seed):
        """Run image editing with attention capture and return results."""
        torch.cuda.empty_cache()
        if image is None:
            return None, None, gr.update(choices=[], value=None), gr.update(maximum=29, value=0)

        if seed is not None:
            seed = int(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed)

        result = generate_image_v2v_mask_v3(
            question=prompt,
            input_image=image,
            temperature=temperature,
            parallel_size=1,
            cfg_weight=cfg1,
            cfg_weight2=cfg2,
            return_attention=True,
        )

        images, attn_maps = result
        edited = images[0]

        attn_maps["input_image"] = image

        token_choices = []
        if "tokens" in attn_maps:
            prompt_indices = attn_maps.get("prompt_token_indices", range(len(attn_maps["tokens"])))
            token_choices = [f"{i}: {attn_maps['tokens'][i]}" for i in prompt_indices]

        n_layers = attn_maps.get("num_layers", 30)

        return (
            edited,
            attn_maps,
            gr.update(choices=token_choices, value=token_choices[0] if token_choices else None),
            gr.update(maximum=max(n_layers - 1, 0), value=0),
        )

    def update_attention_viz(token_choice, attn_maps, result_image, avg_layers, layer_val):
        """Update attention heatmap visualizations when controls change."""
        if attn_maps is None or "tokens" not in attn_maps:
            return None, None, None
        if token_choice is None:
            return None, None, None

        try:
            token_idx = int(token_choice.split(":")[0])
        except (ValueError, AttributeError):
            return None, None, None

        input_image = attn_maps.get("input_image")
        if input_image is None or result_image is None:
            return None, None, None

        li = None if avg_layers else (int(layer_val) if layer_val is not None else None)
        viz_enc, viz_vq, viz_out = get_attention_visualizations(
            token_idx, attn_maps, input_image, result_image, layer_idx=li
        )
        return viz_enc, viz_vq, viz_out

    def toggle_layer_slider(avg_all):
        return gr.update(visible=not avg_all)

    def update_region_viz(attn_maps, result_image, x1, y1, x2, y2, avg_layers, layer_val):
        """Update output region attention preview and bar chart."""
        if attn_maps is None or result_image is None:
            return None, None
        if "tokens" not in attn_maps:
            return None, None
        if any(v is None for v in (x1, y1, x2, y2)):
            return None, None

        preview = draw_bbox_preview(result_image, x1, y1, x2, y2)
        li = None if avg_layers else (int(layer_val) if layer_val is not None else None)
        attn_weights, labels = compute_region_attention(attn_maps, x1, y1, x2, y2, li)
        chart = render_region_attention_chart(attn_weights, labels,
                                              title="Output Region -> Text Attention")
        return preview, chart

    def update_input_region_viz(attn_maps, x1, y1, x2, y2, avg_layers, layer_val):
        """Update input region attention preview with CLIP and VQ bar charts."""
        if attn_maps is None or "tokens" not in attn_maps:
            return None, None, None
        if any(v is None for v in (x1, y1, x2, y2)):
            return None, None, None

        input_image = attn_maps.get("input_image")
        if input_image is None:
            return None, None, None

        if isinstance(input_image, np.ndarray):
            input_pil = Image.fromarray(input_image)
        else:
            input_pil = input_image
        input_384 = input_pil.convert("RGB").resize((384, 384), Image.LANCZOS)
        preview = draw_bbox_preview(input_384, x1, y1, x2, y2)

        li = None if avg_layers else (int(layer_val) if layer_val is not None else None)
        clip_weights, clip_labels = compute_input_region_attention(
            attn_maps, x1, y1, x2, y2, attn_type="encoder", layer_idx=li)
        vq_weights, vq_labels = compute_input_region_attention(
            attn_maps, x1, y1, x2, y2, attn_type="vq", layer_idx=li)
        clip_chart = render_region_attention_chart(
            clip_weights, clip_labels, title="Text -> Input Region Attention (CLIP)")
        vq_chart = render_region_attention_chart(
            vq_weights, vq_labels, title="Text -> Input Region Attention (VQ)")
        return preview, clip_chart, vq_chart

    # --- Wire up events ---
    avg_layers_checkbox.change(
        fn=toggle_layer_slider,
        inputs=[avg_layers_checkbox],
        outputs=[layer_slider]
    )

    edit_btn.click(
        fn=process_edit,
        inputs=[edit_image_input, edit_prompt, edit_cfg1, edit_cfg2, edit_temperature, edit_seed],
        outputs=[edit_image_output, attention_state, token_selector, layer_slider],
    )

    viz_inputs = [token_selector, attention_state, edit_image_output, avg_layers_checkbox, layer_slider]
    viz_outputs = [text_enc_viz, text_vq_viz, out_text_viz]

    for trigger in [token_selector, avg_layers_checkbox, layer_slider]:
        trigger.change(
            fn=update_attention_viz,
            inputs=viz_inputs,
            outputs=viz_outputs,
        )

    # --- Region attention callbacks ---
    # Layer controls are read from inputs but NOT used as triggers to avoid cascading
    # re-renders (layer_slider changes from toggle_layer_slider would re-fire these).
    region_inputs = [attention_state, edit_image_output,
                     region_x1, region_y1, region_x2, region_y2,
                     avg_layers_checkbox, layer_slider]
    region_outputs = [region_preview, region_chart]

    for trigger in [region_x1, region_y1, region_x2, region_y2]:
        trigger.change(
            fn=update_region_viz,
            inputs=region_inputs,
            outputs=region_outputs,
        )

    edit_image_output.change(
        fn=lambda img, maps, x1, y1, x2, y2, avg, ly: update_region_viz(maps, img, x1, y1, x2, y2, avg, ly),
        inputs=[edit_image_output, attention_state,
                region_x1, region_y1, region_x2, region_y2,
                avg_layers_checkbox, layer_slider],
        outputs=region_outputs,
    )

    # --- Input region attention callbacks ---
    in_region_inputs = [attention_state,
                        in_region_x1, in_region_y1, in_region_x2, in_region_y2,
                        avg_layers_checkbox, layer_slider]
    in_region_outputs = [in_region_preview, in_region_clip_chart, in_region_vq_chart]

    for trigger in [in_region_x1, in_region_y1, in_region_x2, in_region_y2]:
        trigger.change(
            fn=update_input_region_viz,
            inputs=in_region_inputs,
            outputs=in_region_outputs,
        )

    edit_image_output.change(
        fn=lambda img, maps, x1, y1, x2, y2, avg, ly: update_input_region_viz(maps, x1, y1, x2, y2, avg, ly),
        inputs=[edit_image_output, attention_state,
                in_region_x1, in_region_y1, in_region_x2, in_region_y2,
                avg_layers_checkbox, layer_slider],
        outputs=in_region_outputs,
    )

demo.launch()