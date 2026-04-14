import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


class JanusAttentionHook:
    """
    Hook to capture attention weights from LlamaAttention layers in Janus model
    during image editing. Captures three types of cross-attention:
      - text -> encoder_image: how text tokens attend to CLIP encoder image tokens
      - text -> input_image: how text tokens attend to input VQ image tokens
      - output_image -> text: how generated image tokens attend to text tokens

    Designed for use with eager attention (_attn_implementation='eager').
    """

    def __init__(self):
        self.hooks = []
        self.enabled = False
        self.phase = "prefill"  # "prefill" or "generate"
        self.gen_step = 0

        # Token boundary info (positions in the prefill sequence)
        self.text_range = None              # (start, end) of text question tokens
        self.input_image_range = None       # (start, end) of VQ input image tokens
        self.encoder_image_range = None     # (start, end) of CLIP encoder image tokens
        self.stream0_idx = 0                # batch index of the fully-conditioned stream

        # Storage -------------------------------------------------------
        # text_to_encoder_image[layer_idx] = list of tensors (num_text, num_enc_img) per call
        # text_to_image[layer_idx] = list of tensors (num_text, num_img) per call
        # output_to_text[layer_idx] = list of tensors (1, num_text) per gen step
        self.text_to_encoder_image: Dict[int, List[torch.Tensor]] = defaultdict(list)
        self.text_to_image: Dict[int, List[torch.Tensor]] = defaultdict(list)
        self.output_to_text: Dict[int, List[torch.Tensor]] = defaultdict(list)

        self.num_layers = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enable(self):
        self.enabled = True
        self.phase = "prefill"
        self.gen_step = 0
        self.text_to_encoder_image.clear()
        self.text_to_image.clear()
        self.output_to_text.clear()

    def disable(self):
        self.enabled = False

    def clear(self):
        self.text_to_encoder_image.clear()
        self.text_to_image.clear()
        self.output_to_text.clear()
        self.phase = "prefill"
        self.gen_step = 0

    def set_token_ranges(self, text_range: Tuple[int, int],
                         input_image_range: Tuple[int, int],
                         encoder_image_range: Optional[Tuple[int, int]] = None,
                         stream0_idx: int = 0):
        self.text_range = text_range
        self.input_image_range = input_image_range
        self.encoder_image_range = encoder_image_range
        self.stream0_idx = stream0_idx

    def set_phase(self, phase: str):
        self.phase = phase

    def step_generation(self):
        self.gen_step += 1

    # ------------------------------------------------------------------
    # Hook registration
    # ------------------------------------------------------------------

    def register_hooks(self, model):
        self.remove_hooks()
        layers = model.language_model.model.layers
        self.num_layers = len(layers)
        for layer_idx, layer in enumerate(layers):
            attn_module = layer.self_attn
            cleanup = self._make_forward_hook(layer_idx, attn_module)
            self.hooks.append(cleanup)

    def remove_hooks(self):
        for cleanup in self.hooks:
            cleanup()
        self.hooks.clear()

    # ------------------------------------------------------------------
    # Internal: monkey-patch forward
    # ------------------------------------------------------------------

    def _make_forward_hook(self, layer_idx: int, attn_module):
        original_forward = attn_module.forward
        hook_self = self

        def hooked_forward(hidden_states, *args, **kwargs):
            # Always call the original forward first
            result = original_forward(hidden_states, *args, **kwargs)

            if not hook_self.enabled:
                return result
            if hook_self.text_range is None or hook_self.input_image_range is None:
                return result

            try:
                hook_self._capture_attention(
                    layer_idx, attn_module, hidden_states, args, kwargs, result
                )
            except Exception as e:
                print(f"[AttentionHook] layer {layer_idx} error: {e}")

            return result

        attn_module.forward = hooked_forward

        def cleanup():
            attn_module.forward = original_forward

        return cleanup

    def _capture_attention(self, layer_idx, attn_module, hidden_states,
                           fwd_args, fwd_kwargs, fwd_result):
        """
        After original_forward ran, the KV cache already contains the full K
        (with RoPE applied). We re-derive Q from hidden_states and read K
        directly from the cache.
        """
        bsz, q_len, hidden_size = hidden_states.shape
        s0 = self.stream0_idx

        num_heads = attn_module.config.num_attention_heads
        num_kv_heads = getattr(attn_module.config, "num_key_value_heads", num_heads)
        head_dim = attn_module.head_dim

        # ---- Compute Q with RoPE ------------------------------------------
        q = attn_module.q_proj(hidden_states)
        q = q.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)

        position_embeddings = fwd_kwargs.get("position_embeddings")
        if position_embeddings is None and len(fwd_args) >= 1:
            position_embeddings = fwd_args[0]

        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, _ = self._apply_rotary_pos_emb(q, q, cos, sin)

        # ---- Get full K from KV cache (already has RoPE, updated by original_forward) ---
        past_key_value = fwd_kwargs.get("past_key_value")
        if past_key_value is None:
            for arg in fwd_args:
                if hasattr(arg, "key_cache"):
                    past_key_value = arg
                    break

        full_k = None
        if past_key_value is not None and hasattr(past_key_value, "key_cache"):
            if layer_idx < len(past_key_value.key_cache) and past_key_value.key_cache[layer_idx].numel() > 0:
                full_k = past_key_value.key_cache[layer_idx]

        if full_k is None:
            k = attn_module.k_proj(hidden_states)
            k = k.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
            if position_embeddings is not None:
                cos, sin = position_embeddings
                _, k = self._apply_rotary_pos_emb(k, k, cos, sin)
            full_k = k

        # ---- expand KV heads for GQA --------------------------------------
        n_rep = num_heads // num_kv_heads
        if n_rep > 1:
            full_k = full_k.repeat_interleave(n_rep, dim=1)

        # ---- extract stream 0 only ----------------------------------------
        q_s0 = q[s0:s0+1].to(torch.float32)
        k_s0 = full_k[s0:s0+1].to(torch.float32)

        scale = 1.0 / (head_dim ** 0.5)

        if self.phase == "prefill":
            self._capture_prefill(layer_idx, q_s0, k_s0, scale)
        elif self.phase == "generate":
            self._capture_generate(layer_idx, q_s0, k_s0, scale)

    # ------------------------------------------------------------------
    # Prefill: text -> input_image (VQ + encoder) and first output -> text
    # ------------------------------------------------------------------

    def _capture_prefill(self, layer_idx, q, k, scale):
        """q, k shape: (1, heads, seq_len, head_dim)"""
        t_start, t_end = self.text_range
        i_start, i_end = self.input_image_range

        q_text = q[:, :, t_start:t_end, :]

        # 1) text -> VQ input image
        k_img = k[:, :, i_start:i_end, :]
        attn_scores = torch.matmul(q_text, k_img.transpose(-2, -1)) * scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_avg = attn_weights.mean(dim=1).squeeze(0)  # (num_text, num_img)
        self.text_to_image[layer_idx].append(attn_avg.detach().cpu())

        # 2) text -> CLIP encoder image
        if self.encoder_image_range is not None:
            e_start, e_end = self.encoder_image_range
            k_enc = k[:, :, e_start:e_end, :]
            attn_enc_scores = torch.matmul(q_text, k_enc.transpose(-2, -1)) * scale
            attn_enc_weights = F.softmax(attn_enc_scores, dim=-1)
            attn_enc_avg = attn_enc_weights.mean(dim=1).squeeze(0)  # (num_text, num_enc_img)
            self.text_to_encoder_image[layer_idx].append(attn_enc_avg.detach().cpu())

        # 3) First output -> text: the last query position (output <img_start>)
        #    predicts the first output image token, so capture its attention to text
        q_last = q[:, :, -1:, :]  # (1, heads, 1, dim)
        k_text = k[:, :, t_start:t_end, :]
        attn_out_scores = torch.matmul(q_last, k_text.transpose(-2, -1)) * scale
        attn_out_weights = F.softmax(attn_out_scores, dim=-1)
        attn_out_avg = attn_out_weights.mean(dim=1).squeeze(0)  # (1, num_text)
        self.output_to_text[layer_idx].append(attn_out_avg.detach().cpu())

    # ------------------------------------------------------------------
    # Generate: output_image_token -> text
    # ------------------------------------------------------------------

    def _capture_generate(self, layer_idx, q, k, scale):
        """q shape: (1, heads, 1, dim), k shape: (1, heads, full_seq, dim)"""
        t_start, t_end = self.text_range

        q_out = q  # (1, heads, 1, dim)
        k_text = k[:, :, t_start:t_end, :]

        attn_scores = torch.matmul(q_out, k_text.transpose(-2, -1)) * scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_avg = attn_weights.mean(dim=1).squeeze(0)  # (1, num_text)

        self.output_to_text[layer_idx].append(attn_avg.detach().cpu())

    # ------------------------------------------------------------------
    # RoPE helper
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_rotary_pos_emb(q, k, cos, sin):
        """Apply rotary positional embeddings (RoPE) to q and k."""
        cos = cos.unsqueeze(1) if cos.dim() == 3 else cos
        sin = sin.unsqueeze(1) if sin.dim() == 3 else sin

        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_text_to_encoder_image_attention(self, layer_idx: Optional[int] = None) -> Optional[torch.Tensor]:
        """
        Returns attention of shape (num_text, num_enc_img) for text -> CLIP encoder image.
        If layer_idx is None, averages across all layers.
        """
        if layer_idx is not None:
            tensors = self.text_to_encoder_image.get(layer_idx)
            if not tensors:
                return None
            return torch.stack(tensors).mean(dim=0)

        all_layers = []
        for li in sorted(self.text_to_encoder_image.keys()):
            t = torch.stack(self.text_to_encoder_image[li]).mean(dim=0)
            all_layers.append(t)
        if not all_layers:
            return None
        return torch.stack(all_layers).mean(dim=0)

    def get_text_to_image_attention(self, layer_idx: Optional[int] = None) -> Optional[torch.Tensor]:
        """
        Returns attention of shape (num_text, num_img) for text -> VQ input image.
        If layer_idx is None, averages across all layers.
        """
        if layer_idx is not None:
            tensors = self.text_to_image.get(layer_idx)
            if not tensors:
                return None
            return torch.stack(tensors).mean(dim=0)

        all_layers = []
        for li in sorted(self.text_to_image.keys()):
            t = torch.stack(self.text_to_image[li]).mean(dim=0)
            all_layers.append(t)
        if not all_layers:
            return None
        return torch.stack(all_layers).mean(dim=0)

    def get_output_to_text_attention(self, layer_idx: Optional[int] = None) -> Optional[torch.Tensor]:
        """
        Returns attention of shape (num_output_img_tokens, num_text).
        If layer_idx is None, averages across all layers.
        """
        if layer_idx is not None:
            tensors = self.output_to_text.get(layer_idx)
            if not tensors:
                return None
            return torch.cat(tensors, dim=0)  # (num_gen_steps, num_text)

        all_layers = []
        for li in sorted(self.output_to_text.keys()):
            t = torch.cat(self.output_to_text[li], dim=0)  # (num_gen_steps, num_text)
            all_layers.append(t)
        if not all_layers:
            return None
        return torch.stack(all_layers).mean(dim=0)

    def get_layer_indices(self) -> List[int]:
        layers = (set(self.text_to_encoder_image.keys())
                  | set(self.text_to_image.keys())
                  | set(self.output_to_text.keys()))
        return sorted(layers)
