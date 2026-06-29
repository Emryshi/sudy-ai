import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Dict

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class QuantizedLinear(nn.Module):
    """
    8-bit Weight-Only Quantized Linear Layer for memory savings in low-RAM systems.
    Weights are stored as int8 and dynamically dequantized to float during forward pass.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.register_buffer("qweight", None)
        self.register_buffer("qscale", None)
        self.quantized = False

    def quantize(self):
        if self.quantized:
            return
        
        w = self.weight.data
        max_vals = torch.max(torch.abs(w), dim=-1, keepdim=True)[0]
        max_vals = torch.clamp(max_vals, min=1e-5)
        
        # Scale to map float to range [-127, 127]
        scale = max_vals / 127.0
        qweight = torch.clamp(torch.round(w / scale), -127, 127).to(torch.int8)
        
        self.register_buffer("qweight", qweight)
        self.register_buffer("qscale", scale)
        self.quantized = True
        
        # Delete original weight parameters to free RAM
        del self.weight
        self.weight = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.quantized:
            w = self.qweight.to(x.dtype) * self.qscale
            return F.linear(x, w, self.bias)
        else:
            return F.linear(x, self.weight, self.bias)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 4096, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.theta = theta
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len, scale: float = 1.0):
        t = torch.arange(seq_len, dtype=torch.float32)
        if scale > 1.0:
            # Dynamic NTK Scaling: modifies the base theta frequency dynamically
            scaled_theta = self.theta * (scale ** (self.dim / (self.dim - 2)))
            inv_freq = 1.0 / (scaled_theta ** (torch.arange(0, self.dim, 2).float() / self.dim))
        else:
            inv_freq = self.inv_freq
            
        freqs = torch.outer(t, inv_freq.to(t.device))
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x, seq_len: int):
        if seq_len > self.max_position_embeddings:
            scale = seq_len / self.max_position_embeddings
            self._set_cos_sin_cache(seq_len, scale=scale)
        else:
            if self.cos_cached.shape[0] < seq_len:
                self._set_cos_sin_cache(seq_len)
        return self.cos_cached[:seq_len].to(x.device), self.sin_cached[:seq_len].to(x.device)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    # q, k shapes: [batch, n_heads, seq_len, head_dim]
    # cos, sin shapes: [seq_len, head_dim] -> unsqueeze to match
    cos = cos.unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, head_dim]
    sin = sin.unsqueeze(0).unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MultiHeadLatentAttention(nn.Module):
    """
    Multi-Head Latent Attention (MLA) architecture.
    Compresses Key-Value cache to a latent space dimension to save memory.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.d_latent = config.d_latent
        self.max_position_embeddings = config.max_position_embeddings
        self.quantize_kv = getattr(config, "quantize_kv", False)

        # Head dimensions
        self.d_head_c = self.hidden_size // self.num_heads
        self.d_rope_head = 64  # decoupled position dim

        # Down-projection for KV
        self.W_down_kv = nn.Linear(self.hidden_size, self.d_latent, bias=False)
        self.norm_kv = RMSNorm(self.d_latent, eps=config.rms_norm_eps)

        # Up-projection for KV
        self.W_up_k = nn.Linear(self.d_latent, self.num_heads * self.d_head_c, bias=False)
        self.W_up_v = nn.Linear(self.d_latent, self.num_heads * self.d_head_c, bias=False)

        # Query projection (MLA compresses or projects directly)
        self.W_q = nn.Linear(self.hidden_size, self.num_heads * self.d_head_c, bias=False)
        
        # Decoupled keys and queries for Rotary position embedding (RoPE)
        self.W_qr = nn.Linear(self.hidden_size, self.num_heads * self.d_rope_head, bias=False)
        self.W_kr = nn.Linear(self.hidden_size, self.d_rope_head, bias=False)  # shared rope key or head-wise

        # Output projection
        self.W_o = nn.Linear(self.num_heads * self.d_head_c, self.hidden_size, bias=False)

        self.rotary_emb = RotaryEmbedding(self.d_rope_head, self.max_position_embeddings, config.rope_theta)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        
        batch_size, seq_len, _ = hidden_states.shape

        # 1. Project Query
        q_c = self.W_q(hidden_states).view(batch_size, seq_len, self.num_heads, self.d_head_c).transpose(1, 2)
        q_r = self.W_qr(hidden_states).view(batch_size, seq_len, self.num_heads, self.d_rope_head).transpose(1, 2)

        # 2. Compress Key and Value
        compressed_kv = self.norm_kv(self.W_down_kv(hidden_states))  # [B, L, d_latent]

        # Decoupled RoPE Key
        k_r = self.W_kr(hidden_states).view(batch_size, seq_len, 1, self.d_rope_head).transpose(1, 2)
        # Expand k_r for heads to make it simple
        k_r = k_r.expand(-1, self.num_heads, -1, -1)

        # Apply Rotary Position Embeddings to decoupled parts
        cos, sin = self.rotary_emb(q_r, seq_len)
        q_r, k_r = apply_rotary_pos_emb(q_r, k_r, cos, sin)

        # 3. Handle KV Cache with optional Int8 Quantization to save VRAM/RAM
        if past_key_value is not None:
            if self.quantize_kv:
                cached_compressed_q, scale_kv, cached_k_r_q, scale_kr = past_key_value
                cached_compressed = cached_compressed_q.to(hidden_states.dtype) * scale_kv
                cached_k_r = cached_k_r_q.to(hidden_states.dtype) * scale_kr
            else:
                cached_compressed, cached_k_r = past_key_value
                
            compressed_kv = torch.cat([cached_compressed, compressed_kv], dim=1)
            k_r = torch.cat([cached_k_r, k_r], dim=2)
            total_seq_len = compressed_kv.shape[1]
        else:
            total_seq_len = seq_len

        # Save Cache
        if use_cache:
            if self.quantize_kv:
                # Quantize KV cache vectors to int8 to save RAM
                max_kv = torch.max(torch.abs(compressed_kv)).clamp(min=1e-5)
                scale_kv = max_kv / 127.0
                compressed_kv_q = torch.clamp(torch.round(compressed_kv / scale_kv), -127, 127).to(torch.int8)
                
                max_kr = torch.max(torch.abs(k_r)).clamp(min=1e-5)
                scale_kr = max_kr / 127.0
                k_r_q = torch.clamp(torch.round(k_r / scale_kr), -127, 127).to(torch.int8)
                
                next_cache = (compressed_kv_q, scale_kv, k_r_q, scale_kr)
            else:
                next_cache = (compressed_kv, k_r)
        else:
            next_cache = None

        # 4. Decompress keys and values for attention
        k_c = self.W_up_k(compressed_kv).view(batch_size, total_seq_len, self.num_heads, self.d_head_c).transpose(1, 2)
        v_c = self.W_up_v(compressed_kv).view(batch_size, total_seq_len, self.num_heads, self.d_head_c).transpose(1, 2)

        # 5. Concatenate content and positional parts for attention
        # Query concat: [B, H, L_q, d_head_c + d_rope]
        # Key concat:   [B, H, L_k, d_head_c + d_rope]
        q_concat = torch.cat([q_c, q_r], dim=-1)
        k_concat = torch.cat([k_c, k_r], dim=-1)

        # PyTorch SDPA (FlashAttention / Memory-Efficient attention)
        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=q_concat.dtype)
            
        context_states = F.scaled_dot_product_attention(
            q_concat,
            k_concat,
            v_c,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False
        )

        # Reshape and project out
        context_states = context_states.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.W_o(context_states)

        return output, next_cache


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed Forward Network."""
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def get_turkish_features(token_text: str) -> List[float]:
    """Extracts 8 binary features representing Turkish grammatical characteristics."""
    if not token_text:
        return [0.0] * 8
        
    features = [0.0] * 8
    
    # 1. Front Vowels (e, i, ö, ü)
    if any(c in token_text for c in "eiöüEİÖÜ"):
        features[0] = 1.0
        
    # 2. Back Vowels (a, ı, o, u)
    if any(c in token_text for c in "aıouAIOU"):
        features[1] = 1.0
        
    # 3. Ends with mutable consonants (p, ç, t, k)
    if token_text[-1] in "pçtkPÇTK":
        features[2] = 1.0
        
    # 4. Ends with a vowel
    if token_text[-1] in "aeıioöuüAEIİOÖUÜ":
        features[3] = 1.0
        
    # 5. Verb suffix candidate
    if any(suffix in token_text for suffix in ["yor", "acak", "ecek", "meli", "malı", "dı", "di", "du", "dü", "tı", "ti", "tu", "tü"]):
        features[4] = 1.0
        
    # 6. Noun suffix / plural candidate
    if any(suffix in token_text for suffix in ["lar", "ler", "da", "de", "ta", "te", "dan", "den", "tan", "ten"]):
        features[5] = 1.0
        
    # 7. Word boundary (starts with space or special BPE space indicator)
    if token_text.startswith("Ġ") or token_text.startswith(" ") or token_text.startswith(" "):
        features[6] = 1.0
        
    # 8. Numeric candidate
    if any(c.isdigit() for c in token_text):
        features[7] = 1.0
        
    return features


class MorphologicalMoERouter(nn.Module):
    """
    Turkish Morphological MoE Router that guides routing decisions 
    based on both token hidden states and token character/suffix patterns.
    """
    def __init__(self, config, num_experts: int):
        super().__init__()
        self.config = config
        self.num_experts = num_experts
        self.hidden_size = config.hidden_size
        
        # Standard hidden router
        self.hidden_router = nn.Linear(self.hidden_size, num_experts, bias=False)
        
        # Linguistic router projecting 8 features to num_experts
        self.linguistic_router = nn.Linear(8, num_experts, bias=False)
        
        # Buffer to store linguistic features for vocabulary size
        self.register_buffer("linguistic_features", torch.zeros((config.vocab_size, 8)))
        
    def setup_features(self, tokenizer):
        """Precomputes and registers linguistic features for vocabulary."""
        features_matrix = torch.zeros((self.linguistic_features.shape[0], 8))
        for i in range(self.linguistic_features.shape[0]):
            try:
                token_text = tokenizer.decode([i])
            except Exception:
                token_text = ""
            features_matrix[i] = torch.tensor(get_turkish_features(token_text))
            
        self.linguistic_features.copy_(features_matrix)

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        # Standard router logits
        hidden_logits = self.hidden_router(hidden_states)  # [B, L, num_experts]
        
        # Fetch linguistic features
        vocab_size = self.linguistic_features.shape[0]
        safe_ids = torch.clamp(input_ids, 0, vocab_size - 1)
        
        # [B, L, 8]
        features = self.linguistic_features[safe_ids].to(dtype=hidden_states.dtype, device=hidden_states.device)
        ling_logits = self.linguistic_router(features)  # [B, L, num_experts]
        
        # Combine hidden states routing with Turkish morphological routing
        combined_logits = hidden_logits + 0.3 * ling_logits
        return combined_logits


class MoELayer(nn.Module):
    """
    Mixture of Experts (MoE) layer with routed experts and shared experts
    using a routed expert and shared expert framework.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.aux_loss_coef = config.moe_aux_loss_coef

        # Expert hidden size (usually smaller in MoE to balance total parameter count)
        expert_hidden_dim = int(2 * self.hidden_size * 2 / 3)

        # Routed Experts
        self.experts = nn.ModuleList([
            SwiGLUFFN(self.hidden_size, expert_hidden_dim) for _ in range(self.num_experts)
        ])
        
        # Morphological Router
        self.router = MorphologicalMoERouter(config, self.num_experts)

        # Shared Experts (always active)
        self.num_shared_experts = config.num_shared_experts
        if self.num_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                SwiGLUFFN(self.hidden_size, expert_hidden_dim) for _ in range(self.num_shared_experts)
            ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)  # [B * L, D]

        # Compute routing logits using the custom MorphologicalMoERouter
        input_ids = getattr(self.config, "current_input_ids", None)
        if input_ids is None:
            input_ids = torch.zeros((batch_size, seq_len), dtype=torch.long, device=x.device)
            
        router_logits = self.router(x, input_ids)  # [B, L, N]
        router_logits = router_logits.view(-1, self.num_experts)  # [B * L, N]
        router_probs = F.softmax(router_logits, dim=-1)  # [B * L, N]

        # Select top-k experts
        top_k_weights, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        # Normalize weights
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        # Compute Load-Balancing (Auxiliary) Loss to prevent routing collapse
        # f_i: fraction of tokens routed to expert i
        # P_i: average routing probability for expert i
        # Aux Loss = N * sum(f_i * P_i)
        tokens_per_expert = torch.zeros(self.num_experts, device=x.device)
        # count indices
        indices_flat = top_k_indices.view(-1)
        tokens_per_expert.scatter_add_(0, indices_flat, torch.ones_like(indices_flat, dtype=torch.float))
        
        f = tokens_per_expert / (batch_size * seq_len * self.top_k)
        P = router_probs.mean(dim=0)
        aux_loss = self.num_experts * torch.sum(f * P) * self.aux_loss_coef

        # Route inputs to experts
        output_flat = torch.zeros_like(x_flat)
        
        # Execute experts
        # For simplicity and correctness in PyTorch, we can loop over experts
        for expert_idx in range(self.num_experts):
            # Find tokens where this expert is in top-k
            mask = (top_k_indices == expert_idx)
            if not mask.any():
                continue
            
            # token_indices: which tokens go to this expert
            # weight_indices: which of the top-k positions it was selected in
            token_indices, weight_indices = torch.where(mask)
            
            # Extract inputs for this expert
            expert_inputs = x_flat[token_indices]
            expert_outputs = self.experts[expert_idx](expert_inputs)
            
            # Scale by routing weight and accumulate
            weights = top_k_weights[token_indices, weight_indices].unsqueeze(-1)
            output_flat[token_indices] += weights * expert_outputs

        output = output_flat.view(batch_size, seq_len, hidden_dim)

        # Run shared experts
        if self.num_shared_experts > 0:
            for shared_expert in self.shared_experts:
                output = output + shared_expert(x)

        return output, aux_loss


class MTPModule(nn.Module):
    """
    Multi-Token Prediction (MTP) module.
    It sequentially predicts the next tokens (t+1, t+2, etc.) using additional prediction heads.
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.num_heads = config.num_mtp_heads

        # Sequential prediction steps
        # Each head will combine the hidden state of the current step and the embedding of the predicted token
        # to predict the token at step + k
        self.projection_layers = nn.ModuleList([
            nn.Linear(self.hidden_size * 2, self.hidden_size) for _ in range(self.num_heads)
        ])
        
        self.mtp_blocks = nn.ModuleList([
            SwiGLUFFN(self.hidden_size, self.hidden_size * 2) for _ in range(self.num_heads)
        ])

        # Prediction heads
        self.prediction_heads = nn.ModuleList([
            nn.Linear(self.hidden_size, self.vocab_size, bias=False) for _ in range(self.num_heads)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        next_token_embeddings: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        hidden_states: last layer hidden states [B, L, H]
        next_token_embeddings: List of token embeddings of length num_heads.
                               next_token_embeddings[0] represents embeddings of token t+1.
                               next_token_embeddings[1] represents embeddings of token t+2.
        """
        predictions = []
        current_state = hidden_states

        for i in range(min(self.num_heads, len(next_token_embeddings))):
            # Combine current state and embedding of predicted token at t + i + 1
            # E.g. to predict t+2, combine h_t and embedding(t+1)
            emb = next_token_embeddings[i]  # [B, L, H]
            combined = torch.cat([current_state, emb], dim=-1)
            
            # Project and apply FFN block
            proj = F.relu(self.projection_layers[i](combined))
            current_state = proj + self.mtp_blocks[i](proj)  # residual connection
            
            # Predict
            logits = self.prediction_heads[i](current_state)
            predictions.append(logits)
            
        return predictions


class SudyBlock(nn.Module):
    """A single Transformer block containing MLA and MoE/FFN."""
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.attn_norm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.attn = MultiHeadLatentAttention(config)
        
        self.ffn_norm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.moe = MoELayer(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Pre-LN MLA
        normed_attn = self.attn_norm(hidden_states)
        attn_out, next_cache = self.attn(
            normed_attn,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache
        )
        hidden_states = hidden_states + attn_out

        # Pre-LN MoE
        normed_ffn = self.ffn_norm(hidden_states)
        moe_out, aux_loss = self.moe(normed_ffn)
        hidden_states = hidden_states + moe_out

        return hidden_states, aux_loss, next_cache


class SudyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList([SudyBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        
        batch_size, seq_len = input_ids.shape
        self.config.current_input_ids = input_ids
        hidden_states = self.embed_tokens(input_ids)

        # Build causal attention mask if not provided
        if attention_mask is None:
            if past_key_values is not None and len(past_key_values) > 0 and past_key_values[0] is not None:
                # past_key_values[0][0] is compressed_kv of shape [B, L_cached, d_latent]
                total_kv_len = past_key_values[0][0].shape[1] + seq_len
            else:
                total_kv_len = seq_len
                
            q_idx = torch.arange(seq_len, device=input_ids.device).unsqueeze(-1)  # [seq_len, 1]
            k_idx = torch.arange(total_kv_len, device=input_ids.device).unsqueeze(0)  # [1, total_kv_len]
            
            mask = torch.zeros((seq_len, total_kv_len), device=input_ids.device)
            mask = mask.masked_fill(k_idx > (total_kv_len - seq_len + q_idx), float("-inf"))
            attention_mask = mask.unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, total_kv_len]

        next_caches = []
        total_aux_loss = torch.tensor(0.0, device=input_ids.device)

        for i, layer in enumerate(self.layers):
            layer_past = past_key_values[i] if past_key_values is not None else None
            hidden_states, aux_loss, next_cache = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=layer_past,
                use_cache=use_cache
            )
            total_aux_loss += aux_loss
            if use_cache:
                next_caches.append(next_cache)

        hidden_states = self.norm(hidden_states)
        self.config.current_input_ids = None # Clean up tensor reference before serialization!

        return hidden_states, total_aux_loss, (next_caches if use_cache else None)


class SudyLMHeadModel(nn.Module):
    """The main language model class including standard LM head and MTP heads."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = SudyModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # MTP Module
        if config.num_mtp_heads > 0:
            self.mtp_module = MTPModule(config)

        # Weight tying
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False
    ) -> Dict[str, torch.Tensor]:
        
        hidden_states, aux_loss, next_caches = self.model(
            input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache
        )

        # Standard language model head prediction (t+1)
        logits = self.lm_head(hidden_states)

        loss = None
        mtp_losses = []
        
        if labels is not None:
            # Shift labels and logits for autoregressive loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            
            # Add aux routing loss
            loss = loss + aux_loss

            # Calculate MTP (Multi-Token Prediction) loss during training
            if self.config.num_mtp_heads > 0:
                # E.g. For prediction head i (which predicts t+i+2), the target labels are shifted by i+2
                # We need embeddings of the token at t+1, t+2, etc.
                next_token_embeddings = []
                for i in range(self.config.num_mtp_heads):
                    # Shift input_ids to get targets
                    # E.g. for i=0 (predicting t+2), we need embedding of t+1
                    # Since we want to predict t+2, we combine h_t and embed(t+1)
                    # We can slice input_ids to get these token embeddings
                    shift_idx = i + 1
                    if shift_idx < input_ids.shape[1]:
                        # pad at the end for sequence length alignment
                        shifted_ids = torch.cat([
                            input_ids[:, shift_idx:],
                            torch.full((input_ids.shape[0], shift_idx), self.config.pad_token_id, device=input_ids.device)
                        ], dim=1)
                    else:
                        shifted_ids = torch.full_like(input_ids, self.config.pad_token_id)
                        
                    emb = self.model.embed_tokens(shifted_ids)
                    next_token_embeddings.append(emb)

                # MTP forward pass
                mtp_logits_list = self.mtp_module(hidden_states, next_token_embeddings)

                for i, mtp_logits in enumerate(mtp_logits_list):
                    # E.g. i=0 predicts t+2, so target labels are labels[..., 2:]
                    shift_target_idx = i + 2
                    if shift_target_idx < labels.shape[1]:
                        # Slice logits and labels to match
                        slice_logits = mtp_logits[:, :-shift_target_idx, :].contiguous()
                        slice_labels = labels[:, shift_target_idx:].contiguous()
                        
                        mtp_loss = loss_fct(slice_logits.view(-1, self.config.vocab_size), slice_labels.view(-1))
                        mtp_losses.append(mtp_loss)
                        # Add scaled MTP loss to main loss (e.g. 0.3 factor)
                        loss = loss + 0.3 * mtp_loss

        return {
            "logits": logits,
            "loss": loss,
            "aux_loss": aux_loss,
            "mtp_losses": mtp_losses,
            "past_key_values": next_caches
        }

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
        eos_token_id: Optional[int] = None
    ) -> torch.Tensor:
        """Simple greedy / temperature-based decoding helper."""
        self.eval()
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id

        past_key_values = None
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Only feed the last token if we have key-value cache
                if past_key_values is not None:
                    model_inputs = input_ids[:, -1:]
                else:
                    model_inputs = input_ids

                outputs = self.forward(model_inputs, past_key_values=past_key_values, use_cache=True)
                logits = outputs["logits"][:, -1, :]
                past_key_values = outputs["past_key_values"]

                if temperature == 0.0:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                else:
                    logits = logits / temperature
                    # Apply top-k filtering if needed
                    if top_k > 0:
                        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                        logits[logits < v[:, [-1]]] = -float("Inf")
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)

                input_ids = torch.cat([input_ids, next_token], dim=-1)
                
                # Check if all batches reached EOS
                if (next_token == eos_token_id).all():
                    break

        return input_ids

    def generate_speculative(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
        eos_token_id: Optional[int] = None,
        mtp_threshold: float = 0.7
    ) -> torch.Tensor:
        """
        Speculative decoding using MTP heads to predict t+2 token in parallel with t+1 token.
        If MTP head confidence is above mtp_threshold, both tokens are accepted in 1 step.
        """
        self.eval()
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id

        has_mtp = hasattr(self, "mtp_module") and self.config.num_mtp_heads > 0
        past_key_values = None
        num_tokens_to_feed = 1

        with torch.no_grad():
            tokens_generated = 0
            while tokens_generated < max_new_tokens:
                # 1. Standard forward pass
                if past_key_values is not None:
                    model_inputs = input_ids[:, -num_tokens_to_feed:]
                else:
                    model_inputs = input_ids

                outputs = self.forward(model_inputs, past_key_values=past_key_values, use_cache=True)
                hidden_states = outputs["logits"]  # standard logits output before project (or logits themselves)
                # To make sure we get last hidden states correctly, we can pass them or do it from self.model
                # Wait, our forward pass returns standard logits.
                # To obtain hidden states, we can run self.model directly or retrieve it from outputs if cached.
                # Let's extract hidden states from self.model
                hidden_states, _, next_kv = self.model(
                    model_inputs,
                    past_key_values=past_key_values,
                    use_cache=True
                )
                past_key_values = next_kv

                # Standard logits for token t+1
                logits_standard = self.lm_head(hidden_states[:, -1, :])

                # Sample token t+1
                if temperature == 0.0:
                    next_token_1 = torch.argmax(logits_standard, dim=-1, keepdim=True)
                else:
                    logits_1 = logits_standard / temperature
                    if top_k > 0:
                        v, _ = torch.topk(logits_1, min(top_k, logits_1.size(-1)))
                        logits_1[logits_1 < v[:, [-1]]] = -float("Inf")
                    probs_1 = F.softmax(logits_1, dim=-1)
                    next_token_1 = torch.multinomial(probs_1, num_samples=1)

                input_ids = torch.cat([input_ids, next_token_1], dim=-1)
                tokens_generated += 1
                num_tokens_to_feed = 1  # reset to default

                if (next_token_1 == eos_token_id).all() or tokens_generated >= max_new_tokens:
                    break

                # 2. Speculative MTP prediction for token t+2
                # Supported only for batch size 1 and if MTP is enabled
                if has_mtp and input_ids.shape[0] == 1 and tokens_generated < max_new_tokens:
                    emb_next = self.model.embed_tokens(next_token_1)  # [1, 1, H]
                    
                    # Feed hidden state of step t and next token embedding of step t+1 to MTP head 0
                    mtp_logits_list = self.mtp_module(hidden_states[:, -1:, :], [emb_next])
                    logits_mtp = mtp_logits_list[0][:, -1, :]  # [1, vocab_size]

                    probs_mtp = F.softmax(logits_mtp, dim=-1)
                    max_prob, next_token_2 = torch.max(probs_mtp, dim=-1)

                    if max_prob.item() > mtp_threshold:
                        next_token_2 = next_token_2.view(1, 1)
                        input_ids = torch.cat([input_ids, next_token_2], dim=-1)
                        tokens_generated += 1
                        num_tokens_to_feed = 2  # Feed last 2 tokens next step to align KV cache
                        
                        if (next_token_2 == eos_token_id).all():
                            break

        return input_ids

    def quantize_model(self):
        """
        Recursively converts all standard linear layers in the model to QuantizedLinear 
        and quantizes their weights to 8-bit to run on low-RAM systems.
        """
        def replace_layers(module):
            for name, child in module.named_children():
                if isinstance(child, nn.Linear):
                    # Skip lm_head if tied to embed_tokens to keep model integrity
                    if name == "lm_head" and getattr(self.config, "tie_word_embeddings", False):
                        continue
                    q_child = QuantizedLinear(child.in_features, child.out_features, child.bias is not None)
                    q_child.weight.data.copy_(child.weight.data)
                    if child.bias is not None:
                        q_child.bias.data.copy_(child.bias.data)
                    q_child.quantize()
                    setattr(module, name, q_child)
                else:
                    replace_layers(child)
        replace_layers(self)
        self.config.quantized = True
        return self

    def setup_linguistic_router(self, tokenizer):
        """Precomputes and registers linguistic features in MoE routers."""
        for layer in self.model.layers:
            if hasattr(layer, "moe") and hasattr(layer.moe, "router") and hasattr(layer.moe.router, "setup_features"):
                layer.moe.router.setup_features(tokenizer)
        return self

    def expand_embeddings(self, new_tokens: List[str]):
        """
        Expands embedding layers, LM head, and MTP heads to accommodate new tokens
        while preserving all previously trained weights.
        """
        if not new_tokens:
            return self
            
        old_vocab_size = self.config.vocab_size
        new_vocab_size = old_vocab_size + len(new_tokens)
        hidden_size = self.config.hidden_size
        
        # 1. Expand input embeddings
        old_embed = self.model.embed_tokens
        new_embed = nn.Embedding(new_vocab_size, hidden_size, padding_idx=self.config.pad_token_id)
        nn.init.normal_(new_embed.weight, mean=0.0, std=self.config.initializer_range)
        with torch.no_grad():
            new_embed.weight[:old_vocab_size].copy_(old_embed.weight)
        self.model.embed_tokens = new_embed
        
        # 2. Expand LM Head
        old_lm_head = self.lm_head
        new_lm_head = nn.Linear(hidden_size, new_vocab_size, bias=False)
        nn.init.normal_(new_lm_head.weight, mean=0.0, std=self.config.initializer_range)
        with torch.no_grad():
            new_lm_head.weight[:old_vocab_size].copy_(old_lm_head.weight)
        self.lm_head = new_lm_head
        
        # Tie word embeddings if configured
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
            
        # 3. Expand MTP heads if active
        if hasattr(self, "mtp_module") and self.config.num_mtp_heads > 0:
            new_prediction_heads = nn.ModuleList()
            for head in self.mtp_module.prediction_heads:
                new_head = nn.Linear(hidden_size, new_vocab_size, bias=False)
                nn.init.normal_(new_head.weight, mean=0.0, std=self.config.initializer_range)
                with torch.no_grad():
                    new_head.weight[:old_vocab_size].copy_(head.weight)
                new_prediction_heads.append(new_head)
            self.mtp_module.prediction_heads = new_prediction_heads
            
        # 4. Expand MoE morphological router features buffer
        for layer in self.model.layers:
            if hasattr(layer, "moe") and hasattr(layer.moe, "router"):
                router = layer.moe.router
                old_features = router.linguistic_features
                new_features = torch.zeros((new_vocab_size, 8), device=old_features.device)
                new_features[:old_vocab_size].copy_(old_features)
                for idx, token_text in enumerate(new_tokens):
                    new_features[old_vocab_size + idx] = torch.tensor(get_turkish_features(token_text))
                router.register_buffer("linguistic_features", new_features)
                
        # 5. Update config vocab_size
        self.config.vocab_size = new_vocab_size
        return self
