from transformers import PretrainedConfig

class SudyConfig(PretrainedConfig):
    model_type = "sudy"

    def __init__(
        self,
        vocab_size=65536,
        hidden_size=512,
        num_hidden_layers=12,
        num_attention_heads=8,
        d_latent=128,
        num_experts=8,
        num_shared_experts=1,
        top_k=2,
        num_mtp_heads=2,
        max_position_embeddings=4096,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        moe_aux_loss_coef=0.001,
        rope_theta=10000.0,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        tie_word_embeddings=True,
        quantize_kv=False,
        quantized=False,
        **kwargs
    ):
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.d_latent = d_latent
        self.num_experts = num_experts
        self.num_shared_experts = num_shared_experts
        self.top_k = top_k
        self.num_mtp_heads = num_mtp_heads
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.moe_aux_loss_coef = moe_aux_loss_coef
        self.rope_theta = rope_theta
        self.quantize_kv = quantize_kv
        self.quantized = quantized
