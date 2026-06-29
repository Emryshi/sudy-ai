import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    """
    LoRA (Low-Rank Adaptation) wrapper for linear layers.
    Freezes original linear layer weights and introduces trainable rank-R matrices.
    """
    def __init__(self, base_layer: nn.Module, r: int = 8, lora_alpha: int = 16):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        
        in_features = base_layer.in_features
        out_features = base_layer.out_features
        
        # Freeze base layer parameters
        for p in self.base_layer.parameters():
            p.requires_grad = False
            
        # Introduce low-rank matrices
        self.lora_A = nn.Parameter(torch.empty((in_features, r)))
        self.lora_B = nn.Parameter(torch.empty((r, out_features)))
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)  # Start with zero so LoRA output is initially zero

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Standard projection output
        base_out = self.base_layer(x)
        
        # LoRA projection: x @ lora_A @ lora_B
        lora_out = (x @ self.lora_A) @ self.lora_B
        
        # Combined output scaled by coefficient
        return base_out + lora_out * self.scaling


def inject_lora(model: nn.Module, r: int = 8, lora_alpha: int = 16) -> nn.Module:
    """
    Recursively injects LoRA layers into model attention projections (W_q, W_qr, W_kr)
    and freezes all other parameters for parameter-efficient fine-tuning (PEFT).
    """
    # 1. Freeze all parameters first
    for p in model.parameters():
        p.requires_grad = False
        
    # 2. Inject LoRA adapters
    # We target W_q, W_qr, W_kr attention projections
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if child_name in ["W_q", "W_qr", "W_kr"] and isinstance(child, nn.Linear):
                # Wrap with LoRALinear
                lora_layer = LoRALinear(child, r=r, lora_alpha=lora_alpha)
                setattr(module, child_name, lora_layer)
                
    print(f"LoRA adapters (r={r}) injected successfully into model attention projections.")
    
    # Print trainable vs total parameters count
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.4f}%)")
    
    return model
