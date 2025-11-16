import torch

def safe_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    
    if not torch.is_tensor(x):
        raise ValueError("safe_normalize expects a torch.Tensor")

    if x.numel() == 0:
        return x

    x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)

    norm = torch.norm(x, p=2, dim=-1, keepdim=True)

    norm = torch.clamp(norm, min=eps)

    normalized = x / norm

    normalized = torch.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=-1.0)
    
    return normalized

def check_tensor_health(x: torch.Tensor, name: str = "tensor"):
    print(f"\n--- {name} Health Check ---")
    print(f"Shape: {x.shape}")
    print(f"NaN: {torch.isnan(x).any().item()}")
    print(f"Inf: {torch.isinf(x).any().item()}")
    print(f"Zero: {(x == 0).all().item()}")
    print(f"Min: {x.min().item():.6f}")
    print(f"Max: {x.max().item():.6f}")
    print(f"Mean: {x.mean().item():.6f}")
    print(f"Std: {x.std().item():.6f}")