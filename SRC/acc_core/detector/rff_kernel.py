import torch
import torch.nn as nn
import numpy as np

class RandomFourierFeatures(nn.Module):
    """
    Implements the RFF map for O(1) kernel approximation.
    Maps input z (dim D) -> phi(z) (dim 2*R)
    
    [UAI 2026 FIX] Orthogonalized Projection to eliminate hardware bias
    and ensure only semantic drift is detected, not numerical noise.
    """
    def __init__(self, input_dim=128, rff_dim=512, sigma=1.0, device='cpu'):
        super().__init__()
        self.input_dim = input_dim
        self.rff_dim = rff_dim
        self.sigma = sigma
        self.device = device
        
        # 1. Initialize Orthogonalized Random Projection Matrix Omega
        # [FIX] Use QR decomposition to ensure orthonormality
        # This prevents noise amplification from 4-bit quantization
        # Target shape: omega = (rff_dim, input_dim) for z @ omega.T
        # Example: (512, 4096) so that (batch, 4096) @ (4096, 512) = (batch, 512)
        
        random_matrix = torch.randn(rff_dim, input_dim, device=device) * (1.0 / sigma)
        
        # For rff_dim < input_dim (e.g., 512 < 4096):
        # QR on (512, 4096) gives Q of shape (512, 512), R of shape (512, 4096)
        # We want the full (512, 4096) matrix, so we use the random_matrix directly
        # but orthogonalize its rows
        if rff_dim < input_dim:
            # Orthogonalize rows: transpose, QR, transpose back
            Q, _ = torch.linalg.qr(random_matrix.T)  # Q: (4096, 512)
            self.omega = Q.T  # (512, 4096)
        else:
            # rff_dim >= input_dim
            Q, _ = torch.linalg.qr(random_matrix)
            self.omega = Q[:, :input_dim]
        
        self.omega = self.omega.to(device)
        
        # 2. Random Bias (0 to 2pi for shift invariance)
        self.bias = torch.rand(rff_dim, device=device) * 2 * np.pi
        
        # Freezing weights ensures the map is deterministic
        self.omega.requires_grad = False
        self.bias.requires_grad = False

    def forward(self, z):
        """
        Compute phi(z) = [cos(Omega @ z + b), sin(Omega @ z + b)]
        Input z: [Batch, D] or [D]
        Output: [Batch, 2*R]
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
            
        # Projection: [Batch, R]
        # We assume input z is already on the correct device
        proj = torch.matmul(z, self.omega.T) + self.bias
        
        # Feature Map: [Batch, 2*R]
        # We concatenate cos and sin to approximate the Gaussian Kernel
        phi = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
        
        # Normalize by 1/sqrt(R) to maintain unit variance expectation
        return phi / np.sqrt(self.rff_dim)
