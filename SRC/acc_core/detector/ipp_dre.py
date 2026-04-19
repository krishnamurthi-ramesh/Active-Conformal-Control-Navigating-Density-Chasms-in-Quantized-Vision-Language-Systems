import torch
import numpy as np
from .rff_kernel import RandomFourierFeatures

class IncrementalDriftTracker:
    """
    Incremental Projection Pursuit Density Ratio Estimator (i-ppDRE).
    Tracks the drift score w(x) = alpha^T * phi(x).
    
    [UAI 2026 CRITICAL FIX]:
    - Orthogonalized Projection: Prevents noise amplification from 4-bit quantization
    - Simplified Scoring: Returns to original w(x) = alpha . phi(x) formulation
    """
    def __init__(self, input_dim=128, rff_dim=512, alpha_lambda=0.99, device='cpu'):
        self.device = device
        
        # 1. Feature Mapper with Orthogonalized Projection
        self.rff = RandomFourierFeatures(input_dim, rff_dim, device=device)
        self.feature_dim = 2 * rff_dim
        
        # 2. Trainable Weights (The Density Ratio)
        # Random initialization breaks symmetry to prevent 0.000 drift score
        self.alpha = torch.randn(self.feature_dim, device=device) * 0.01

        # 3. Forgetting Factor (Lambda)
        # 0.99 means effective memory window of ~100 steps
        self.alpha_lambda = alpha_lambda
        
        # 4. Calibration Stats (Mean of the Teacher)
        self.teacher_mean_phi = torch.zeros(self.feature_dim, device=device)

    def update_teacher_baseline(self, z_batch):
        """Called once during Phase 1 using real teacher activations or synthetic calibration."""
        z_batch = z_batch.to(self.device)
        with torch.no_grad():
            phis = self.rff(z_batch)
            self.teacher_mean_phi = torch.mean(phis, dim=0)

    def score(self, z_t):
        """
        Calculate Drift Score w(x) for a single step.
        Latency: O(R) ~ microseconds
        
        [UAI 2026 FIX] Using orthogonalized RFF projection ensures  
        only semantic drift is detected, not quantization noise.
        """
        z_t = torch.as_tensor(z_t, device=self.device, dtype=torch.float32)
        with torch.no_grad():
            phi_t = self.rff(z_t).squeeze()
            
            # The drift score is the dot product of weights and features
            # [UAI 2026 FIX] Use abs() to capture magnitude of deviation
            w_x = torch.abs(torch.dot(self.alpha, phi_t))
            return w_x.item()

    def update(self, z_t):
        """
        Update weights alpha based on new observation z_t.
        Implements Online Gradient Descent to minimize KL Divergence.
        """
        z_t = torch.as_tensor(z_t, device=self.device, dtype=torch.float32)
        with torch.no_grad():
            phi_t = self.rff(z_t).squeeze()
            
            # Error = (Current_Observation - Ground_Truth_Teacher)
            # This pushes alpha to highlight directions where Student != Teacher
            error = phi_t - self.teacher_mean_phi
            
            # Gradient Step (PEGASOS-style update)
            learning_rate = 0.05
            self.alpha = self.alpha_lambda * self.alpha - learning_rate * error
            
            # Normalization (Constraint: Expectation must be approx 1)
            # Simple L2 clamp to prevent explosion during instability
            norm = torch.norm(self.alpha)
            if norm > 10.0:
                self.alpha = self.alpha / (norm + 1e-9)
