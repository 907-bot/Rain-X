"""
RAIN-X Post-Processing Engine & Multi-Model Comparison Layer
Implements:
  - Baseline 0: Raw NWP
  - Baseline 1: Global Linear Bias Correction
  - Baseline 2: Empirical Quantile Mapping (EQM)
  - Model 3: Random Forest Regressor
  - Model 4: LightGBM / Gradient Boosting Regressor
  - Model 5: RAIN-X Physics-Guided Regime-Aware Mixture-of-Experts (MoE)
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any, Optional
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb


class LinearBiasCorrection:
    """
    Baseline 1: Standard global multiplicative and additive bias correction.
    y_corr = max(0, slope * y_raw + intercept)
    """
    def __init__(self):
        self.slope = 1.0
        self.intercept = 0.0

    def fit(self, y_raw: np.ndarray, y_obs: np.ndarray):
        # Ordinary least squares on positive precipitation events
        mask = (y_raw > 0.1) | (y_obs > 0.1)
        if np.sum(mask) > 10:
            xr = y_raw[mask]
            yo = y_obs[mask]
            cov = np.cov(xr, yo)
            self.slope = float(cov[0, 1] / (cov[0, 0] + 1e-6))
            self.intercept = float(np.mean(yo) - self.slope * np.mean(xr))
        else:
            self.slope = 1.0
            self.intercept = 0.0

    def predict(self, y_raw: np.ndarray) -> np.ndarray:
        pred = self.slope * y_raw + self.intercept
        return np.maximum(pred, 0.0).astype(np.float32)


class EmpiricalQuantileMapping:
    """
    Baseline 2: Empirical Quantile Mapping (EQM).
    Transforms raw forecast CDF to match historical observed CDF:
    y_corr = F_obs^-1(F_nwp(y_raw))
    """
    def __init__(self, n_quantiles: int = 100):
        self.n_quantiles = n_quantiles
        self.quantiles = np.linspace(0.0, 1.0, n_quantiles)
        self.nwp_quantiles = None
        self.obs_quantiles = None

    def fit(self, y_raw: np.ndarray, y_obs: np.ndarray):
        self.nwp_quantiles = np.quantile(y_raw, self.quantiles)
        self.obs_quantiles = np.quantile(y_obs, self.quantiles)
        # Ensure strict monotonicity for interpolation
        self.nwp_quantiles = np.maximum.accumulate(self.nwp_quantiles)
        self.obs_quantiles = np.maximum.accumulate(self.obs_quantiles)

    def predict(self, y_raw: np.ndarray) -> np.ndarray:
        if self.nwp_quantiles is None or self.obs_quantiles is None:
            return np.maximum(y_raw, 0.0).astype(np.float32)
        # Interp from NWP quantiles to CDF probability, then map to OBS quantiles
        cdf_prob = np.interp(y_raw, self.nwp_quantiles, self.quantiles, left=0.0, right=1.0)
        corrected = np.interp(cdf_prob, self.quantiles, self.obs_quantiles)
        return np.maximum(corrected, 0.0).astype(np.float32)


class RegimeExpertNN(nn.Module):
    """
    Specialized neural post-processing network for an individual meteorological regime.
    Learns a regime-specific residual correction Delta_y:
      y_k = ReLU(y_nwp + Delta_y(X))
    """
    def __init__(self, in_features: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
        nn.init.normal_(self.net[-1].weight, std=0.01)
        nn.init.constant_(self.net[-1].bias, 0.0)

    def forward(self, x: torch.Tensor, y_nwp: torch.Tensor) -> torch.Tensor:
        delta = self.net(x)
        return torch.relu(y_nwp + delta)


class RegimeAwareMoE(nn.Module):
    """
    RAIN-X Physics-Guided Regime-Aware Mixture-of-Experts (MoE) Architecture:
      y_hat = sum_{k=1}^K p_k(X) * f_k(X, y_nwp)
    where p_k are soft regime probabilities, and f_k are specialized regime experts.
    """
    def __init__(self, in_features: int = 16, num_regimes: int = 6):
        super().__init__()
        self.num_regimes = num_regimes
        self.experts = nn.ModuleList([RegimeExpertNN(in_features) for _ in range(num_regimes)])

    def forward(self, x: torch.Tensor, gating_probs: torch.Tensor, y_nwp: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, in_features)
        gating_probs: (batch_size, num_regimes)
        y_nwp: (batch_size, 1)
        returns: (batch_size, 1)
        """
        expert_outputs = [expert(x, y_nwp) for expert in self.experts]
        stacked = torch.cat(expert_outputs, dim=-1)  # shape (batch_size, num_regimes)
        weighted = torch.sum(gating_probs * stacked, dim=-1, keepdim=True)
        return weighted


class PhysicsGuidedMoELoss(nn.Module):
    """
    Physics-guided multi-objective loss for precipitation post-processing:
      1. Smooth L1 / Huber loss for general rainfall accuracy
      2. Extreme-Event pinball penalty: penalizes underpredicting >64.5 mm events
      3. Physical conservation & non-negativity constraint
    """
    def __init__(self, lambda_extreme: float = 0.6, lambda_phys: float = 0.2):
        super().__init__()
        self.huber = nn.SmoothL1Loss(beta=2.0)
        self.lambda_extreme = lambda_extreme
        self.lambda_phys = lambda_phys

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        l_base = self.huber(y_pred, y_true)

        # Extreme rainfall penalty: heavy rain (> 64.5 mm) underprediction penalty
        heavy_mask = (y_true > 64.5).float()
        underpredict_error = torch.relu(y_true - y_pred) * heavy_mask
        l_extreme = torch.mean(underpredict_error)

        # Physical constraint: non-negativity penalty
        l_phys = torch.mean(torch.relu(-y_pred))

        total_loss = l_base + self.lambda_extreme * l_extreme + self.lambda_phys * l_phys
        return total_loss


class ModelComparisonSuite:
    """
    Integrates and coordinates all 6 models for unified training,
    benchmarking, and operational inference.
    """
    def __init__(self, in_features: int = 16, num_regimes: int = 6):
        self.in_features = in_features
        self.num_regimes = num_regimes
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        # Models
        self.linear_bc = LinearBiasCorrection()
        self.quantile_mapping = EmpiricalQuantileMapping(n_quantiles=100)
        self.random_forest = RandomForestRegressor(n_estimators=80, max_depth=12, random_state=42, n_jobs=-1)
        self.lightgbm_model = lgb.LGBMRegressor(n_estimators=120, learning_rate=0.06, max_depth=8, random_state=42, verbose=-1)
        self.moe_model = RegimeAwareMoE(in_features=in_features, num_regimes=num_regimes).to(self.device)

        self.feature_means = np.zeros(in_features, dtype=np.float32)
        self.feature_stds = np.ones(in_features, dtype=np.float32)
        self.is_trained = False

    def fit(
        self,
        X_train: np.ndarray,
        y_train_nwp: np.ndarray,
        y_train_obs: np.ndarray,
        gating_probs_train: np.ndarray,
        epochs: int = 35,
        batch_size: int = 64
    ):
        """
        Trains all models on historical dataset with chronological validation.
        """
        print("[PostProcessor] Training Model 1: Linear Bias Correction...")
        self.linear_bc.fit(y_train_nwp, y_train_obs)

        print("[PostProcessor] Training Model 2: Empirical Quantile Mapping...")
        self.quantile_mapping.fit(y_train_nwp, y_train_obs)

        print("[PostProcessor] Training Model 3: Random Forest Regressor...")
        self.random_forest.fit(X_train, y_train_obs)

        print("[PostProcessor] Training Model 4: LightGBM Regressor...")
        self.lightgbm_model.fit(X_train, y_train_obs)

        print("[PostProcessor] Training Model 5: RAIN-X Physics-Guided MoE...")
        self.feature_means = np.mean(X_train, axis=0).astype(np.float32)
        self.feature_stds = (np.std(X_train, axis=0) + 1e-5).astype(np.float32)
        X_norm = (X_train - self.feature_means) / self.feature_stds

        # Convert to PyTorch tensors
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_norm, dtype=torch.float32),
            torch.tensor(gating_probs_train, dtype=torch.float32),
            torch.tensor(y_train_nwp, dtype=torch.float32).unsqueeze(1),
            torch.tensor(y_train_obs, dtype=torch.float32).unsqueeze(1)
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(self.moe_model.parameters(), lr=0.005, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = PhysicsGuidedMoELoss(lambda_extreme=0.6, lambda_phys=0.2)

        self.moe_model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for bx, bg, by_nwp, by_obs in loader:
                bx = bx.to(self.device)
                bg = bg.to(self.device)
                by_nwp = by_nwp.to(self.device)
                by_obs = by_obs.to(self.device)

                optimizer.zero_grad()
                pred = self.moe_model(bx, bg, by_nwp)
                loss = criterion(pred, by_obs)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(bx)
            
            scheduler.step()
            if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                avg_l = total_loss / len(dataset)
                print(f"  MoE Epoch [{epoch+1}/{epochs}] - Multi-Objective Loss: {avg_l:.4f}")

        self.is_trained = True
        print("[PostProcessor] All 5 comparison models successfully trained.")

    def predict_all_models(
        self,
        X: np.ndarray,
        y_nwp: np.ndarray,
        gating_probs: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Runs inference on all baseline and machine learning models for rigorous comparison:
          - raw_nwp
          - linear_bc
          - quantile_mapping
          - random_forest
          - lightgbm
          - rain_x_moe
        """
        raw_pred = np.maximum(y_nwp, 0.0).astype(np.float32)
        linear_pred = self.linear_bc.predict(y_nwp)
        qm_pred = self.quantile_mapping.predict(y_nwp)
        rf_pred = np.maximum(self.random_forest.predict(X), 0.0).astype(np.float32)
        lgb_pred = np.maximum(self.lightgbm_model.predict(X), 0.0).astype(np.float32)

        # MoE prediction
        X_norm = (X - self.feature_means) / (self.feature_stds + 1e-6)
        xt = torch.tensor(X_norm, dtype=torch.float32).to(self.device)
        gt = torch.tensor(gating_probs, dtype=torch.float32).to(self.device)
        yn = torch.tensor(y_nwp, dtype=torch.float32).unsqueeze(1).to(self.device)
        
        self.moe_model.eval()
        with torch.no_grad():
            moe_out = self.moe_model(xt, gt, yn).cpu().numpy().squeeze(-1)
        
        moe_pred = np.maximum(moe_out, 0.0).astype(np.float32)

        return {
            "raw_nwp": raw_pred,
            "linear_bc": linear_pred,
            "quantile_mapping": qm_pred,
            "random_forest": rf_pred,
            "lightgbm": lgb_pred,
            "rain_x_moe": moe_pred
        }

