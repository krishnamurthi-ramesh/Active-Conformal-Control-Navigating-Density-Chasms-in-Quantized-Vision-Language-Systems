"""
Conformal Safety Gate — UAI 2026

Implements the Safety Gate using Conformal Prediction with
Bayesian Quadrature (BQ) extension per Snell & Griffiths (ICML 2025).

Two complementary signals:
1. Drift score w(x) > lambda* — standard conformal threshold
2. Prediction set expansion |C(x_t)| > κ — BQ proxy for posterior risk

The BQ interpretation (Snell-Griffiths):
  The conformal prediction set size |C(x_t)| approximates the integral
  ∫ L(y) dP(y|x) — i.e., posterior expected loss. A sudden expansion in
  |C(x_t)| indicates that quantization noise has overwhelmed the reasoning
  signal (low Signal-to-Noise Ratio), triggering the Safety Gate.
"""
import numpy as np
from typing import List, Optional


class ConformalSafetyGate:
    """
    Conformal Safety Gate with Bayesian Quadrature extension.

    Standard split-conformal calibration produces threshold lambda* from
    calibration drift scores.  The BQ extension additionally monitors
    the prediction set size |C(x_t)| during inference and triggers
    intervention when it expands beyond a calibrated bound κ.
    """

    def __init__(
        self,
        epsilon: float = 0.05,
        min_threshold: float = 0.005,
        pred_set_kappa: Optional[float] = None,
    ):
        """
        Args:
            epsilon: Miscoverage rate (1-ε = coverage guarantee).
            min_threshold: Noise-floor for lambda* to prevent zero thresholds.
            pred_set_kappa: If set, override the calibrated κ for |C(x_t)|.
                            Otherwise κ is calibrated from data.
        """
        self.epsilon = epsilon
        self.min_threshold = min_threshold
        self.pred_set_kappa = pred_set_kappa

        # Drift-score calibration
        self.calibration_scores: List[float] = []
        self.lambda_star: float = float('inf')

        # BQ / prediction-set calibration
        self.calibration_pred_sets: List[int] = []
        self.kappa: float = float('inf')  # |C(x_t)| threshold

        # Posterior risk tracking (running estimate)
        self._risk_window: List[float] = []
        self._risk_window_size: int = 50

        # ── Leaky Integrator (Velocity-Based Control) ─────────────────────
        # S_t = α * S_{t-1} + (1-α) * w_t
        # Trigger only when S_t > lambda* for `min_consecutive` steps.
        # Alpha 0.70 provides faster response for short-form tasks (POPE).
        # min_consecutive=1 allows first smoothed crossing to trigger.
        self.leaky_alpha: float = 0.70        
        self.min_consecutive: int = 1         
        self._S_t: float = 0.0               
        self._consecutive_above: int = 0      
        # ─────────────────────────────────────────────────────────────────

        self.is_calibrated: bool = False

        # Counters for diagnostics
        self.and_logic_interventions: int = 0
        self.and_logic_non_interventions: int = 0
        self.drift_only_alerts: int = 0
        self.bq_only_alerts: int = 0
        self.non_intervention_log: List[dict] = []

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def add_calibration_score(
        self, w_x: float, pred_set_size: Optional[int] = None
    ):
        """Collect a calibration sample.

        Args:
            w_x: Drift score from i-ppDRE.
            pred_set_size: |C(x_t)| — number of tokens with p > floor.
        """
        self.calibration_scores.append(w_x)
        if pred_set_size is not None:
            self.calibration_pred_sets.append(pred_set_size)

    def calibrate(self) -> float:
        """Calibrate both drift threshold lambda* and prediction-set bound κ.

        Returns:
            lambda* (drift threshold).
        """
        if not self.calibration_scores:
            return self.min_threshold

        # --- 1. Drift-score threshold (standard conformal) ---
        scores = np.sort(self.calibration_scores)
        n = len(scores)

        q_idx = int(np.ceil((n + 1) * (1 - self.epsilon)))
        q_idx = min(q_idx, n - 1)

        raw_lambda = float(scores[q_idx])
        self.lambda_star = max(raw_lambda, self.min_threshold)

        # --- 2. Prediction-set bound κ (BQ extension) ---
        if self.calibration_pred_sets:
            pred_arr = np.sort(self.calibration_pred_sets)
            m = len(pred_arr)
            k_idx = int(np.ceil((m + 1) * (1 - self.epsilon)))
            k_idx = min(k_idx, m - 1)
            cal_kappa = float(pred_arr[k_idx])
            # Use manual override if provided, else calibrated value
            self.kappa = self.pred_set_kappa if self.pred_set_kappa is not None else cal_kappa
        elif self.pred_set_kappa is not None:
            self.kappa = self.pred_set_kappa

        print(f" Conformal Gate Calibrated (N={n}, epsilon={self.epsilon})")
        print(f"   lambda* (drift)     = {self.lambda_star:.5f}")
        if self.calibration_pred_sets:
            print(f"   κ  (|C(x_t)|)  = {self.kappa:.0f}")

        self.is_calibrated = True
        return self.lambda_star

    # ------------------------------------------------------------------
    # Inference-time check
    # ------------------------------------------------------------------

    def check(
        self,
        w_x: float,
        pred_set_size: Optional[int] = None,
        require_both: bool = True,
    ) -> bool:
        """Check if the current state triggers intervention via Leaky Integrator.

        S_t = α * S_{t-1} + (1-α) * w_t

        Trigger condition: S_t > lambda* for `min_consecutive` consecutive steps.
        This filters single-token quantization jitter and identifies sustained
        manifold departure — the true precursor of hallucination.

        Requires calibrate() to be called first.
        """
        if not self.is_calibrated:
            return False

        # 1. Update leaky integrator: S_t = α*S_{t-1} + (1-α)*w_t
        self._S_t = self.leaky_alpha * self._S_t + (1 - self.leaky_alpha) * float(w_x)

        # 2. Check if smoothed signal exceeds threshold
        drift_alert = self._S_t > self.lambda_star

        # 3. Consecutive-step guard — require min_consecutive steps above threshold
        if drift_alert:
            self._consecutive_above += 1
        else:
            self._consecutive_above = 0  # reset on any below-threshold step

        smoothed_trigger = (self._consecutive_above >= self.min_consecutive)

        # 4. Optional BQ (prediction-set) signal
        bq_alert = False
        if pred_set_size is not None and self.kappa < float('inf'):
            bq_alert = pred_set_size > self.kappa

        # Update running posterior risk estimate
        if pred_set_size is not None:
            self._risk_window.append(pred_set_size)
            if len(self._risk_window) > self._risk_window_size:
                self._risk_window.pop(0)

        # 5. Gate decision — drift-only mode by default (OR logic)
        # require_both=True activates the AND gate with BQ signal
        if require_both and pred_set_size is not None and self.kappa < float('inf'):
            should_intervene = smoothed_trigger and bq_alert
            if smoothed_trigger and not bq_alert:
                self.and_logic_non_interventions += 1
                self.non_intervention_log.append({
                    "S_t": self._S_t, "w_x": w_x,
                    "pred_set_size": pred_set_size,
                    "decision": "skip_intervention",
                    "reason": "Integrator above threshold but model not confused (BQ)"
                })
            elif should_intervene:
                self.and_logic_interventions += 1
                self._consecutive_above = 0  # reset after firing
        else:
            # Drift integrator only — primary ACC gate mode
            # This properly identifies sustained manifold departure
            should_intervene = smoothed_trigger
            if should_intervene:
                self.drift_only_alerts += 1
                self._consecutive_above = 0  # reset after firing, new window

        return should_intervene

    def reset_integrator(self):
        """Call between samples to reset the smoothed state."""
        self._S_t = 0.0
        self._consecutive_above = 0


    # ------------------------------------------------------------------
    # BQ Risk Monitoring (for visualization / paper)
    # ------------------------------------------------------------------

    def posterior_risk_estimate(self) -> float:
        """Estimate the current posterior risk R̂ from the sliding window.

        Per Snell-Griffiths: R̂ ≈ mean(|C(x_t)|) / V_max, where V_max
        is the vocabulary size.  A rising R̂ indicates increasing epistemic
        uncertainty in the quantized model.

        Returns:
            Normalized posterior risk ∈ [0, 1].
        """
        if not self._risk_window:
            return 0.0
        return float(np.mean(self._risk_window))

    def get_diagnostics(self) -> dict:
        """Return diagnostic information for logging / plotting."""
        return {
            "lambda_star": self.lambda_star,
            "kappa": self.kappa if self.kappa < float('inf') else None,
            "posterior_risk": self.posterior_risk_estimate(),
            "risk_window_len": len(self._risk_window),
            "is_calibrated": self.is_calibrated,
            "n_calibration_scores": len(self.calibration_scores),
        }
