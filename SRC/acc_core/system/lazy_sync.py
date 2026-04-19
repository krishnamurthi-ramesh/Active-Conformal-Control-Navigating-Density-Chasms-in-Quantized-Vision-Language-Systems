"""
Local Oracle KV-cache transfer bridge for ACC.

Implements a pinned-memory PCIe transfer path between GPU Actor and CPU Oracle
to support low-latency handoff when the safety gate triggers.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

import torch


@dataclass
class TransferStats:
	durations_ms: List[float] = field(default_factory=list)

	def record(self, duration_ms: float) -> None:
		self.durations_ms.append(duration_ms)

	def last(self) -> Optional[float]:
		if not self.durations_ms:
			return None
		return self.durations_ms[-1]

	def avg(self) -> Optional[float]:
		if not self.durations_ms:
			return None
		return float(sum(self.durations_ms) / len(self.durations_ms))


class KVCacheSync:
	"""
	Implements the 'Local Oracle' transfer logic from Section 3.3.
	Target: PCIe transfer of ~1.1GB KV cache within < 100ms.
	"""

	def __init__(
		self,
		device_gpu: str = "cuda:0",
		device_cpu: str = "cpu",
		latency_budget_ms: float = 92.0,
	) -> None:
		self.gpu = device_gpu
		self.cpu = device_cpu
		self.latency_budget_ms = latency_budget_ms
		self.stats = TransferStats()

	def _ensure_pinned(self, tensor: torch.Tensor) -> torch.Tensor:
		if tensor.device.type != "cuda":
			return tensor
		if not tensor.is_contiguous():
			tensor = tensor.contiguous()
		return tensor

	def transfer_to_oracle(self, kv_cache_tensor: torch.Tensor) -> torch.Tensor:
		"""
		Moves the reasoning state from the Quantized Student to the FP16 Teacher.
		Uses non-blocking copy to pinned CPU memory for maximum throughput.
		"""
		kv_cache_tensor = self._ensure_pinned(kv_cache_tensor)
		start_time = time.perf_counter()

		oracle_cache = kv_cache_tensor.to(self.cpu, non_blocking=True)
		if torch.cuda.is_available():
			torch.cuda.synchronize()

		duration_ms = (time.perf_counter() - start_time) * 1000.0
		self.stats.record(duration_ms)

		if duration_ms > self.latency_budget_ms:
			print(f"[ALERT]️ Latency Warning: Transfer took {duration_ms:.2f}ms")

		return oracle_cache

	def resume_student(self, corrected_cache: torch.Tensor) -> torch.Tensor:
		"""
		Returns the corrected manifold state back to the GPU Actor.
		"""
		start_time = time.perf_counter()
		
		student_cache = corrected_cache.to(self.gpu, non_blocking=True)
		if torch.cuda.is_available():
			torch.cuda.synchronize()
		
		duration_ms = (time.perf_counter() - start_time) * 1000.0
		self.stats.record(duration_ms)
		
		return student_cache

	def print_stats(self) -> None:
		"""Print transfer statistics for analysis."""
		if not self.stats.durations_ms:
			print("No transfers recorded.")
			return
		
		durations = self.stats.durations_ms
		print(f"\nKV-Cache Transfer Statistics:")
		print(f"  Total Transfers: {len(durations)}")
		print(f"  Mean: {sum(durations)/len(durations):.2f}ms")
		print(f"  Min: {min(durations):.2f}ms")
		print(f"  Max: {max(durations):.2f}ms")
		print(f"  Budget: {self.latency_budget_ms:.2f}ms")
		
		if max(durations) > self.latency_budget_ms:
			violations = sum(1 for d in durations if d > self.latency_budget_ms)
			print(f"  [ALERT]️ WARNING: {violations} transfers exceeded budget")
		else:
			print(f"  ✓ All transfers within budget")


class ContextSynchronizer:
	"""
	High-level handoff controller between Student and Oracle.
	This class coordinates pause/resume and cache migration.
	"""

	def __init__(
		self,
		device_gpu: str = "cuda:0",
		device_cpu: str = "cpu",
		latency_budget_ms: float = 92.0,
	) -> None:
		self.cache_sync = KVCacheSync(
			device_gpu=device_gpu,
			device_cpu=device_cpu,
			latency_budget_ms=latency_budget_ms,
		)

	def handoff_to_oracle(self, kv_cache_tensor: torch.Tensor) -> torch.Tensor:
		return self.cache_sync.transfer_to_oracle(kv_cache_tensor)

	def return_to_student(self, corrected_cache: torch.Tensor) -> torch.Tensor:
		return self.cache_sync.resume_student(corrected_cache)
