"""Profiling and performance tracking for MCMC moves."""

import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from contextlib import contextmanager


# Use perf_counter instead of time.time() for better precision and lower overhead
try:
    from time import perf_counter
except ImportError:
    from time import time as perf_counter


@dataclass
class OperationTimings:
    """Track timing for a specific operation type."""
    
    op_name: str
    total_time: float = 0.0
    call_count: int = 0
    
    def add_time(self, elapsed: float) -> None:
        """Record a single operation execution."""
        self.total_time += elapsed
        self.call_count += 1
    
    @property
    def avg_time(self) -> float:
        """Average time per call (milliseconds)."""
        return (self.total_time / self.call_count * 1000) if self.call_count > 0 else 0.0
    
    def __str__(self) -> str:
        return (
            f"{self.op_name:40s} | "
            f"Calls: {self.call_count:8d} | "
            f"Time/call: {self.avg_time:10.4f}ms | "
            f"Total: {self.total_time:10.3f}s"
        )


@dataclass
class MoveTimings:
    """Track timing and performance statistics for a specific move type."""
    
    move_name: str
    total_time: float = 0.0
    call_count: int = 0
    accept_count: int = 0
    proposal_count: int = 0
    operations: Dict[str, OperationTimings] = field(default_factory=dict)
    
    def add_timing(self, elapsed: float, accepted: bool = False, proposals: int = 1, accepts: int = 0) -> None:
        """Record a single move execution."""
        self.total_time += elapsed
        self.call_count += 1
        self.proposal_count += proposals
        self.accept_count += accepts
        if accepted:
            self.accept_count += 1
    
    @property
    def avg_time(self) -> float:
        """Average time per call (milliseconds)."""
        return (self.total_time / self.call_count * 1000) if self.call_count > 0 else 0.0
    
    @property
    def acceptance_rate(self) -> float:
        """Overall acceptance rate."""
        return (self.accept_count / self.proposal_count) if self.proposal_count > 0 else 0.0
    
    def __str__(self) -> str:
        return (
            f"{self.move_name:25s} | "
            f"Calls: {self.call_count:6d} | "
            f"Time/call: {self.avg_time:8.4f}ms | "
            f"Accept rate: {self.acceptance_rate:7.1%} | "
            f"Total: {self.total_time:8.3f}s"
        )


class MCMCProfiler:
    """Track timing and performance metrics for MCMC operations.
    
    Profiling Levels:
    - LEVEL 0 (OFF): No profiling overhead
    - LEVEL 1 (FAST): Only stage-level timings (minimal overhead)
    - LEVEL 2 (DETAILED): Stage + move timings (moderate overhead)
    - LEVEL 3 (COMPREHENSIVE): Add detailed operation tracking
    
    Use profile_level to select efficiency vs detail tradeoff.
    """
    
    # Profiling level constants
    OFF = 0
    FAST = 1
    DETAILED = 2
    COMPREHENSIVE = 3
    
    def __init__(self, profile_level: int = DETAILED, sampling_rate: float = 1.0):
        """Initialize profiler.
        
        Args:
            profile_level: Profiling detail level (0=OFF, 1=FAST, 2=DETAILED, 3=COMPREHENSIVE)
            sampling_rate: Sample only this fraction of operations (0-1). 
                          Useful for reducing overhead with frequent operations.
        """
        self.profile_level = profile_level
        self.sampling_rate = sampling_rate
        self.move_times: Dict[str, MoveTimings] = {}
        self.stage_times: Dict[str, float] = {}  # stage -> total time
        self.stage_calls: Dict[str, int] = {}    # stage -> call count
        self.global_operations: Dict[str, OperationTimings] = {}  # global operation times
        self.enabled = profile_level > self.OFF
        self._current_move: Optional[str] = None
        self._call_counter = 0  # For sampling
    
    def reset(self) -> None:
        """Clear all timing data."""
        self.move_times.clear()
        self.stage_times.clear()
        self.stage_calls.clear()
        self.global_operations.clear()
        self._current_move = None
        self._call_counter = 0
    
    def _should_sample(self) -> bool:
        """Check if this call should be profiled based on sampling rate."""
        if self.sampling_rate >= 1.0:
            return True
        self._call_counter += 1
        # Simple sampling: record every 1/sampling_rate calls
        if self._call_counter >= int(1.0 / self.sampling_rate):
            self._call_counter = 0
            return True
        return False
    
    def record_move(
        self, 
        move_name: str, 
        elapsed: float, 
        accepted: bool = False,
        proposals: int = 1,
        accepts: int = 0
    ) -> None:
        """Record timing for a block move.
        
        Only records if profile_level >= DETAILED.
        """
        # Early exit if not tracking moves (LEVEL 1 and below skip this)
        if not self.enabled or self.profile_level < self.DETAILED:
            return
        
        # Fast exit with sampling
        if not self._should_sample():
            return
        
        if move_name not in self.move_times:
            self.move_times[move_name] = MoveTimings(move_name)
        
        self.move_times[move_name].add_timing(elapsed, accepted, proposals, accepts)
    
    def record_stage(self, stage_name: str, elapsed: float) -> None:
        """Record timing for a MCMC stage (update_z, update_tau, etc.).
        
        Only records if profile_level >= FAST.
        """
        if not self.enabled or self.profile_level < self.FAST:
            return
        
        if stage_name not in self.stage_times:
            self.stage_times[stage_name] = 0.0
            self.stage_calls[stage_name] = 0
        
        self.stage_times[stage_name] += elapsed
        self.stage_calls[stage_name] += 1
    
    def record_operation(self, op_name: str, elapsed: float, move_name: Optional[str] = None) -> None:
        """Record timing for a detailed operation (distance calc, Z* calc, etc.).
        
        Only records if profile_level >= COMPREHENSIVE.
        """
        if not self.enabled or self.profile_level < self.COMPREHENSIVE:
            return
        
        if not self._should_sample():
            return
        
        # Global operation tracking
        if op_name not in self.global_operations:
            self.global_operations[op_name] = OperationTimings(op_name)
        self.global_operations[op_name].add_time(elapsed)
        
        # Per-move operation tracking (if we're inside a move)
        if move_name or self._current_move:
            move = move_name or self._current_move
            if move and move in self.move_times:
                if op_name not in self.move_times[move].operations:
                    self.move_times[move].operations[op_name] = OperationTimings(op_name)
                self.move_times[move].operations[op_name].add_time(elapsed)
    
    @contextmanager
    def operation(self, op_name: str, move_name: Optional[str] = None):
        """Context manager for timing an operation."""
        if not self.enabled or self.profile_level < self.COMPREHENSIVE:
            yield
            return
        
        t_start = perf_counter()
        try:
            yield
        finally:
            elapsed = perf_counter() - t_start
            self.record_operation(op_name, elapsed, move_name)
    
    @contextmanager
    def move_context(self, move_name: str):
        """Context manager to mark which move we're currently in."""
        if not self.enabled or self.profile_level < self.COMPREHENSIVE:
            yield
            return
        
        old_move = self._current_move
        self._current_move = move_name
        try:
            yield
        finally:
            self._current_move = old_move
    
    def get_move_summary(self) -> str:
        """Return a formatted summary of move timings."""
        if not self.move_times:
            return "No move timings recorded."
        
        lines = ["\n" + "="*120]
        lines.append("MOVE PERFORMANCE SUMMARY")
        lines.append("="*120)
        lines.append(f"{'Move Type':25s} | {'Calls':>8s} | {'Time/call':>12s} | {'Accept Rate':>14s} | {'Total Time':>12s}")
        lines.append("-"*120)
        
        total_time = sum(m.total_time for m in self.move_times.values())
        total_calls = sum(m.call_count for m in self.move_times.values())
        
        for move in sorted(self.move_times.values(), key=lambda x: x.total_time, reverse=True):
            lines.append(str(move))
        
        lines.append("-"*120)
        lines.append(f"{'TOTAL':25s} | {total_calls:8d} | {total_time/total_calls*1000:11.4f}ms | {'':14s} | {total_time:11.3f}s")
        lines.append("="*120)
        return "\n".join(lines)
    
    def get_move_detail_summary(self) -> str:
        """Return detailed operation breakdown per move type."""
        if not self.move_times or not any(m.operations for m in self.move_times.values()):
            return "No detailed operation timings available."
        
        lines = ["\n" + "="*120]
        lines.append("DETAILED OPERATION BREAKDOWN BY MOVE")
        lines.append("="*120)
        
        for move in sorted(self.move_times.values(), key=lambda x: x.total_time, reverse=True):
            if not move.operations:
                continue
            
            lines.append(f"\n{move.move_name.upper()} ({move.call_count} calls, {move.total_time:.3f}s total)")
            lines.append("-"*120)
            
            move_total = move.total_time
            for op in sorted(move.operations.values(), key=lambda x: x.total_time, reverse=True):
                pct = (op.total_time / move_total * 100) if move_total > 0 else 0
                lines.append(f"  {op.op_name:38s} | {op.call_count:8d} | {op.avg_time:10.4f}ms | {pct:6.1f}% | {op.total_time:10.3f}s")
        
        lines.append("="*120)
        return "\n".join(lines)
    
    def get_stage_summary(self) -> str:
        """Return a formatted summary of stage timings."""
        if not self.stage_times:
            return "No stage timings recorded."
        
        lines = ["\n" + "="*90]
        lines.append("STAGE PERFORMANCE SUMMARY")
        lines.append("="*90)
        lines.append(f"{'Stage':25s} | {'Calls':>8s} | {'Time/call':>12s} | {'Total Time':>12s}")
        lines.append("-"*90)
        
        total_time = sum(self.stage_times.values())
        
        for stage in sorted(self.stage_times.keys(), key=lambda x: self.stage_times[x], reverse=True):
            calls = self.stage_calls[stage]
            time_per_call = self.stage_times[stage] / calls if calls > 0 else 0
            lines.append(
                f"{stage:25s} | {calls:8d} | {time_per_call*1000:11.4f}ms | {self.stage_times[stage]:11.3f}s"
            )
        
        lines.append("-"*90)
        lines.append(f"{'TOTAL':25s} | {'':8s} | {'':12s} | {total_time:11.3f}s")
        lines.append("="*90)
        return "\n".join(lines)
    
    def get_operation_summary(self) -> str:
        """Return a formatted summary of global operation timings."""
        if not self.global_operations:
            return "No operation timings recorded."
        
        lines = ["\n" + "="*120]
        lines.append("GLOBAL OPERATION PERFORMANCE SUMMARY")
        lines.append("="*120)
        lines.append(str(OperationTimings("Operation")))
        lines.append("-"*120)
        
        total_time = sum(o.total_time for o in self.global_operations.values())
        
        for op in sorted(self.global_operations.values(), key=lambda x: x.total_time, reverse=True):
            pct = (op.total_time / total_time * 100) if total_time > 0 else 0
            name_with_pct = f"{op.op_name} [{pct:.1f}%]"
            line_str = (
                f"{name_with_pct:55s} | "
                f"Calls: {op.call_count:8d} | "
                f"Time/call: {op.avg_time:10.4f}ms | "
                f"Total: {op.total_time:10.3f}s"
            )
            lines.append(line_str)
        
        lines.append("-"*120)
        lines.append(f"{'TOTAL':55s} | {'':10s} | {'':15s} | {total_time:10.3f}s")
        lines.append("="*120)
        return "\n".join(lines)
    
    def get_full_summary(self) -> str:
        """Return combined summary of moves, stages, and operations."""
        parts = []
        if self.move_times:
            parts.append(self.get_move_summary())
        if self.global_operations:
            parts.append(self.get_operation_summary())
        if self.move_times:
            parts.append(self.get_move_detail_summary())
        if self.stage_times:
            parts.append(self.get_stage_summary())
        
        if not parts:
            return "No profiling data available."
        
        return "\n".join(parts)
    
    def print_summary(self) -> None:
        """Print the full profiling summary."""
        print(self.get_full_summary())
    
    def get_slowest_moves(self, top_n: int = 5) -> List[tuple]:
        """Return the top N slowest moves (move_name, total_time)."""
        if not self.move_times:
            return []
        
        return sorted(
            [(m.move_name, m.total_time) for m in self.move_times.values()],
            key=lambda x: x[1],
            reverse=True
        )[:top_n]
    
    def get_slowest_operations(self, top_n: int = 5) -> List[tuple]:
        """Return the top N slowest operations (op_name, total_time)."""
        if not self.global_operations:
            return []
        
        return sorted(
            [(o.op_name, o.total_time) for o in self.global_operations.values()],
            key=lambda x: x[1],
            reverse=True
        )[:top_n]
    
    def get_slowest_stages(self, top_n: int = 5) -> List[tuple]:
        """Return the top N slowest stages (stage_name, total_time)."""
        if not self.stage_times:
            return []
        
        return sorted(
            [(s, t) for s, t in self.stage_times.items()],
            key=lambda x: x[1],
            reverse=True
        )[:top_n]


# Global profiler instance
_global_profiler: Optional[MCMCProfiler] = None


def enable_profiling(profile_level: int = MCMCProfiler.DETAILED, sampling_rate: float = 1.0) -> MCMCProfiler:
    """Enable global profiling with specified detail level.
    
    Args:
        profile_level: Detail level (0=OFF, 1=FAST, 2=DETAILED, 3=COMPREHENSIVE)
        sampling_rate: Fraction of operations to profile (0-1). Useful for reducing overhead.
    
    Returns:
        The global MCMCProfiler instance.
    """
    global _global_profiler
    _global_profiler = MCMCProfiler(profile_level=profile_level, sampling_rate=sampling_rate)
    return _global_profiler


def disable_profiling() -> None:
    """Disable global profiling."""
    global _global_profiler
    if _global_profiler is not None:
        _global_profiler.enabled = False


def get_profiler() -> Optional[MCMCProfiler]:
    """Get the global profiler instance (or None if not enabled)."""
    return _global_profiler


def reset_profiling() -> None:
    """Reset the global profiler data."""
    global _global_profiler
    if _global_profiler is not None:
        _global_profiler.reset()

