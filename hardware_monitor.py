#!/usr/bin/env python3
"""
Hardware Safety Monitor for UAI 2026 Campaign
HP Z4 G5 Workstation: RTX 2000 Ada (16GB) + Xeon W5-2565X (64GB RAM)

Features:
- Live terminal display with refresh
- JSON logging for post-analysis
- Safety thresholds to prevent thermal/memory overruns
- Automatic alerts when limits approached
- Designed for 40-60 hour unattended campaign runs

Usage:
    python hardware_monitor.py                  # Live display + JSON log
    python hardware_monitor.py --interval 10    # Custom interval (seconds)
    python hardware_monitor.py --log-dir logs/  # Custom log directory
"""

import os
import sys
import time
import json
import signal
import argparse
import subprocess
import psutil
from pathlib import Path
from datetime import datetime


# ============================================================================
# HARDWARE SAFETY THRESHOLDS (HP Z4 G5 Workstation)
# ============================================================================

THRESHOLDS = {
    # GPU: RTX 2000 Ada Generation
    "gpu_temp_warn": 80,       # Celsius - start warning
    "gpu_temp_critical": 90,   # Celsius - throttle/pause
    "gpu_vram_warn": 14000,    # MB (~85% of 16GB) - warn
    "gpu_vram_critical": 15500,# MB (~95% of 16GB) - risk of OOM
    "gpu_power_limit": 70,     # Watts (TDP)

    # CPU: Xeon W5-2565X
    "cpu_temp_warn": 85,       # Celsius - start warning
    "cpu_temp_critical": 95,   # Celsius - throttle/pause
    "ram_warn_pct": 85,        # % of 64GB - warn
    "ram_critical_pct": 95,    # % of 64GB - risk of OOM/swap
}


# ============================================================================
# GPU MONITORING (nvidia-smi)
# ============================================================================

def get_gpu_stats():
    """Query nvidia-smi for GPU metrics."""
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=temperature.gpu,memory.used,memory.total,"
             "utilization.gpu,utilization.memory,power.draw,power.limit,"
             "fan.speed,clocks.sm,clocks.mem",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            return None

        parts = [p.strip() for p in result.stdout.strip().split(",")]
        return {
            "gpu_temp_c": int(parts[0]),
            "gpu_vram_used_mb": int(parts[1]),
            "gpu_vram_total_mb": int(parts[2]),
            "gpu_util_pct": int(parts[3]),
            "gpu_mem_util_pct": int(parts[4]),
            "gpu_power_w": float(parts[5]),
            "gpu_power_limit_w": float(parts[6]),
            "gpu_fan_pct": int(parts[7]) if parts[7] != "[N/A]" else -1,
            "gpu_clock_sm_mhz": int(parts[8]),
            "gpu_clock_mem_mhz": int(parts[9]),
        }
    except Exception as e:
        return {"error": str(e)}


# ============================================================================
# CPU MONITORING (psutil + /sys/class/thermal)
# ============================================================================

def get_cpu_stats():
    """Query CPU metrics via psutil and system files."""
    stats = {}

    # CPU usage
    stats["cpu_util_pct"] = psutil.cpu_percent(interval=0.1)
    stats["cpu_freq_mhz"] = int(psutil.cpu_freq().current) if psutil.cpu_freq() else 0

    # Per-core usage (useful for AMX load detection)
    per_core = psutil.cpu_percent(interval=0, percpu=True)
    stats["cpu_cores_active"] = sum(1 for c in per_core if c > 5.0)
    stats["cpu_cores_total"] = len(per_core)
    stats["cpu_max_core_pct"] = max(per_core) if per_core else 0

    # RAM
    mem = psutil.virtual_memory()
    stats["ram_used_gb"] = round(mem.used / (1024**3), 1)
    stats["ram_total_gb"] = round(mem.total / (1024**3), 1)
    stats["ram_used_pct"] = mem.percent
    stats["ram_available_gb"] = round(mem.available / (1024**3), 1)

    # Swap
    swap = psutil.swap_memory()
    stats["swap_used_gb"] = round(swap.used / (1024**3), 1)
    stats["swap_total_gb"] = round(swap.total / (1024**3), 1)

    # CPU temperature (try multiple methods)
    cpu_temp = None
    try:
        temps = psutil.sensors_temperatures()
        if "coretemp" in temps:
            cpu_temp = max(t.current for t in temps["coretemp"])
        elif temps:
            first_key = next(iter(temps))
            cpu_temp = max(t.current for t in temps[first_key])
    except Exception:
        pass

    if cpu_temp is None:
        # Fallback: read from thermal zones
        for i in range(10):
            path = f"/sys/class/thermal/thermal_zone{i}/temp"
            if os.path.exists(path):
                try:
                    with open(path) as f:
                        t = int(f.read().strip()) / 1000
                        if cpu_temp is None or t > cpu_temp:
                            cpu_temp = t
                except Exception:
                    pass

    stats["cpu_temp_c"] = int(cpu_temp) if cpu_temp else -1

    # Load average (1, 5, 15 min)
    load1, load5, load15 = os.getloadavg()
    stats["load_avg_1m"] = round(load1, 2)
    stats["load_avg_5m"] = round(load5, 2)
    stats["load_avg_15m"] = round(load15, 2)

    return stats


# ============================================================================
# SAFETY CHECK
# ============================================================================

def check_safety(gpu, cpu):
    """Check if any thresholds are breached. Returns list of alerts."""
    alerts = []

    if gpu and "error" not in gpu:
        # GPU temperature
        if gpu["gpu_temp_c"] >= THRESHOLDS["gpu_temp_critical"]:
            alerts.append(("CRITICAL", f"GPU TEMP {gpu['gpu_temp_c']}C >= {THRESHOLDS['gpu_temp_critical']}C"))
        elif gpu["gpu_temp_c"] >= THRESHOLDS["gpu_temp_warn"]:
            alerts.append(("WARNING", f"GPU TEMP {gpu['gpu_temp_c']}C >= {THRESHOLDS['gpu_temp_warn']}C"))

        # GPU VRAM
        if gpu["gpu_vram_used_mb"] >= THRESHOLDS["gpu_vram_critical"]:
            alerts.append(("CRITICAL", f"GPU VRAM {gpu['gpu_vram_used_mb']}MB >= {THRESHOLDS['gpu_vram_critical']}MB"))
        elif gpu["gpu_vram_used_mb"] >= THRESHOLDS["gpu_vram_warn"]:
            alerts.append(("WARNING", f"GPU VRAM {gpu['gpu_vram_used_mb']}MB >= {THRESHOLDS['gpu_vram_warn']}MB"))

    if cpu:
        # CPU temperature
        if cpu["cpu_temp_c"] > 0:
            if cpu["cpu_temp_c"] >= THRESHOLDS["cpu_temp_critical"]:
                alerts.append(("CRITICAL", f"CPU TEMP {cpu['cpu_temp_c']}C >= {THRESHOLDS['cpu_temp_critical']}C"))
            elif cpu["cpu_temp_c"] >= THRESHOLDS["cpu_temp_warn"]:
                alerts.append(("WARNING", f"CPU TEMP {cpu['cpu_temp_c']}C >= {THRESHOLDS['cpu_temp_warn']}C"))

        # RAM
        if cpu["ram_used_pct"] >= THRESHOLDS["ram_critical_pct"]:
            alerts.append(("CRITICAL", f"RAM {cpu['ram_used_pct']}% >= {THRESHOLDS['ram_critical_pct']}%"))
        elif cpu["ram_used_pct"] >= THRESHOLDS["ram_warn_pct"]:
            alerts.append(("WARNING", f"RAM {cpu['ram_used_pct']}% >= {THRESHOLDS['ram_warn_pct']}%"))

        # Swap usage (any swap is a bad sign)
        if cpu["swap_used_gb"] > 1.0:
            alerts.append(("WARNING", f"SWAP in use: {cpu['swap_used_gb']}GB (memory pressure)"))

    return alerts


# ============================================================================
# LIVE DISPLAY
# ============================================================================

def format_bar(value, maximum, width=20):
    """Create a text progress bar."""
    pct = min(value / maximum, 1.0) if maximum > 0 else 0
    filled = int(pct * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {pct*100:5.1f}%"


def display_live(gpu, cpu, alerts, tick, start_time):
    """Print live dashboard to terminal."""
    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    mins = int((elapsed % 3600) // 60)
    secs = int(elapsed % 60)

    lines = []
    lines.append("\033[2J\033[H")  # Clear screen
    lines.append("=" * 72)
    lines.append("  HARDWARE MONITOR  |  UAI 2026 Campaign  |  HP Z4 G5 Workstation")
    lines.append(f"  Tick #{tick}  |  Uptime: {hours:02d}h {mins:02d}m {secs:02d}s  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 72)

    # GPU Section
    lines.append("")
    lines.append("  GPU: NVIDIA RTX 2000 Ada Generation (16 GB)")
    lines.append("  " + "-" * 50)
    if gpu and "error" not in gpu:
        lines.append(f"  Temp:    {gpu['gpu_temp_c']:3d}C    {format_bar(gpu['gpu_temp_c'], 100)}")
        lines.append(f"  VRAM:    {gpu['gpu_vram_used_mb']:5d} / {gpu['gpu_vram_total_mb']} MB   {format_bar(gpu['gpu_vram_used_mb'], gpu['gpu_vram_total_mb'])}")
        lines.append(f"  Util:    {gpu['gpu_util_pct']:3d}%   {format_bar(gpu['gpu_util_pct'], 100)}")
        lines.append(f"  Power:   {gpu['gpu_power_w']:5.1f} / {gpu['gpu_power_limit_w']:.0f} W    {format_bar(gpu['gpu_power_w'], gpu['gpu_power_limit_w'])}")
        lines.append(f"  Clocks:  SM {gpu['gpu_clock_sm_mhz']} MHz  |  MEM {gpu['gpu_clock_mem_mhz']} MHz")
    else:
        lines.append("  [GPU data unavailable]")

    # CPU Section
    lines.append("")
    lines.append(f"  CPU: Intel Xeon W5-2565X ({cpu['cpu_cores_total']} threads)")
    lines.append("  " + "-" * 50)
    temp_str = f"{cpu['cpu_temp_c']}C" if cpu['cpu_temp_c'] > 0 else "N/A"
    lines.append(f"  Temp:    {temp_str:>5s}   {'  ' + format_bar(cpu['cpu_temp_c'], 100) if cpu['cpu_temp_c'] > 0 else ''}")
    lines.append(f"  Usage:   {cpu['cpu_util_pct']:5.1f}%  {format_bar(cpu['cpu_util_pct'], 100)}")
    lines.append(f"  Freq:    {cpu['cpu_freq_mhz']} MHz")
    lines.append(f"  Cores:   {cpu['cpu_cores_active']}/{cpu['cpu_cores_total']} active")
    lines.append(f"  Load:    {cpu['load_avg_1m']:.2f} / {cpu['load_avg_5m']:.2f} / {cpu['load_avg_15m']:.2f}  (1/5/15 min)")

    # Memory Section
    lines.append("")
    lines.append("  MEMORY")
    lines.append("  " + "-" * 50)
    lines.append(f"  RAM:     {cpu['ram_used_gb']:5.1f} / {cpu['ram_total_gb']:.0f} GB    {format_bar(cpu['ram_used_gb'], cpu['ram_total_gb'])}")
    lines.append(f"  Avail:   {cpu['ram_available_gb']:5.1f} GB")
    lines.append(f"  Swap:    {cpu['swap_used_gb']:5.1f} / {cpu['swap_total_gb']:.0f} GB    {format_bar(cpu['swap_used_gb'], max(cpu['swap_total_gb'], 1))}")

    # Alerts Section
    lines.append("")
    if alerts:
        lines.append("  ALERTS")
        lines.append("  " + "-" * 50)
        for level, msg in alerts:
            marker = "!!!" if level == "CRITICAL" else " ! "
            lines.append(f"  {marker} [{level}] {msg}")
    else:
        lines.append("  STATUS: ALL SAFE")

    lines.append("")
    lines.append("=" * 72)
    lines.append("  Press Ctrl+C to stop monitoring")
    lines.append("=" * 72)

    print("\n".join(lines))


# ============================================================================
# JSON LOGGING
# ============================================================================

def log_to_json(log_file, gpu, cpu, alerts, tick):
    """Append a JSON record to the log file."""
    record = {
        "timestamp": datetime.now().isoformat(),
        "tick": tick,
        "gpu": gpu if gpu else {},
        "cpu": cpu,
        "alerts": [{"level": a[0], "message": a[1]} for a in alerts],
    }

    with open(log_file, "a") as f:
        f.write(json.dumps(record) + "\n")


# ============================================================================
# MAIN LOOP
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Hardware Safety Monitor for UAI 2026 Campaign")
    parser.add_argument("--interval", type=int, default=5, help="Polling interval in seconds (default: 5)")
    parser.add_argument("--log-dir", type=str, default="logs/hardware", help="Directory for JSON logs")
    parser.add_argument("--no-display", action="store_true", help="Disable live terminal display (log only)")
    args = parser.parse_args()

    # Setup log directory
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"hw_monitor_{timestamp}.jsonl"

    print(f"Hardware Monitor starting...")
    print(f"  Interval: {args.interval}s")
    print(f"  Log file: {log_file}")
    print(f"  Thresholds:")
    print(f"    GPU temp warn/critical: {THRESHOLDS['gpu_temp_warn']}C / {THRESHOLDS['gpu_temp_critical']}C")
    print(f"    GPU VRAM warn/critical: {THRESHOLDS['gpu_vram_warn']}MB / {THRESHOLDS['gpu_vram_critical']}MB")
    print(f"    CPU temp warn/critical: {THRESHOLDS['cpu_temp_warn']}C / {THRESHOLDS['cpu_temp_critical']}C")
    print(f"    RAM warn/critical: {THRESHOLDS['ram_warn_pct']}% / {THRESHOLDS['ram_critical_pct']}%")
    print()

    # Write initial metadata
    meta = {
        "type": "session_start",
        "timestamp": datetime.now().isoformat(),
        "hardware": {
            "gpu": "NVIDIA RTX 2000 Ada Generation (16GB)",
            "cpu": "Intel Xeon W5-2565X",
            "ram": "64GB DDR5",
            "workstation": "HP Z4 G5",
        },
        "thresholds": THRESHOLDS,
        "interval_seconds": args.interval,
    }
    with open(log_file, "w") as f:
        f.write(json.dumps(meta) + "\n")

    # Graceful shutdown
    running = True
    def signal_handler(sig, frame):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    tick = 0
    start_time = time.time()
    critical_count = 0

    while running:
        tick += 1

        # Collect metrics
        gpu = get_gpu_stats()
        cpu = get_cpu_stats()
        alerts = check_safety(gpu, cpu)

        # Log to JSON
        log_to_json(log_file, gpu, cpu, alerts, tick)

        # Live display
        if not args.no_display:
            display_live(gpu, cpu, alerts, tick, start_time)

        # Track critical alerts
        critical_alerts = [a for a in alerts if a[0] == "CRITICAL"]
        if critical_alerts:
            critical_count += 1
            if critical_count >= 3:
                # Log critical event
                event = {
                    "type": "critical_sustained",
                    "timestamp": datetime.now().isoformat(),
                    "tick": tick,
                    "message": "3+ consecutive critical alerts detected",
                    "alerts": [a[1] for a in critical_alerts],
                }
                with open(log_file, "a") as f:
                    f.write(json.dumps(event) + "\n")
        else:
            critical_count = 0

        time.sleep(args.interval)

    # Write session end
    end_record = {
        "type": "session_end",
        "timestamp": datetime.now().isoformat(),
        "total_ticks": tick,
        "duration_seconds": round(time.time() - start_time, 1),
    }
    with open(log_file, "a") as f:
        f.write(json.dumps(end_record) + "\n")

    print("\nMonitor stopped. Log saved to:", log_file)


if __name__ == "__main__":
    main()
