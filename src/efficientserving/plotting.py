"""
Plots of the per-run stats saved by the client (see `client.save_stats`).

You will implement `plot_throughput_latency`, which draws throughput against latency
(median by default; p99 / p90 / mean also available), with one line per
(server type, KV vs baseline, batch size, block size for paged runs).

By default, it drops a point where the server can't keep up with the offered
load if its achieved throughput is below 90% of the offered load
(min_sustained, set to 0 to plot the full saturated tail).
"""
import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")  # headless / no display needed
import matplotlib.pyplot as plt

# Metric key in each stats file -> human-readable axis label.
METRIC_LABELS = {
    "median_latency": "median (p50) E2E latency (s)",
    "p90_latency": "p90 E2E latency (s)",
    "p99_latency": "p99 E2E latency (s)",
    "mean_latency": "mean E2E latency (s)",
}
METRIC_KEYS = {"median": "median_latency", "p90": "p90_latency",
               "p99": "p99_latency", "mean": "mean_latency"}
X_LABELS = {
    "rate": "offered load (req/s)",
    "achieved_throughput": "achieved throughput (req/s)",
    "effective_rate": "effective submission rate (req/s)",
}


def load_records(stat_dir: str) -> list[dict]:
    records = []
    for fp in sorted(glob.glob(os.path.join(stat_dir, "*.json"))):
        try:
            with open(fp) as f:
                records.append(json.load(f))
        except (json.JSONDecodeError, OSError) as e:
            print(f"[plot] skipping {fp}: {e}")
    return records


def save_figure(out: str):
    out_dir = os.path.dirname(out) or "."
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()


def trace_name(record: dict) -> str:
    """Short name of the trace a run replayed, e.g. 'short_uniform'."""
    trace_file = record.get("trace_file")
    return os.path.splitext(os.path.basename(trace_file))[0] if trace_file else "trace"


def plot_throughput_latency(stat_dir: str = "stat", out: str = "graph/throughput_latency.png",
                            metric: str = "median", x: str = "achieved_throughput",
                            server_type: str | None = None, min_sustained: float = 0.9,
                            rates: str | None = None, trace: str | None = None):
    """
    Plots latency against throughput for one trace, one line per
    (server type, KV vs baseline, batch size, block size for paged runs).

    Args:
        stat_dir: directory of client stats JSON files
        out: output image path
        metric: latency statistic for the y-axis ('median', 'p90', 'p99' or 'mean')
        x: x-axis quantity ('rate', 'achieved_throughput' or 'effective_rate')
        server_type: only plot runs with this server_type (default: all)
        min_sustained: drop points whose achieved_throughput < this fraction of the
        offered load (rate); set 0 to keep the saturated tail
        rates: comma-separated offered loads to include (default: every rate present)
        trace: only plot runs of this trace, by name (e.g. 'short_uniform') or path;
        required when `stat_dir` holds runs of more than one trace
    """
    # TODO: Plot one throughput-latency figure for one trace and save it to `out`.
    #   1. Load every stats record in `stat_dir` (`load_records`). Each record holds one
    #      run's config (server_type, use_kv, batch_size, block_size, rate, trace_file, ...)
    #      and its results
    #      (achieved_throughput, median_latency, p99_latency, ...); see `client.save_stats`.
    #   2. Keep only the records of `trace` (compare against `trace_name(record)`), and apply
    #      the `server_type` and `rates` filters when they are given.
    #   3. Drop points the server couldn't sustain: achieved_throughput < min_sustained * rate.
    #   4. Draw one line per (server type, KV vs baseline, batch size, and block size for
    #      paged runs): x = record[x], y = the latency statistic named by `metric`
    #      (METRIC_KEYS maps it to the record key), sorted by x. Runs that differ in any of
    #      these must not be merged into one line. Label each line and make KV and baseline
    #      easy to tell apart.
    #   5. Label the axes (X_LABELS, METRIC_LABELS), title the plot with the trace name,
    #      add a legend, and save it with `save_figure(out)`.
    raise NotImplementedError("plot_throughput_latency")


def plot_memory(stat_dir: str = "stat", out: str = "graph/kv_memory.png",
                trace: str | None = None, rates: str | None = None):
    """
    Plots KV cache memory utilization (%) over time of continuous (slot) and paged runs
    for one trace, one line per run.

    Args:
        stat_dir: directory of client stats JSON files
        out: output image path
        trace: only plot runs of this trace, by name (e.g. 'mixed_short_long') or path;
        required when `stat_dir` holds runs of more than one trace
        rates: comma-separated offered loads to include (default: every rate present)
    """
    # TODO (Part 4): Plot Graph 5 of the handout and save it to `out`.
    #   Each continuous or paged record has a `memory_log`: one sample per scheduler step,
    #   each a dict with 'time', 'cached' (tokens actually in the KV cache) and 'reserved'
    #   (tokens set aside for the running requests).
    #   x-axis: time (s) since the run's first sample.
    #   y-axis: memory utilization (%) = 100 * cached / total KV cache tokens, where the
    #   total is the record's `kv_cache_tokens` (batch_size * max_seq_len if it is None).
    #   Draw one line per run (server type, and block size for paged runs), comparing slots
    #   and blocks at the same KV cache size, same trace and same rate.
    raise NotImplementedError("plot_memory")


# TODO: Add your own plotting functions below (use `load_records` and `save_figure`),
# and call them from `run.py`.
