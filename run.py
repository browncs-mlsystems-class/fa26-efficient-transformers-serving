"""
Entry point for every experiment in the assignment.

    uv run run.py kvcache       # offline: naive vs KV-cache generation on one batch
    uv run run.py speculative   # offline: naive vs KV cache vs speculative decoding, batch size 1
    uv run run.py serve         # one Poisson-arrival client run against one server
    uv run run.py sweep         # throughput-latency sweep over batch size x rate (with or without KV)
    uv run run.py plot          # plot the stats saved by `serve` / `sweep`

Run `uv run run.py <command> --help` to see each command's options.
"""
import argparse
import asyncio
import os
from pathlib import Path

# Let PyTorch's GPU memory allocator grow existing memory segments instead of reserving new
# ones for every new tensor size, which reduces fragmentation. Must be set before CUDA is used.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

DEFAULT_TRACE = str(Path(__file__).resolve().parent / "data" / "humaneval.jsonl")


def add_model_args(p: argparse.ArgumentParser, init_from: str):
    p.add_argument('--init_from', type=str, default=init_from,
                   help="either 'resume' (from --ckpt_path) or a gpt2 variant (e.g. 'gpt2-xl')")
    p.add_argument('--ckpt_path', type=str, default='out/ckpt.pt',
                   help="checkpoint to load; ignored if init_from is not 'resume'")
    p.add_argument('--device', type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu",
                   help="e.g. 'cpu', 'cuda', 'cuda:0', 'cuda:1'")
    p.add_argument('--dtype', type=str, default='auto', choices=['auto', 'bf16', 'fp16', 'fp32'],
                   help="precision to run the model in; 'auto' uses bf16 where the GPU supports it, "
                        "else fp16 (fp32 if DEBUG is set in model.py)")


def add_offline_parser(sub, name: str, help: str, batch_size: int, prompt_length: int,
                       max_new_tokens: int):
    p = sub.add_parser(name, help=help)
    add_model_args(p, init_from='gpt2-large')
    p.add_argument('--start', type=str, default=DEFAULT_TRACE,
                   help='prompt string, a .jsonl file of prompts, or "FILE:prompt.txt" to load from a file')
    p.add_argument('--batch_size', type=int, default=batch_size)
    p.add_argument('--prompt_length', type=int, default=prompt_length)
    p.add_argument('--max_new_tokens', type=int, default=max_new_tokens,
                   help='number of tokens generated in each sample')
    p.add_argument('--num_samples', type=int, default=1, help='number of samples to draw')
    p.add_argument('--num_warmup', type=int, default=1,
                   help='how many warmups to do before benchmarking')
    p.add_argument('--warmup_max_new_tokens', type=int, default=4)
    p.add_argument('--temperature', type=float, default=0.4,
                   help='1.0 = no change, < 1.0 = less random, > 1.0 = more random')
    p.add_argument('--top_k', type=int, default=200,
                   help='retain only the top_k most likely tokens')
    p.add_argument('--draft_model', type=str, default='gpt2',
                   help='gpt2 variant to use as the draft model for speculative decoding')
    p.add_argument('--speculative_tokens', type=int, default=3,
                   help='how many tokens should the draft model decode?')
    p.add_argument('--seed', type=int, default=1337)
    p.add_argument('--compile', action='store_true', default=False,
                   help='use PyTorch 2.0 to compile the model to be faster')


def add_trace_args(p: argparse.ArgumentParser):
    p.add_argument('--trace_file', type=str, default=DEFAULT_TRACE,
                   help="Path to trace file (JSONL with a 'prompt' field per line; with "
                        "--trace_type file, also 'prompt_length' and 'max_new_tokens'), "
                        "e.g. traces/short_uniform.jsonl")
    p.add_argument('--trace_type', type=str, default='uniform', choices=['uniform', 'mixed', 'file'],
                   help="'uniform' for same-length requests, 'mixed' for varied lengths, "
                        "'file' to take every request's lengths from --trace_file")
    p.add_argument('--prompt_length', type=int, default=128,
                   help="prompt length in tokens (uniform trace only)")
    p.add_argument('--max_new_tokens', type=int, default=128,
                   help="max_new_tokens per request (uniform trace only)")
    p.add_argument('--num_requests', type=int, default=None,
                   help="use only the first N trace entries (None = all); the client cycles "
                        "through them until --max_time")
    p.add_argument('--temperature', type=float, default=1.0, help="temperature for sampling")
    p.add_argument('--top_k', type=int, default=None, help="top_k for sampling")
    p.add_argument('--server_type', type=str, default='static', choices=['static', 'continuous'])
    p.add_argument('--max_wait', type=float, default=2.0,
                   help="Max wait time for static batching (seconds)")
    p.add_argument('--max_time', type=float, default=120.0,
                   help="Maximum time to submit requests (seconds)")
    p.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    p.add_argument('--stat_dir', type=str, default='stat', help="where to save stats")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)

    add_offline_parser(sub, 'kvcache', 'offline: naive vs KV-cache generation',
                       batch_size=8, prompt_length=128, max_new_tokens=128)
    add_offline_parser(sub, 'speculative', 'offline: speculative decoding (batch size 1)',
                       batch_size=1, prompt_length=192, max_new_tokens=256)

    p = sub.add_parser('serve', help='one client run against one server')
    add_model_args(p, init_from='gpt2')
    add_trace_args(p)
    p.add_argument('--rate', type=float, default=5.0,
                   help="Arrival rate (lambda) in requests per second for Poisson process")
    p.add_argument('--batch_size', type=int, default=8, help="Static batch size")
    p.add_argument('--use_kv', action='store_true', default=False,
                   help="Serve with the KV cache. Off = recompute the whole sequence every step.")
    p.add_argument('--server_kwargs', type=str, default='{}',
                   help="Additional kwargs for server constructor as JSON string")

    p = sub.add_parser('sweep', help='throughput-latency sweep')
    add_model_args(p, init_from='gpt2')
    add_trace_args(p)
    p.add_argument('--rates', type=str, default='1,2,4,8',
                   help="comma-separated offered loads (req/s)")
    p.add_argument('--batch_sizes', type=str, default='1,2,4,8',
                   help="comma-separated static batch sizes")
    p.add_argument('--use_kv', action='store_true', default=False,
                   help="Sweep the KV-cache server. Off = the baseline (no cache). Run the "
                        "sweep once each way, with rates around each one's capacity.")

    p = sub.add_parser('plot', help='plot throughput-latency curves from saved stats')
    p.add_argument("--stat_dir", default="stat", help="directory of stats JSON files")
    p.add_argument("--out", default="graph/throughput_latency.png", help="output image path")
    p.add_argument("--metric", default="median", choices=["median", "p90", "p99", "mean"],
                   help="latency statistic for the y-axis")
    p.add_argument("--x", default="achieved_throughput", choices=["rate", "achieved_throughput", "effective_rate"],
                   help="x-axis quantity")
    p.add_argument("--server_type", default=None,
                   help="only plot runs with this server_type (default: all)")
    p.add_argument("--min_sustained", type=float, default=0.9,
                   help="drop points whose achieved_throughput < this fraction of the "
                        "offered load (rate). Set 0 to keep the saturated tail.")
    p.add_argument("--rates", default=None,
                   help="comma-separated offered loads to include (default: all)")
    p.add_argument("--trace", default=None,
                   help="only plot runs of this trace, e.g. short_uniform "
                        "(required if --stat_dir has runs of several traces)")

    return parser.parse_args()


def main():
    args = parse_args()
    if getattr(args, 'dtype', 'auto') != 'auto':
        from efficientserving.utils import set_dtype
        set_dtype(args.dtype)
    if args.command in ('kvcache', 'speculative'):
        from efficientserving.offline import run_offline
        run_offline(args, args.command)
    elif args.command == 'serve':
        from efficientserving.client import run_client
        asyncio.run(run_client(args))
    elif args.command == 'sweep':
        from efficientserving.experiment import run_sweep
        asyncio.run(run_sweep(args))
    elif args.command == 'plot':
        from efficientserving.plotting import plot_throughput_latency
        plot_throughput_latency(stat_dir=args.stat_dir, out=args.out, metric=args.metric,
                                x=args.x, server_type=args.server_type,
                                min_sustained=args.min_sustained, rates=args.rates,
                                trace=args.trace)


if __name__ == '__main__':
    main()
