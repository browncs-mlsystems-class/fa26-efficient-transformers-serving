"""
Client that submits Poisson-distributed requests to an inference server.

The client and server run in the same process, as tasks on one asyncio event loop.

Supports trace formats:
    - uniform: all requests share the same prompt length and max_new_tokens
    - mixed: requests have varying prompt lengths and max_new_tokens
    - file: every request's prompt length and max_new_tokens come from a trace file
      (see the traces/ folder)
"""
from __future__ import annotations

import asyncio
import itertools
import json
import os
import random
import time

import tiktoken
import torch

from efficientserving import utils
from efficientserving.model import GPT
from efficientserving.server.interface import InferenceRequest, InferenceServer


def generate_uniform_trace(prompts: list[str], prompt_length: int,
                            max_new_tokens: int) -> list[dict]:
    """Generate a trace where all requests have the same prompt length and max_new_tokens.

    Returns list of dicts with keys: 'prompt', 'prompt_length', 'max_new_tokens'.
    """
    trace = []
    for prompt in prompts:
        trace.append({
            'prompt': prompt,
            'prompt_length': prompt_length,
            'max_new_tokens': max_new_tokens,
        })
    return trace


def generate_mixed_trace(prompts: list[str], enc: tiktoken.Encoding) -> list[dict]:
    """Generate a trace with mixed prompt lengths and max_new_tokens.

    Prompt length is the prompt's actual length in tokens (capped at 256 tokens).
    max_new_tokens varies based on prompt length (shorter prompts get more generation).

    Returns list of dicts with keys: 'prompt', 'prompt_length', 'max_new_tokens'.
    """
    trace = []
    for prompt in prompts:
        # Use actual prompt length (capped at 256 tokens)
        actual_length = min(len(utils.encode(enc, prompt)), 256)
        # Vary max_new_tokens based on prompt characteristics
        base_max_new = 64
        # Scale inversely with prompt length (shorter prompts → more generation)
        scale = max(0.5, 1.0 - actual_length / 512)
        max_new = int(base_max_new / scale)
        max_new = min(max_new, 256)  # cap at 256
        trace.append({
            'prompt': prompt,
            'prompt_length': actual_length,
            'max_new_tokens': max_new,
        })
    return trace


def build_trace(prompts: list[str], trace_type: str, prompt_length: int,
                max_new_tokens: int, enc: tiktoken.Encoding) -> list[dict]:
    """Build a request trace.
    'uniform' keeps only prompts with >= prompt_length tokens (so static batches are homogeneous)
    'mixed' varies both lengths."""
    if trace_type == 'uniform':
        prompts = [p for p in prompts if len(utils.encode(enc, p)) >= prompt_length]
        return generate_uniform_trace(prompts, prompt_length, max_new_tokens)
    return generate_mixed_trace(prompts, enc)


TRACE_FILE_KEYS = ('prompt', 'prompt_length', 'max_new_tokens')


def load_trace_file(trace_file: str, enc: tiktoken.Encoding,
                    num_requests: int | None = None) -> list[dict]:
    """Load a trace from a JSONL file where every line has 'prompt', 'prompt_length'
    (in tokens) and 'max_new_tokens'.

    Returns list of dicts with keys: 'prompt', 'prompt_length', 'max_new_tokens'.
    """
    trace = []
    with open(trace_file, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            missing = [k for k in TRACE_FILE_KEYS if k not in entry]
            if missing:
                raise ValueError(f"{trace_file}:{line_no} is missing {missing}")
            # prompts are cut to prompt_length tokens when sent, so they must be at least that long
            num_tokens = len(utils.encode(enc, entry['prompt']))
            if num_tokens < entry['prompt_length']:
                raise ValueError(f"{trace_file}:{line_no} has prompt_length {entry['prompt_length']}, "
                                 f"but its prompt is only {num_tokens} tokens")
            trace.append({k: entry[k] for k in TRACE_FILE_KEYS})
    if num_requests is not None:
        trace = trace[:num_requests]
    return trace


def load_trace(trace_file: str, trace_type: str, prompt_length: int, max_new_tokens: int,
               enc: tiktoken.Encoding, num_requests: int | None = None) -> list[dict]:
    """Load the trace to replay.

    'file' reads every request's prompt_length and max_new_tokens from `trace_file`
    (see `load_trace_file`); 'uniform' and 'mixed' build a trace from the prompts in
    `trace_file` (see `build_trace`).
    """
    if trace_type == 'file':
        return load_trace_file(trace_file, enc, num_requests)
    prompts = utils.load_prompts(trace_file, num_requests)
    return build_trace(prompts, trace_type, prompt_length, max_new_tokens, enc)


async def poisson_arrival_client(
    server: InferenceServer,
    device: str,
    trace: list[dict],
    rate: float,
    max_time: float,
    temperature: float = 1.0,
    top_k: int | None = None,
    enc: tiktoken.Encoding | None = None,
) -> dict:
    """Submit requests using a Poisson arrival process, cycling through the trace until
    max_time has passed (so a run never stops early because the trace ran out).

    Args:
        server: InferenceServer instance to submit requests to
        device: device to put request tokens on
        trace: List of trace entries, each with 'prompt', 'prompt_length', 'max_new_tokens'
        rate: Arrival rate (lambda) in requests per second
        max_time: Wall-clock time (s) to keep submitting requests
        temperature: Temperature for sampling
        top_k: Top-k for sampling
        enc: tokenizer; defaults to GPT-2's

    Returns:
        Dictionary with timing statistics
    """
    enc = enc or utils.get_encoder()
    if not trace:
        raise ValueError("trace is empty: no requests to send (are the prompts shorter than "
                         "--prompt_length?)")

    # map request_id to submit wall time;completion wall time;prompt/new-token sizes
    send_timestamps: dict[int, float] = {}
    receive_timestamps: dict[int, float] = {}
    request_meta: dict[int, dict] = {}
    inflight: list[InferenceRequest] = []
    sender_done = asyncio.Event()
    num_sent = 0
    start_time = time.time()

    print(f"[Client] Starting Poisson arrival process")
    print(f"[Client] Rate: {rate} req/s, Inter-arrival: {1.0/rate:.3f}s")
    print(f"[Client] Max submission time: {max_time}s")
    print(f"[Client] Trace size: {len(trace)} requests (cycled until max_time)")
    print()

    async def sender():
        nonlocal num_sent
        for entry in itertools.cycle(trace):
            # Check if we've exceeded max_time
            if time.time() - start_time > max_time:
                print(f"[Client] Reached max_time ({max_time}s), stopping submission")
                break

            # Wait for next Poisson arrival (exponentially distributed)
            await asyncio.sleep(random.expovariate(rate))
            if time.time() - start_time > max_time:
                break

            # Encode + truncate the prompt, move to device
            prompt_length = entry.get('prompt_length', 64)
            tokens = utils.encode(enc, entry['prompt'])[:prompt_length]
            idx = torch.tensor([tokens], dtype=torch.long, device=device)

            # Non-blocking submit
            req = await server.submit_nowait(
                idx=idx,
                max_new_tokens=entry['max_new_tokens'],
                temperature=temperature,
                top_k=top_k,
            )
            send_timestamps[req.request_id] = time.time()
            request_meta[req.request_id] = {
                'prompt_length': idx.size(1),
                'max_new_tokens': entry['max_new_tokens'],
            }
            inflight.append(req)
            num_sent += 1

            if num_sent % 10 == 0:
                elapsed = time.time() - start_time
                print(f"[Client] Submitted {num_sent} requests in {elapsed:.2f}s "
                      f"({num_sent/max(elapsed, 0.001):.1f} req/s)")

        sender_done.set()

    failures: dict[int, str] = {}

    async def responder():
        """Listen for completed requests on the server's response queue."""
        received = 0
        while True:
            if sender_done.is_set() and (
                received >= num_sent or all(r.future.done() for r in inflight)
            ):
                break
            try:
                req = await asyncio.wait_for(server.response_queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue  # re-check the exit condition
            receive_timestamps[req.request_id] = time.time()
            received += 1
            # Retrieve any exception and record it as a failed request.
            exc = req.future.exception() if req.future.done() else None
            if exc is not None:
                failures[req.request_id] = repr(exc)

    # Run the sender and responder concurrently
    await asyncio.gather(sender(), responder())

    # Drain any completions the responder didn't pull
    while not server.response_queue.empty():
        req = server.response_queue.get_nowait()
        receive_timestamps.setdefault(req.request_id, time.time())
        if req.future.done() and req.future.exception() is not None:
            failures[req.request_id] = repr(req.future.exception())

    elapsed = time.time() - start_time

    # Per-request end-to-end latency excluding failed ones
    latencies = {
        rid: receive_timestamps[rid] - send_timestamps[rid]
        for rid in receive_timestamps
        if rid in send_timestamps and rid not in failures
    }
    results = [
        {
            'request_id': rid,
            'prompt_length': request_meta[rid]['prompt_length'],
            'max_new_tokens': request_meta[rid]['max_new_tokens'],
            'latency': latencies[rid],
        }
        for rid in sorted(latencies)
    ]

    def percentile(sorted_vals, p):
        if not sorted_vals:
            return 0.0
        k = min(len(sorted_vals) - 1, max(0, int(round((p / 100.0) * (len(sorted_vals) - 1)))))
        return sorted_vals[k]

    lat_values = sorted(latencies.values())
    mean_latency = sum(lat_values) / len(lat_values) if lat_values else 0.0
    median_latency = percentile(lat_values, 50)
    p90_latency = percentile(lat_values, 90)
    p99_latency = percentile(lat_values, 99)
    num_completed = len(latencies)
    achieved_throughput = num_completed / max(elapsed, 1e-9)

    print()
    print(f"[Client] Done. Submitted {num_sent}, received {len(receive_timestamps)} "
          f"({len(failures)} failed) in {elapsed:.3f}s")
    if elapsed > 0:
        print(f"[Client] Effective rate: {num_sent/elapsed:.1f} req/s")

    return {
        'total_requests': num_sent,
        'total_received': len(receive_timestamps),
        'num_completed': num_completed,
        'num_failed': len(failures),
        'total_time': elapsed,
        'effective_rate': num_sent / max(elapsed, 0.001),
        'achieved_throughput': achieved_throughput,
        'mean_latency': mean_latency,
        'median_latency': median_latency,
        'p90_latency': p90_latency,
        'p99_latency': p99_latency,
        'latencies': latencies,
        'results': results,
    }


# Keys describing a single run's configuration (the independent variables).
_CONFIG_KEYS = ('server_type', 'use_kv', 'batch_size', 'max_wait', 'rate',
                'max_time', 'trace_type', 'trace_file', 'init_from', 'prompt_length',
                'max_new_tokens', 'seed')


def trace_name_of(trace_file: str | None) -> str:
    """Short name of a trace, e.g. 'short_uniform' for traces/short_uniform.jsonl."""
    return os.path.splitext(os.path.basename(trace_file))[0] if trace_file else 'trace'


def config_from_args(args) -> dict:
    """Build a run-config dict (used for stat records) from parsed CLI args."""
    return {k: getattr(args, k) for k in _CONFIG_KEYS}


def save_stats(stats: dict, config: dict, stat_dir: str = "stat") -> str:
    os.makedirs(stat_dir, exist_ok=True)
    record = {
        **{k: config.get(k) for k in _CONFIG_KEYS},
        'total_requests': stats['total_requests'],
        'total_received': stats['total_received'],
        'num_completed': stats['num_completed'],
        'num_failed': stats['num_failed'],
        'total_time': stats['total_time'],
        'effective_rate': stats['effective_rate'],
        'achieved_throughput': stats['achieved_throughput'],
        'mean_latency': stats['mean_latency'],
        'median_latency': stats['median_latency'],
        'p90_latency': stats['p90_latency'],
        'p99_latency': stats['p99_latency'],
        'latencies': list(stats['latencies'].values()),
    }
    ts = time.strftime("%Y%m%d_%H%M%S")
    kv_tag = "kv" if config['use_kv'] else "base"
    trace_name = trace_name_of(config.get('trace_file'))
    fname = os.path.join(
        stat_dir,
        f"{trace_name}_{config['server_type']}_{kv_tag}_bs{config['batch_size']}_rate{config['rate']}_{ts}.json",
    )
    with open(fname, 'w') as f:
        json.dump(record, f, indent=2)
    print(f"[Client] Saved stats to {fname}")
    return fname


def build_server(model: GPT, server_type: str, device: str, use_kv: bool,
                 batch_size: int = 1, max_wait: float = 60.0,
                 server_kwargs: dict | None = None) -> InferenceServer:
    """Build an inference server instance.

    Args:
        model: The GPT model instance
        server_type: Type of server ('static' or 'continuous')
        device: device the model runs on
        use_kv: whether the server generates with the KV cache
        batch_size: static batching only -- maximum batch size
        max_wait: static batching only -- maximum time (s) to wait to fill a batch
        server_kwargs: additional kwargs for the server constructor

    Returns:
        An InferenceServer subclass instance
    """
    kwargs = {'device': device, 'use_kv': use_kv}
    if server_type == 'static':
        from efficientserving.server.static_batching import StaticBatchServer
        kwargs.update(batch_size=batch_size, max_wait_time_s=max_wait)
        kwargs.update(server_kwargs or {})
        return StaticBatchServer(model=model, **kwargs)
    elif server_type == 'continuous':
        from efficientserving.server.continuous_batching import ContinuousBatchServer
        kwargs.update(server_kwargs or {})
        return ContinuousBatchServer(model=model, **kwargs)
    else:
        raise ValueError(f"Unknown server type: {server_type}")


async def run_client(args):
    """Runs one client against one server, then prints and saves the stats. See `run.py serve`."""
    random.seed(args.seed)
    enc = utils.get_encoder()

    # load or build the trace
    trace = load_trace(args.trace_file, args.trace_type, args.prompt_length,
                       args.max_new_tokens, enc, args.num_requests)
    prompts = [entry['prompt'] for entry in trace]
    print(f"[Client] Trace loaded ({args.trace_type} mode, {len(trace)} requests from {args.trace_file})")

    # Load model
    print("[Client] Loading model...")
    model = utils.load_model(args.init_from, args.ckpt_path, args.device)

    # call Warmup process (separately)
    utils.warmup_model(model, prompts, args.device, enc, use_kv=args.use_kv)

    # Create inference server
    server = build_server(model, args.server_type, args.device, args.use_kv,
                          batch_size=args.batch_size, max_wait=args.max_wait,
                          server_kwargs=json.loads(args.server_kwargs))
    print(f"[Client] Server type: {args.server_type}, use_kv: {args.use_kv}")
    if args.server_type == 'static':
        print(f"[Client] Batch size: {args.batch_size}, Max wait: {args.max_wait}s")
    print()

    # Start server
    await server.start()

    # Run client
    try:
        stats = await poisson_arrival_client(
            server=server,
            device=args.device,
            trace=trace,
            rate=args.rate,
            max_time=args.max_time,
            temperature=args.temperature,
            top_k=args.top_k,
            enc=enc,
        )
    finally:
        await server.shutdown()

    # Print summary
    print("\n" + "=" * 60)
    print("CLIENT SUMMARY")
    print("=" * 60)
    print(f"Total requests submitted: {stats['total_requests']}")
    print(f"Total requests received:  {stats['total_received']} "
          f"({stats['num_completed']} completed, {stats['num_failed']} failed)")
    print(f"Total time: {stats['total_time']:.3f}s")
    print(f"Effective submission rate: {stats['effective_rate']:.1f} req/s")
    print(f"Achieved throughput: {stats['achieved_throughput']:.2f} req/s")

    if stats['latencies']:
        print(f"E2E latency  mean={stats['mean_latency']:.3f}s  "
              f"p50={stats['median_latency']:.3f}s  "
              f"p90={stats['p90_latency']:.3f}s  "
              f"p99={stats['p99_latency']:.3f}s")

    if stats['num_failed']:
        print(f"WARNING: {stats['num_failed']} requests failed (see the [Server] errors above).")

    save_stats(stats, config_from_args(args), stat_dir=args.stat_dir)
