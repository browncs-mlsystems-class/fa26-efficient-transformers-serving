"""
Sweep driver: run a whole throughput-latency experiment in ONE process on ONE
GPU. Loads the model once, warms up once, then sweeps

    batch_size x offered_load (rate)

with or without the KV cache (--use_kv), running the Poisson-arrival client for each
point and writing one stats file per point into --stat_dir (default: stat/). Run it once
without and once with --use_kv (each with rates around its own capacity), then
`run.py plot` aggregates the stats into the KV-vs-baseline throughput-latency curves.

Running everything in one job keeps every point on the same GPU (comparable) and
avoids reloading the model per point.
"""
import random
import time

from efficientserving import client, utils


def _int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(',') if x.strip()]


def _float_list(s: str) -> list[float]:
    return [float(x) for x in s.split(',') if x.strip()]


async def run_sweep(args):
    """Runs every point of the sweep. See `run.py sweep`."""
    enc = utils.get_encoder()
    use_kv = args.use_kv

    rates = _float_list(args.rates)
    batch_sizes = _int_list(args.batch_sizes)

    trace = client.load_trace(args.trace_file, args.trace_type, args.prompt_length,
                              args.max_new_tokens, enc, args.num_requests)
    prompts = [entry['prompt'] for entry in trace]
    print(f"[Sweep] Trace: {args.trace_type} ({args.trace_file}), {len(trace)} requests")

    print(f"[Sweep] Loading model {args.init_from} on {args.device}...")
    model = utils.load_model(args.init_from, args.ckpt_path, args.device)
    utils.warmup_model(model, prompts, args.device, enc, use_kv=use_kv)

    total_points = len(batch_sizes) * len(rates)
    tag = 'KV' if use_kv else 'baseline'
    print(f"[Sweep] {tag}: {total_points} points = "
          f"{len(batch_sizes)} batch size(s) x {len(rates)} rate(s)")

    point = 0
    sweep_start = time.time()
    for bs in batch_sizes:
        for rate in rates:
            point += 1
            print(f"\n[Sweep] ({point}/{total_points}) {tag}  batch={bs}  rate={rate} req/s")

            # reseed per point, so a point's arrival times don't depend on the points before it
            random.seed(args.seed)
            server = client.build_server(model, args.server_type, args.device, use_kv,
                                         batch_size=bs, max_wait=args.max_wait)
            await server.start()
            try:
                stats = await client.poisson_arrival_client(
                    server=server, device=args.device, trace=trace,
                    rate=rate, max_time=args.max_time,
                    temperature=args.temperature, top_k=args.top_k, enc=enc,
                )
            finally:
                await server.shutdown()

            config = {
                'server_type': args.server_type, 'use_kv': use_kv, 'batch_size': bs,
                'max_wait': args.max_wait, 'rate': rate, 'max_time': args.max_time,
                'trace_type': args.trace_type, 'trace_file': args.trace_file,
                'init_from': args.init_from,
                'prompt_length': args.prompt_length, 'max_new_tokens': args.max_new_tokens,
                'seed': args.seed,
            }
            client.save_stats(stats, config, stat_dir=args.stat_dir)
            print(f"[Sweep]   -> completed {stats['num_completed']}/{stats['total_requests']}, "
                  f"p50={stats['median_latency']:.3f}s p99={stats['p99_latency']:.3f}s "
                  f"throughput={stats['achieved_throughput']:.2f} req/s")

    print(f"\n[Sweep] Done. {total_points} points in {time.time()-sweep_start:.1f}s. "
          f"Stats in {args.stat_dir}/. Now run:  uv run run.py plot")
