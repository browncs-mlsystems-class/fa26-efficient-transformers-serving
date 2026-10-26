"""
Offline benchmarks: generate from one fixed batch of prompts in a single process (no
server), timing naive generation against KV-cache generation and, for a single
request, speculative decoding. Also checks that every method generates the same tokens.
"""
from __future__ import annotations

import time
from collections.abc import Callable

import torch

from efficientserving import utils
from efficientserving.generation import generate, generate_kv, generate_speculative


def load_batch(start: str, prompt_length: int, batch_size: int, enc, device: str) -> torch.Tensor:
    """
    Builds the batch of prompts to generate from.

    `start` is either a .jsonl file (the first `prompt_length` tokens of up to
    `batch_size` prompts that are longer than that), "FILE:<path>" to read one prompt
    from a file, or the prompt text itself.
    """
    print(f"Using prompts from {start}")
    if start.endswith('jsonl'):
        start_ids = [utils.encode(enc, x) for x in utils.load_prompts(start)]
        long_enough = [x[:prompt_length] for x in start_ids if len(x) > prompt_length]
        print('NUM SATISFYING PROMPTS =', len(long_enough))
        x = torch.tensor(long_enough[:batch_size], dtype=torch.long, device=device)
        print('RUNNING WITH BATCH SIZE =', x.size(0))
        return x
    if start.startswith('FILE:'):
        with open(start[5:], 'r', encoding='utf-8') as f:
            start = f.read()
    return torch.tensor(utils.encode(enc, start), dtype=torch.long, device=device)[None, ...]


def benchmark(name: str, gen: Callable[[torch.Tensor, int], torch.Tensor], x: torch.Tensor,
              args, ctx, decode: Callable[[list[int]], str]) -> tuple[list[list[int]], float]:
    """
    Times `gen(x, max_new_tokens)` after warming it up.

    Returns:
        tuple[list[list[int]], float]: every generated sequence, and the elapsed seconds
    """
    print(f'\n--------------- Running {name} inference with top_k={args.top_k} and temperature={args.temperature:.3f} ---------------')
    print('Starting warmup... ', end='')
    with torch.no_grad(), ctx:
        for _ in range(args.num_warmup):
            gen(x, args.warmup_max_new_tokens)
    utils.synchronize(args.device)
    print('Finished')

    generations = []
    t0 = time.time()
    with torch.no_grad(), ctx:
        for k in range(args.num_samples):
            torch.manual_seed(k + args.seed) # we want consistency
            generations.extend(gen(x, args.max_new_tokens).tolist())
    utils.synchronize(args.device)
    elapsed = time.time() - t0

    print("\nGenerated tokens for first batch:")
    print(decode(generations[0]))
    print("-----------------------------------------------")
    return generations, elapsed


def run_offline(args, mode: str):
    """
    Runs the offline benchmark. See `run.py kvcache` and `run.py speculative`.

    Args:
        mode (str): 'kvcache' compares naive and KV-cache generation; 'speculative'
        also compares speculative decoding (batch size 1 only)
    """
    utils.setup_torch(args.seed)
    ctx = utils.get_torch_ctx(args.device)
    model = utils.load_model(args.init_from, args.ckpt_path, args.device)
    if args.compile:
        model = torch.compile(model) # requires PyTorch 2.0 (optional)
    enc = utils.get_encoder()
    decode = enc.decode

    x = load_batch(args.start, args.prompt_length, args.batch_size, enc, args.device)
    if mode == 'speculative' and x.size(0) != 1:
        raise ValueError("Speculative decoding only supports --batch_size 1")

    sampling = dict(temperature=args.temperature, top_k=args.top_k)
    results = {}
    results['NAIVE'] = benchmark(
        'naive', lambda x, n: generate(model, x, n, **sampling), x, args, ctx, decode)
    results['KV'] = benchmark(
        'kv cache', lambda x, n: generate_kv(model, x, n, **sampling), x, args, ctx, decode)

    if mode == 'speculative':
        print('Loading draft model')
        draft_model = utils.load_model(args.draft_model, args.ckpt_path, args.device)
        if args.compile:
            draft_model = torch.compile(draft_model) # requires PyTorch 2.0 (optional)
        print('Finished loading draft model')
        results['SPEC'] = benchmark(
            f'speculative (speculative_tokens={args.speculative_tokens})',
            lambda x, n: generate_speculative(model, draft_model, x, n,
                                              num_speculative=args.speculative_tokens, **sampling),
            x, args, ctx, decode)

    tokens_generated = x.size(0) * args.num_samples * args.max_new_tokens
    naive_gens, naive_time = results['NAIVE']
    print()
    for name, (_, elapsed) in results.items():
        speedup = '' if name == 'NAIVE' else f' ({naive_time/elapsed:.3f}x speed-up)'
        print(f'[{name}]'.ljust(8) + f'Generated {tokens_generated} tokens in {elapsed:.3f} seconds '
              f'for {tokens_generated/elapsed:.3f} tok/sec{speedup}.')

    # verify matched predictions
    all_matched = True
    for name, (gens, _) in results.items():
        if name == 'NAIVE':
            continue
        num_matching = sum(1 for a, b in zip(naive_gens, gens) if a == b)
        print(f'NAIVE and {name} generations matched on {num_matching}/{len(naive_gens)} positions')
        all_matched &= num_matching == len(naive_gens)

    if all_matched:
        print('All generations matched! :)')
    else:
        print('Not all generations matched! :(')
        print("NVIDIA's underlying libaries aren't wholly deterministic, so don't worry about getting an exact match as long as you're getting good outputs")
        print("You can set `DEBUG` to `True` in `model.py` to make operations much more deterministic, at the cost of speed")
