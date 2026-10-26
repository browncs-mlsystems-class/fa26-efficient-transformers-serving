"""
Setup shared by the offline benchmarks, the client, and the servers: torch flags,
mixed precision, model loading, prompt loading, and warmup.
"""
from __future__ import annotations

import json
import os
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import tiktoken
import torch

from efficientserving.model import DEBUG, GPT, GPTConfig

# Where pretrained GPT-2 weights are cached after the first download: <project root>/weights,
# unless the EFFICIENTSERVING_WEIGHTS_DIR environment variable says otherwise.
WEIGHTS_DIR = Path(os.environ.get("EFFICIENTSERVING_WEIGHTS_DIR",
                                  Path(__file__).resolve().parents[2] / "weights"))


DTYPES = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}
_dtype_override: torch.dtype | None = None


def set_dtype(name: str):
    """Forces the dtype the model runs in ('bf16', 'fp16' or 'fp32'); 'auto' picks one (see `get_dtype`)."""
    global _dtype_override
    _dtype_override = None if name == 'auto' else DTYPES[name]


def get_dtype() -> torch.dtype:
    if _dtype_override is not None:
        return _dtype_override
    if DEBUG:
        return torch.float32
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def get_torch_ctx(device: str):
    """Mixed-precision context to run the model under."""
    ptdtype = get_dtype()
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    print(f'Running on {device} in {ptdtype}')
    return nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)


def setup_torch(seed: int):
    """Seeds torch and sets the determinism flags controlled by `DEBUG` in model.py."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if DEBUG:
        # cuBLAS needs a fixed workspace to be deterministic; torch.use_deterministic_algorithms errors without it
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cuda.matmul.allow_tf32 = False # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = False # allow tf32 on cudnn
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn


def synchronize(device: str):
    """Waits for queued GPU work to finish, so timings include it."""
    if 'cuda' in device:
        torch.cuda.synchronize()


def load_checkpoint(ckpt_path: str | Path, device: str) -> GPT:
    """Load a GPT model from a checkpoint with 'model_args' (its GPTConfig) and 'model' (its state dict)."""
    # load onto the CPU: the model is built on the CPU and moved to `device` once, by `load_model`
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    # the causal mask buffers (attn.bias) are rebuilt by the model, so checkpoints may leave them out
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    missing = [k for k in missing if not k.endswith('.attn.bias')]
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint {ckpt_path} does not match the model: "
                           f"missing keys {missing}, unexpected keys {unexpected}")
    return model


def load_pretrained(model_type: str, device: str) -> GPT:
    """
    Load a pretrained gpt2 variant, downloading it from Hugging Face only the first
    time; afterwards it is loaded from WEIGHTS_DIR/<model_type>.pt.
    """
    path = WEIGHTS_DIR / f"{model_type}.pt"
    if path.exists():
        print(f"loading cached weights from {path}")
        return load_checkpoint(path, device)

    model = GPT.from_pretrained(model_type, dict(dropout=0.0))
    state_dict = {k: v for k, v in model.state_dict().items() if not k.endswith('.attn.bias')}
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    # write to a temporary file first, so a job that dies mid-save can't leave a broken cache entry
    tmp_path = path.with_suffix(f".tmp{os.getpid()}")
    torch.save({'model_args': asdict(model.config), 'model': state_dict}, tmp_path)
    os.replace(tmp_path, path)
    print(f"cached weights in {path}")
    return model


def load_model(init_from: str, ckpt_path: str, device: str) -> GPT:
    """Load a GPT model, either from a checkpoint ('resume') or a pretrained gpt2 variant."""
    if init_from == "resume":
        model = load_checkpoint(ckpt_path, device)
    elif init_from.startswith('gpt2'):
        model = load_pretrained(init_from, device)
    else:
        raise ValueError(f"Unknown init_from: {init_from!r} (use 'resume' or a gpt2 variant)")
    model.eval()
    model.to(device)
    return model


def get_encoder() -> tiktoken.Encoding:
    return tiktoken.get_encoding("gpt2")


def encode(enc: tiktoken.Encoding, text: str) -> list[int]:
    return enc.encode(text, allowed_special={"<|endoftext|>"})


def load_prompts(trace_file: str, num_requests: int | None = None) -> list[str]:
    """Load prompts from a JSONL file. Each line should have a 'prompt' field."""
    prompts = []
    with open(trace_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line)['prompt'])
    if num_requests is not None:
        prompts = prompts[:num_requests]
    return prompts


def warmup_model(model: GPT, prompts: list[str], device: str, enc: tiktoken.Encoding,
                 use_kv: bool = False):
    """Runs a few short generations so CUDA kernels are loaded before anything is timed.
    Warms up naive generation, and also KV-cache generation if `use_kv`."""
    from efficientserving.generation import generate, generate_kv

    ctx = get_torch_ctx(device)
    warmup_prompt_length = 64
    batch_size = 8
    warmup_max_new_tokens = 4
    num_warmup = 2
    start_ids = [encode(enc, x) for x in prompts]
    x = torch.tensor([x[:warmup_prompt_length] for x in start_ids if len(x) >
                      warmup_prompt_length][:batch_size],
                     dtype=torch.long, device=device)
    print('Starting warmup... ', end='')
    with torch.no_grad():
        with ctx:
            for _ in range(num_warmup):
                generate(model, x, warmup_max_new_tokens, temperature=0.4, top_k=200)
                if use_kv:
                    generate_kv(model, x, warmup_max_new_tokens, temperature=0.4, top_k=200)
    synchronize(device)
    print('Finished')
