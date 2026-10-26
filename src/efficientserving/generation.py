"""
Generation loops: naive, KV cache, padded (mixed-length batches), and speculative decoding.

All functions should be used with the model in evaluation mode (`model.eval()`).
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from efficientserving.kvcache import KVCache
from efficientserving.model import GPT


def sample_next(logits: torch.Tensor, temperature: float, top_k: int | None) -> torch.Tensor:
    """
    Samples the next token id from final-position logits.

    Args:
        logits (torch.Tensor): logits at the final position, with shape (batch size, vocab)
        temperature (float): a scaling factor in [0, 1] for token generation,
        such that values close to 1 make the output distribution more uniform (random),
        while values close to 0 make it more concentrated
        top_k (int | None): limits the token sampling to the top `k`
        highest-probability tokens if specified, otherwise, no restriction is applied

    Returns:
        torch.Tensor: the sampled token ids, with shape (batch size, 1)
    """
    if top_k == 1:
        return torch.argmax(logits, dim=-1, keepdim=True) # using argmax here preserves RNG state
    logits = logits / temperature
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits = logits.masked_fill(logits < v[:, [-1]], -float('Inf'))
    # apply softmax to convert logits to (normalized) probabilities
    probs = F.softmax(logits, dim=-1)
    # sample from the distribution
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate(model: GPT, idx: torch.Tensor, max_new_tokens: int,
             temperature: float = 1.0, top_k: int | None = None) -> torch.Tensor:
    """
    Receives a conditioning sequence of tokens and predicts the following tokens
    in the sequence, feeding the predictions back into the model each time.

    Args:
        model (GPT): the model to generate with
        idx (torch.Tensor): the conditioning sequence of tokens, of shape
        (batch size, sequence length)
        max_new_tokens (int): maximum number of new tokens to predict
        temperature (float): see `sample_next`; defaults to 1
        top_k (int | None): see `sample_next`; defaults to None

    Returns:
        torch.Tensor: the extended sequence with the generated predictions
        concatenated on the end of the input sequence
    """
    block_size = model.config.block_size
    for _ in range(max_new_tokens):
        # if the sequence context is growing too long we must crop it at block_size
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        # forward the model to get the logits for the index in the sequence
        logits, _ = model(idx_cond)
        idx_next = sample_next(logits[:, -1, :], temperature, top_k)
        # append sampled index to the running sequence and continue
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


@torch.no_grad()
def generate_kv(model: GPT, idx: torch.Tensor, max_new_tokens: int,
                temperature: float = 1.0, top_k: int | None = None) -> torch.Tensor:
    """
    Like `generate`, but uses a KV cache so each step only runs the model on the newest
    token. Every prompt in the batch must have the same length.

    Args:
        model (GPT): the model to generate with
        idx (torch.Tensor): the conditioning sequence of tokens, of shape
        (batch size, sequence length)
        max_new_tokens (int): maximum number of new tokens to predict
        temperature (float): see `sample_next`; defaults to 1
        top_k (int | None): see `sample_next`; defaults to None

    Returns:
        torch.Tensor: the extended sequence with the generated predictions
        concatenated on the end of the input sequence
    """
    # TODO (KV cache): Implement!
    raise NotImplementedError


@torch.no_grad()
def generate_speculative(model: GPT, draft_model: GPT, idx: torch.Tensor,
                         max_new_tokens: int, temperature: float = 1.0,
                         top_k: int | None = None, num_speculative: int = 4) -> torch.Tensor:
    """
    Like `generate`, but uses speculative decoding: a smaller draft model guesses the
    next `num_speculative` tokens, and the main model checks all the guesses in one
    forward pass.
    Note that in the tests, your speculative decoding method will ONLY be
    called on 1 sequence at a time; making it working with batching is out
    of scope. Use a KV cache for the draft model.

    Args:
        model (GPT): the main model
        draft_model (GPT): draft model to speculative decode with
        idx (torch.Tensor): the conditioning sequence of tokens, of shape (1, sequence length)
        max_new_tokens (int): maximum number of new tokens to predict
        temperature (float): see `sample_next`; defaults to 1
        top_k (int | None): see `sample_next`; defaults to None
        num_speculative (int): number of speculative tokens to generate using
        draft model; defaults to 4

    Returns:
        torch.Tensor: the extended sequence with the generated predictions
        concatenated on the end of the input sequence
    """
    # outside of batch size 1 out of scope
    if idx.size(0) != 1:
        raise ValueError("Speculative decoding only works with batch size 1")

    # TODO (speculative decoding): Implement!
    raise NotImplementedError


@torch.no_grad()
def generate_padded(model: GPT, idx: torch.Tensor, attention_mask: torch.Tensor,
                    max_new_tokens: int, temperature: float = 1.0,
                    top_k: int | None = None) -> torch.Tensor:
    """
    Like `generate`, but for a left-padded batch of prompts of different lengths.

    Args:
        model (GPT): the model to generate with
        idx (torch.Tensor): left-padded prompts, of shape (batch size, sequence length)
        attention_mask (torch.Tensor): (batch size, sequence length) mask with 1 for
        real tokens and 0 for padding
        max_new_tokens (int): number of new tokens to predict for every prompt
        temperature (float): see `sample_next`; defaults to 1
        top_k (int | None): see `sample_next`; defaults to None

    Returns:
        torch.Tensor: the padded prompts with the generated predictions concatenated
        on the end, of shape (batch size, sequence length + max_new_tokens)
    """
    # TODO (static batching): Implement!
    raise NotImplementedError
