"""
KV cache storage.

A cache holds the keys and values of every token a batch has processed so far, for
every layer, so each decoding step only has to run the model on the newest token.
"""
import torch


class KVCache:
    # TODO (KV cache): Implement!
    pass
