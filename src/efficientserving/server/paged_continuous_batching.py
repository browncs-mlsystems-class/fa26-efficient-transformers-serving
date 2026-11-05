from efficientserving.model import GPT
from efficientserving.server.interface import InferenceServer


class PagedContinuousBatchServer(InferenceServer):
    """Instantiation of InferenceServer with continuous batching over a paged KV cache.

    Students must fill in custom_scheduler_step and custom_cleanup
    according to the paged KV cache spec in the handout.
    """

    def __init__(self, model: GPT, device: str, use_kv: bool = True,
                 num_blocks: int = 256, block_size: int = 16, max_seq_len: int = 512):
        if not use_kv:
            raise ValueError("Paged continuous batching requires use_kv=True")
        super().__init__(model, device, use_kv)
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.max_seq_len = max_seq_len

    async def custom_scheduler_step(self):
        # TODO (paged batching): Implement! Append one sample per step to self.memory_log with
        # keys 'time', 'cached' and 'reserved', as ContinuousBatchServer does.
        raise NotImplementedError("Paged continuous batching is not implemented yet")

    async def custom_cleanup(self):
        # TODO (paged batching): Implement!
        raise NotImplementedError("Paged continuous batching is not implemented yet")
