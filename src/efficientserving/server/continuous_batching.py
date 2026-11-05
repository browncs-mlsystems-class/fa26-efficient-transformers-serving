from efficientserving.model import GPT
from efficientserving.server.interface import InferenceServer


class ContinuousBatchServer(InferenceServer):
    """Instantiation of InferenceServer with continuous batching.

    Students must fill in custom_scheduler_step and custom_cleanup
    according to the continuous batching spec in the handout.
    """

    def __init__(self, model: GPT, device: str, use_kv: bool = True,
                 num_slots: int = 8, max_seq_len: int = 512):
        if not use_kv:
            raise ValueError("Continuous batching requires use_kv=True")
        super().__init__(model, device, use_kv)
        self.num_slots = num_slots
        self.max_seq_len = max_seq_len

    async def custom_scheduler_step(self):
        # TODO (continuous batching): Implement! Append one sample per step to self.memory_log
        # with keys 'time', 'cached' and 'reserved' (see handout Part 3).
        raise NotImplementedError("Continuous batching is not implemented yet")

    async def custom_cleanup(self):
        # TODO (continuous batching): Implement!
        raise NotImplementedError("Continuous batching is not implemented yet")
