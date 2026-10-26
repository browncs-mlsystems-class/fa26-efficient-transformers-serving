from efficientserving import utils
from efficientserving.model import GPT
from efficientserving.server.interface import InferenceServer


class StaticBatchServer(InferenceServer):
    """Instantiation of InferenceServer with static batching technique.

    Students must fill in custom_scheduler_step and custom_cleanup
    according to the static batching spec in the handout.
    """

    def __init__(self, model: GPT, device: str, use_kv: bool = False,
                 batch_size: int = 1, max_wait_time_s: float = 60.0):
        super().__init__(model, device, use_kv)
        self.batch_size = batch_size
        self.max_wait_time_s = max_wait_time_s
        self.ctx = utils.get_torch_ctx(device)

    async def custom_scheduler_step(self):
        # TODO (static batching): Implement!
        raise NotImplementedError

    async def custom_cleanup(self):
        # TODO (static batching): Implement!
        raise NotImplementedError
