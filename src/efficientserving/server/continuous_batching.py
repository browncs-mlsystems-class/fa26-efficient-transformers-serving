from efficientserving.model import GPT
from efficientserving.server.interface import InferenceServer


class ContinuousBatchServer(InferenceServer):
    """Instantiation of InferenceServer with continuous batching.

    Students must fill in custom_scheduler_step and custom_cleanup
    according to the continuous batching spec in the handout.
    """

    def __init__(self, model: GPT, device: str, use_kv: bool = False):
        super().__init__(model, device, use_kv)

    async def custom_scheduler_step(self):
        # TODO (continuous batching): Implement!
        raise NotImplementedError("Continuous batching is not implemented yet")

    async def custom_cleanup(self):
        # TODO (continuous batching): Implement!
        raise NotImplementedError("Continuous batching is not implemented yet")
