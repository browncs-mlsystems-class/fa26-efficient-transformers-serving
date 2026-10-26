from __future__ import annotations

import asyncio
import time
from abc import ABC
from dataclasses import dataclass, field

import torch

from efficientserving.model import GPT


@dataclass
class InferenceRequest:
    idx: torch.Tensor # (1, any_prompt_len) -- 1 request of any length
    max_new_tokens: int # number of new tokens to sample
    temperature: float = 1.0
    top_k: int | None = None
    request_id: int = field(default=0)
    arrival_time: float = field(default=0.0)
    prefill_completion_time: float = field(default=0.0)
    decode_completion_time: float = field(default=0.0)
    future: asyncio.Future = field(default_factory=asyncio.Future)

class InferenceServer(ABC):
    """
    Interface for all inference server strategies you will implement.
    Subclasses will implement their own engine loop logic and
    relevant batching/admission logic.

    Every server must support both values of `use_kv`: generating with the KV
    cache (True) or recomputing the whole sequence at every step (False).
    """

    def __init__(self, model: GPT, device: str, use_kv: bool = False):
        self.model = model
        self.device = device
        self.use_kv = use_kv
        self.queue: asyncio.Queue[InferenceRequest] = asyncio.Queue()
        self.response_queue: asyncio.Queue[InferenceRequest] = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self._stop_signal = asyncio.Event()
        self._next_id = 0

    async def start(self):
        self._task = asyncio.create_task(self.run_engine_loop())

    async def stop(self):
        self._stop_signal.set()

    async def shutdown(self, timeout: float | None = 10.0):
        """Stops accepting requests and waits for the engine loop (and its cleanup) to finish."""
        await self.stop()
        if self._task is None:
            return
        try:
            await asyncio.wait_for(self._task, timeout=timeout)
        except asyncio.TimeoutError:
            self._task.cancel()

    async def submit_nowait(self, idx: torch.Tensor, max_new_tokens: int,
                            temperature: float = 1.0,
                            top_k: int | None = None) -> InferenceRequest:
        if self._stop_signal.is_set():
            raise ValueError("Stop signal set, server not accepting requests.")
        req = InferenceRequest(
            idx=idx, max_new_tokens=max_new_tokens,
            temperature=temperature, top_k=top_k,
            request_id=self._next_id,
            arrival_time=asyncio.get_running_loop().time(),
        )
        self._next_id += 1
        print(f"[Server] Queued request {req.request_id} at time {time.time():.3f}")
        await self.queue.put(req)
        return req

    async def submit(self, idx: torch.Tensor, max_new_tokens: int,
                      temperature: float = 1.0, top_k: int | None = None) -> torch.Tensor:
        req = await self.submit_nowait(idx, max_new_tokens, temperature, top_k)
        return await req.future

    def _complete(self, req: InferenceRequest, result: torch.Tensor):
        req.decode_completion_time = asyncio.get_running_loop().time()
        req.future.set_result(result)
        self.response_queue.put_nowait(req)

    def _fail(self, req: InferenceRequest, exc: BaseException):
        req.decode_completion_time = asyncio.get_running_loop().time()
        if not req.future.done():
            req.future.set_exception(exc)
        self.response_queue.put_nowait(req)

    async def run_engine_loop(self):
        while not self._stop_signal.is_set():
            await self.custom_scheduler_step()
        await self.custom_cleanup()

    async def custom_scheduler_step(self):
        """Runs one step/tick of scheduler.

        Put logic to collect new requests,
        run inference, etc."""
        raise NotImplementedError("Must be implemented in all subclasses.")

    async def custom_cleanup(self):
        """Cleanup function which runs after the final step.

        Put any custom logic here to drain requests.
        """
        raise NotImplementedError("Must be implemented in all subclasses.")
