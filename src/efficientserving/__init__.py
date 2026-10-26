"""
efficientserving: a small GPT-2 inference engine and serving stack.

Modules:
    model       -- the GPT-2 model
    kvcache     -- KV cache storage
    generation  -- generation loops (naive, KV cache, padded, speculative)
    server/     -- inference servers (static and continuous batching)
    client      -- Poisson-arrival client that replays a trace against a server
    experiment  -- throughput-latency sweeps over server configurations
    offline     -- single-process generation benchmarks (no server)
    plotting    -- plots of saved stats
    utils       -- model/prompt loading and torch setup shared by everything above
"""
