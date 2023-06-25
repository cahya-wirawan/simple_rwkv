import logging
import ray
from typing import List
from pathlib import Path

from simple_rwkv.get_models import MODEL, TOKENIZER_PATH, get_model_path

# if RWKV_CUDA_ON='1' then use CUDA kernel for seq mode (much faster)
# these settings must be configured before attempting to import rwkv
from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS

STRATEGIES = {
    "streaming": "cuda fp16i8 *40+ -> cpu fp32 *1",  # Quite slow, take ~3gb VRAM
    "fp16i8": "cuda fp16i8 *40 -> cpu fp32 *1",  # fits the 14b on a T4, quite fast
    "cpu": "cpu fp32 *1",  # requires a lot of RAM
    "gpu-fp16": "cuda fp16",
    "gpu-fp32": "cuda fp32",
}

STRATEGY = STRATEGIES["gpu-fp32"]

logger = logging.getLogger(__file__)

ctx_limit = 4096


def get_model(cfg):
    if Path(cfg.model.title).exists():
        model_path = Path(cfg.model.title)
    else:
        model_path = get_model_path(cfg)

    if cfg.use_ray:
        from simple_rwkv import ray_model
        ray.init() 
        model = ray_model.RayRWKV()
    else:
        model = RWKV(model=model_path, strategy=STRATEGY)  # stream mode w/some static

    if cfg.model.world:
        pipeline = PIPELINE(model, "rwkv_vocab_v20230424")
    else:
        pipeline = PIPELINE(model, str(TOKENIZER_PATH))

    return model, pipeline

def complete(
    instruction,
    model,
    pipeline: PIPELINE,
    prompt="",
    token_count=200,
    temperature=1.0,
    top_p=0.7,
    presencePenalty=0.1,
    countPenalty=0.1,
    stop_words=None,
):
    args = PIPELINE_ARGS(
        temperature=max(0.2, float(temperature)),
        top_p=float(top_p),
        alpha_frequency=countPenalty,
        alpha_presence=presencePenalty,
        token_ban=[],  # ban the generation of some tokens
        token_stop=[0],
        stop_words=stop_words,
    )  # stop generation whenever you see any token here

    for delta in pipeline.igenerate(ctx=instruction, token_count=token_count, args=args):
        yield delta


def embedding(
    inputs: List[str],
    model,
    pipeline,
    temperature=1.0,  # TODO remove
    top_p=0.7,
    presencePenalty=0.1,
    countPenalty=0.1,
):
    PIPELINE_ARGS(
        temperature=max(0.2, float(temperature)),
        top_p=float(top_p),
        alpha_frequency=countPenalty,
        alpha_presence=presencePenalty,
        token_ban=[],  # ban the generation of some tokens
        token_stop=[0],
    )  # stop generation whenever you see any token here

    context = [pipeline.encode(ctx)[-ctx_limit:] for ctx in inputs]
    _, state = model.forward(context[0], None)
    *_, embedding = state

    if len(embedding.shape) == 1:
        embedding = embedding.unsqueeze(0)
    return embedding
