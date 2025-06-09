import os
import sys
import logging
logging.basicConfig(level=logging.INFO)
from pathlib import Path

abs_base = Path(__file__).resolve().parent
model_id = f"{abs_base}/../../../../modelbest/sync/checkpoint-1170"
logging.info(f"will load model from {model_id}")

import torch
from torch import distributed as dist
from sglang.srt.entrypoints.verl_engine import VerlEngine
from torch.distributed.device_mesh import init_device_mesh

def initialize_global_process_group(timeout_second=36000):
    from datetime import timedelta

    import torch.distributed

    # NOTE MODIFIED should provide backend=None to have nccl+gloo
    # torch.distributed.init_process_group('nccl', timeout=timedelta(seconds=timeout_second))
    torch.distributed.init_process_group(timeout=timedelta(seconds=timeout_second))

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if torch.distributed.is_initialized():
        torch.cuda.set_device(local_rank)
    logging.info(f"local_rank: {local_rank}, rank: {rank}, world_size: {world_size}")
    return local_rank, rank, world_size


def test_sglang_spmd():
    assert torch.cuda.device_count() >= 2, "At least 2 GPUs is required to run tp+dp tests."
    initialize_global_process_group()


    local_model_path = model_id

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    def _log(text):
        import datetime
        t = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{t}] [rank={rank}] {text}")

    _log(
        f'start {local_rank=} {rank=} {world_size=} {sys.argv=} {os.environ.get("CUDA_VISIBLE_DEVICES")}'
    )

    tp_size = 4
    dp_size = 2
    assert world_size == tp_size * dp_size

    device_mesh_kwargs = dict(
        mesh_shape=(tp_size, dp_size, 1), mesh_dim_names=["tp", "dp", "pp"]
    )
    device_mesh_cpu = init_device_mesh("cpu", **device_mesh_kwargs)
    _log(f"{device_mesh_cpu=}")

    tp_rank = device_mesh_cpu.get_local_rank("tp")
    dp_rank = device_mesh_cpu.get_local_rank("dp")
    _log(f"{tp_rank=} {tp_size=} ; {dp_rank=} {dp_size=}")
    # if dist.get_rank() == 0:
    #   import ipdb; ipdb.set_trace()
    # dist.barrier()

    for k in ["TORCHELASTIC_USE_AGENT_STORE"]:
        if k in os.environ:
            del os.environ[k]
    print("building sglang rollout engine")
    llm = VerlEngine(
        model_path=local_model_path,
        dtype="bfloat16",
        mem_fraction_static=0.5,
        device_mesh_cpu=device_mesh_cpu["tp"],
        base_gpu_id=dp_rank,
        gpu_id_step=dp_size,
        trust_remote_code=True,
    )

    llm.release_memory_occupation()

    print("start generation")
    role_user = "what is in image 或者说图片里有什么"
    prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n(<image>./</image>)\n{role_user}<|im_end|>\n<|im_start|>assistant\n"
    from PIL import Image
    img = Image.open("../2.jpg")
    print(img.size)
    output = llm.generate(prompt, image_data=img)
    if dist.get_rank() == 0:
      print(output)
    dist.barrier()

    print("Check Pass")


if __name__ == "__main__":
    #run with
    # `torchrun --nproc_per_node 4  --standalone  test_tpdp.py`
    test_sglang_spmd()