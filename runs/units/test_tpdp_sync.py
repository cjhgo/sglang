import os
import logging
logging.basicConfig(level=logging.INFO)
from pathlib import Path

abs_base = Path(__file__).resolve().parent
model_id = "deepseek-ai/Janus-Pro-1B"
tmpl_chat_key = "janus-pro"
# model_id = "openbmb/MiniCPM-v-2_6"
# tmpl_chat_key = "minicpmv"
# model_id = f"{abs_base}/../../../../modelbest/sync/checkpoint-1170" 
logging.info(f"will load model from {model_id} ")

import torch
from torch import distributed as dist
from sglang.srt.entrypoints.verl_engine import VerlEngine
from torch.distributed.device_mesh import init_device_mesh



def initialize_global_process_group(timeout_second=36000):
    import torch.distributed
    torch.distributed.init_process_group()

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if torch.distributed.is_initialized():
        torch.cuda.set_device(local_rank)
    # logger.info(f"local_rank: {local_rank}, rank: {rank}, world_size: {world_size}")
    return local_rank, rank, world_size

def gen_mm_prompt(tmpl_chat_key, user_prompt, image_path):
    from sglang.srt.conversation import chat_templates
    conv = chat_templates[tmpl_chat_key].copy()
    image_token = chat_templates[tmpl_chat_key].image_token
    user_prompt = f"{image_token} {user_prompt}"
    conv.append_message(conv.roles[0], user_prompt)
    prompt = conv.get_prompt()

    from PIL import Image
    image = Image.open(image_path).convert("RGB")

    sampling_params = {
        "max_new_tokens": 2048,
        'presence_penalty': 0.0, 'frequency_penalty': 0.0, 
        'repetition_penalty': 1.0, 'temperature': 1.0, 
        'top_k': -1, 'top_p': 1, 'ignore_eos': False
        # "temperature": 0.5,
        # "top_p": 0.95,
    }
    return prompt, image, sampling_params


def load_hf_model(model_id):
    if "janus" in model_id.lower():
        # pip install git+https://github.com/deepseek-ai/Janus.git
        from janus.models import MultiModalityCausalLM
        # from transformers import AutoModelForCausalLM
        load_cls = MultiModalityCausalLM
    else:
        from transformers import AutoModelForCausalLM
        load_cls = AutoModelForCausalLM
    model = load_cls.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(torch.cuda.current_device())

    return model

def test_sglang_spmd():
    assert torch.cuda.device_count() >= 2, "At least 2 GPUs is required to run tp+dp tests."
    _, rank, world_size = initialize_global_process_group()
    if rank == 0:
        print("rank", rank, "world_size", world_size)

    tensor_parallel_size = 2
    data_parallel_size = world_size // tensor_parallel_size
    # device_mesh_kwargs = dict(mesh_shape=(data_parallel_size, tensor_parallel_size), mesh_dim_names=["tp", "dp"])
    device_mesh_kwargs = dict(mesh_shape=(data_parallel_size, tensor_parallel_size), mesh_dim_names=["dp", "tp"])
    inference_device_mesh_cpu = init_device_mesh("cpu", **device_mesh_kwargs)

    for k in ["TORCHELASTIC_USE_AGENT_STORE"]:
        if k in os.environ:
            del os.environ[k]
    print("building sglang rollout engine")
    # print("TP Ranks:", dist.get_process_group_ranks(inference_device_mesh_cpu.get_group("tp")))


    llm = VerlEngine(
        model_path=model_id,
        dtype="bfloat16",
        mem_fraction_static=0.5,
        device_mesh_cpu=inference_device_mesh_cpu["tp"],
        base_gpu_id=inference_device_mesh_cpu.get_local_rank("dp"),
        gpu_id_step=data_parallel_size,
        trust_remote_code=True,
    )

    print("start generation")
    user_prompt = "what is in image 或者说图片里有什么"
    # role_user = "计算"+ str(dist.get_rank())+"的阶乘"

    image_path = abs_base / ".." / "2.jpg"
    prompt, image, sampling_params = gen_mm_prompt(tmpl_chat_key, user_prompt, image_path)
    output = llm.generate(prompt, image_data=image, sampling_params=sampling_params)
    print(output)
    if dist.get_rank() == 0:
        print(output)
    dist.barrier()

    print("load weights from hf model")

    model = load_hf_model(model_id)
    llm.update_weights_from_tensor(model.named_parameters(), load_format=None)

    print("start generation again")
    output = llm.generate(prompt, image_data=image, sampling_params=sampling_params)

    if dist.get_rank() == 0:
        # import ipdb; ipdb.set_trace()
        print(f"rank {dist.get_rank()}:",output)
    dist.barrier()
    llm.shutdown()

    print("Check Pass")
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    #run with
    # `torchrun --nproc_per_node 4  --standalone  test_tpdp_sync.py`
    test_sglang_spmd()