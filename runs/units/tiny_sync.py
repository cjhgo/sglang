import logging
logging.basicConfig(level=logging.INFO)
from dataclasses import dataclass
import torch
from typing import Any, Dict, List, Tuple, TypedDict


@dataclass
class ExpArgs:
    model_path: str = ""
    chat_template: str = ""
    trust_remote_code: bool = True

exp_cpmv = ExpArgs(
    model_path="openbmb/MiniCPM-v-2_6",
    chat_template="minicpmv",
)

exp_janus = ExpArgs(
    model_path="deepseek-ai/Janus-Pro-1B",
    chat_template="janus-pro",
)


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

def gen_prompt(args: ExpArgs):
    from sglang.srt.conversation import chat_templates
    conv = chat_templates[args.chat_template].copy()
    conv.append_message(conv.roles[0], "介绍一下北京")
    return conv.get_prompt()

def exp_sgl_wsync(args: ExpArgs):
    from sglang.srt.server_args import ServerArgs
    import sglang as sgl
    server_args = ServerArgs(
        model_path=args.model_path,
        chat_template=args.chat_template,
        trust_remote_code=args.trust_remote_code,
        mem_fraction_static=0.4,
    )
    vlm = sgl.Engine(server_args=server_args)
    prompt = gen_prompt(args)
    print(vlm.generate(prompt))
    vlm.release_memory_occupation()
    model = load_hf_model(args.model_path)
    vlm.update_weights_from_tensor(list(model.named_parameters()) ,load_format=None)
    print(vlm.generate(prompt))


if __name__ == "__main__":
    logging.info("start")
    args = exp_cpmv
    exp_sgl_wsync(args)






