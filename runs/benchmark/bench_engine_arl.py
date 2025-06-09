from pathlib import Path
abs_base = Path(__file__).resolve().parent
import logging
# logging.basicConfig(level=logging.INFO)
import dataclasses
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, TypedDict
from arl_data import load_arl_dataset


@dataclass
class BenchArgs:
    model_path: str = ""
    chat_template: str = ""
    trust_remote_code: bool = True
    batch_size: int = 5
    max_new_tokens: int = 512
    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    ignore_eos: bool = False
    n: int = 1

bench_cpmv = BenchArgs(
    # model_path=f"{abs_base}/../../../../modelbest/sync/checkpoint-1170",
    model_path="openbmb/MiniCPM-v-2_6",
    chat_template="minicpmv",
)
bench_cpmo = BenchArgs(
    model_path="openbmb/MiniCPM-o-2_6",
    chat_template="minicpmo",
)

bench_janus = BenchArgs(
    model_path="deepseek-ai/Janus-Pro-1B",
    chat_template="janus-pro",
)
bench_qwen2_5_vl = BenchArgs(
    model_path="Qwen/Qwen2.5-VL-7B-Instruct",#Qwen/Qwen2.5-VL-7B-Instruct
    chat_template="qwen2-vl",
)

class BatchArl:
    def __init__(self, chat_template="janus-pro"):
        self.dataset = load_arl_dataset()
        self.chat_template = chat_template
        self.cur_index = 0
    
    def _reset_index(self):
        self.cur_index = 0

    def to_prompt(self,item):
        system_promt = item["system_prompt"]
        user_promt = item["user_prompt"]
        from sglang.srt.conversation import chat_templates
        conv = chat_templates[self.chat_template].copy()
        conv.system_message = system_promt
        image_token = conv.image_token
        user_prompt = f"{user_promt} {image_token}"
        conv.append_message(conv.roles[0], user_prompt)
        prompt = conv.get_prompt()
        return prompt

    def get_batch(self, batch_size=100):
        batch = []
        for _ in range(batch_size):
            if self.cur_index >= len(self.dataset):
                self._reset_index()
                break
            item = self.dataset[self.cur_index]
            self.cur_index += 1
            item["prompt"] = self.to_prompt(item)
            batch.append(item)
        return batch

def load_vlm_engine(args: BenchArgs):
    from sglang.srt.server_args import ServerArgs
    import sglang as sgl
    server_args = ServerArgs(
        model_path=args.model_path,
        chat_template=args.chat_template,
        trust_remote_code=args.trust_remote_code,
        mem_fraction_static=0.8,
        # schedule_conservativeness=0.3
    )
    vlm = sgl.Engine(server_args=server_args)
    vlm.release_memory_occupation()
    return vlm

def batch_size_bench(vlm, vqa, BatchInput):
    batch_size_list = [1, 2, 4, 8, 16, 32]#, 64, 128, 256, 512, 3000]
    for batch_size in batch_size_list:
        batch = vqa.get_batch(batch_size=batch_size)
        batch_input = BatchInput(
            prompt=[item["prompt"] for item in batch],
            image_data=[item["image"] for item in batch]
        )
        out = vlm.generate(**dataclasses.asdict(batch_input))
        logging.warning(f"batch_size: {batch_size}, e2e_latency: {out[-1]['meta_info']['e2e_latency']}")
        for item in out:
            print(item)
        import ipdb; ipdb.set_trace()

def sampn_bench(vlm, vqa, BatchInput):
    warmup_batch = vqa.get_batch(batch_size=10)
    batch_100 = vqa.get_batch(batch_size=100)
    batch_200 = []
    for item in batch_100:
        batch_200.append(item)
        batch_200.append(item)

    output_list = []
    batch_inputs = [warmup_batch, batch_200, batch_100]
    for idx, batch_input in enumerate(batch_inputs):
        cur_batch_input = BatchInput(
            prompt=[item["prompt"] for item in batch_input],
            image_data=[item["image"] for item in batch_input]
        )
        if idx == 2:
            cur_batch_input.sampling_params["n"] = 2
        
        begin_time = -time.time()
        out = vlm.generate(**dataclasses.asdict(cur_batch_input))
        duration = time.time() + begin_time
        output_list.append(out)
        print("-"*50)
        logging.warning(f"{idx}: {duration}, batch_size: {len(batch_input)}, e2e_latency: {out[-1]['meta_info']['e2e_latency']}")

def bench_vlm_engine(args: BenchArgs):
    vlm = load_vlm_engine(args)
    vqa = BatchArl(chat_template=args.chat_template)
    sampling_params = {
        'n': args.n, 'max_new_tokens': args.max_new_tokens, 
        'presence_penalty': args.presence_penalty, 'frequency_penalty': args.frequency_penalty, 
        'repetition_penalty': args.repetition_penalty, 'temperature': args.temperature, 
        'top_k': args.top_k, 'top_p': args.top_p, 'ignore_eos': args.ignore_eos
    }
    @dataclass
    class BatchInput:
        prompt: list[str]
        image_data: list[Any]
        sampling_params: dict = dataclasses.field(default_factory=lambda: sampling_params)
    
    batch_size_bench(vlm, vqa, BatchInput)
    # sampn_bench(vlm, vqa, BatchInput)
    import ipdb; ipdb.set_trace()

if __name__ == "__main__":
    # args = bench_qwen2_5_vl
    # args = bench_cpmo
    # args.batch_size = 100
    logging.info("start")
    args = bench_cpmv
    bench_vlm_engine(args)






