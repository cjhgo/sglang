from pathlib import Path
abs_base = Path(__file__).resolve().parent
import logging
# logging.basicConfig(level=logging.INFO)
import dataclasses
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, TypedDict


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

class VQADataset:
    def __init__(self, chat_template="janus-pro"):
        from datasets import load_dataset
        self.dataset = load_dataset("OpenStellarTeam/Chinese-SimpleVQA")
        self.chat_template = chat_template
        self._reset_iter()

    def _reset_iter(self):
        self._iter = iter(self.dataset["train"])

    def base64_to_pil(self, base64_str):
        from PIL import Image
        import base64
        import io 
        image_data = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_data))
        return image

    def __iter__(self):
        self._reset_iter()
        return self

    def __next__(self):
        item = next(self._iter)
        image = self.base64_to_pil(item["image_base64"])
        item = {
            "image": image,
            "question": item["recognition_question"],
            "answer": item["recognition_answer"],
        }
        item["prompt"] = self.to_prompt(item)
        return item

    def to_prompt(self,item):
        system_prompt = "You are a helpful language and vision assistant. "\
                "You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using chinese language."
        user_prompt = item["question"]
        from sglang.srt.conversation import chat_templates
        conv = chat_templates[self.chat_template].copy()
        # conv.system_message = system_prompt
        image_token = conv.image_token
        user_prompt = f"{user_prompt} {image_token}"
        conv.append_message(conv.roles[0], user_prompt)
        prompt = conv.get_prompt()
        return prompt

    def get_batch(self, batch_size=100):
        batch = []
        for _ in range(batch_size):
            try:
                batch.append(next(self))
            except StopIteration:
                self._reset_iter()
                batch.append(next(self))
        return batch

def load_vlm_engine(args: BenchArgs):
    from sglang.srt.server_args import ServerArgs
    import sglang as sgl
    server_args = ServerArgs(
        model_path=args.model_path,
        chat_template=args.chat_template,
        trust_remote_code=args.trust_remote_code,
        mem_fraction_static=0.8,
        mm_attention_backend="fa3"
        # schedule_conservativeness=0.3
    )
    vlm = sgl.Engine(server_args=server_args)
    vlm.release_memory_occupation()
    return vlm

def batch_size_bench(vlm, vqa, BatchInput):
    batch_size_list = [10, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 3000]
    for batch_size in batch_size_list:
        batch = vqa.get_batch(batch_size=batch_size)
        batch_input = BatchInput(
            prompt=[item["prompt"] for item in batch],
            image_data=[item["image"] for item in batch]
        )
        out = vlm.generate(**dataclasses.asdict(batch_input))
        if batch_size < 10:
            # import ipdb; ipdb.set_trace()
            pass
        logging.warning(f"batch_size: {batch_size}, e2e_latency: {out[-1]['meta_info']['e2e_latency']}")

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
    vqa = VQADataset(chat_template=args.chat_template)
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






