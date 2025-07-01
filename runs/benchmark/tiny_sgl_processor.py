from pathlib import Path
abs_base = Path(__file__).resolve().parent
import logging
# logging.basicConfig(level=logging.INFO)
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, TypedDict
import time

# 导入 VQADataset
from vlm_data import VQADataset



@dataclass
class ExpArgs:
    model_path: str = ""
    chat_template: str = ""
    trust_remote_code: bool = True
    batch_size: int = 10

exp_cpmv = ExpArgs(
    # model_path="openbmb/MiniCPM-v-2_6",
    model_path=f"{abs_base}/../../../../modelbest/sync/checkpoint-1170",
    chat_template="minicpmv",
)

exp_janus = ExpArgs(
    model_path="deepseek-ai/Janus-Pro-1B",
    chat_template="janus-pro",
)



def test_sgl_processor(exp_args: ExpArgs):
    from sglang.srt.hf_transformers_utils import get_processor, get_tokenizer
    from sglang.srt.conversation import chat_templates
    from sglang.srt.managers.multimodal_processor import (
    get_dummy_processor, get_mm_processor, import_processors)
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.managers.multimodal_processors.base_processor import MultimodalSpecialTokens
    
    print(f"加载 SGLang processor: {exp_args.model_path}")
    
    # 获取 processor 和 tokenizer
    processor = get_processor(
        exp_args.model_path,
        trust_remote_code=True,
    )
    tokenizer = get_tokenizer(
        exp_args.model_path,
        tokenizer_mode="auto",
        trust_remote_code=True,
    )
    
    # 创建 ServerArgs 和 ModelConfig（参考 fixcpmv.py）
    server_args = ServerArgs(
        model_path=exp_args.model_path,
        trust_remote_code=exp_args.trust_remote_code,
    )
    model_config = ModelConfig.from_server_args(server_args)
    
    # 导入 processors 并创建 mm_processor（参考 tokenizer_manager.py）
    import_processors()
    _processor = get_processor(
        server_args.tokenizer_path,
        tokenizer_mode=server_args.tokenizer_mode,
        trust_remote_code=server_args.trust_remote_code,
        revision=server_args.revision,
        use_fast=not server_args.disable_fast_image_processor,
    )
    
    # 创建 mm_processor
    mm_processor = get_mm_processor(
        model_config.hf_config, server_args, _processor
    )
    
    print(f"成功创建 mm_processor: {type(mm_processor)}")
    
    # 获取测试数据
    vqa = VQADataset(chat_template="minicpmv")
    batch = vqa.get_batch(batch_size=exp_args.batch_size)
    input_text = [item["prompt"] for item in batch]
    image_data = [[item["image"]] for item in batch]
    begin = -time.time()
    hf_results = mm_processor.process_data_task(input_text, image_data)
    print(f"Processor time: {time.time() + begin}")
    max_req_input_len = 1024
    multimodal_tokens = MultimodalSpecialTokens(
        image_token=mm_processor.image_token, audio_token=mm_processor.audio_token
    )
    mm_result = mm_processor.load_mm_data(input_text[0], image_data=image_data[0], max_req_input_len=max_req_input_len, multimodal_tokens=multimodal_tokens)

    # 直接使用 asyncio.run 调用异步方法
    import asyncio
    print("开始异步调用 process_mm_data_async...")
    
    # 使用 lambda 创建 mock request_obj
    request_obj = lambda: None
    request_obj.audio_data = []
    
    async_begin = -time.time()
    sgl_result = asyncio.run(mm_processor.process_mm_data_async(
        image_data=image_data[0],
        input_text=input_text[0],
        request_obj=request_obj,
        max_req_input_len=max_req_input_len
    ))
    print(f"Async processor time: {time.time() + async_begin}")
    
    import ipdb; ipdb.set_trace()
    print(f"开始测试 SGLang processor，batch_size={exp_args.batch_size}")
    print(f"mm_processor 已成功创建并可以使用")



if __name__ == "__main__":
    logging.info("start")
    
    # 测试不同的 processor
    print("="*50)
    print("测试 Processor 性能")
    print("="*50)
    
    test_sgl_processor(exp_cpmv)
    print("\n" + "="*50 + "\n")
    






