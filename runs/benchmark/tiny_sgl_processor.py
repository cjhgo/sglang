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
    max_req_input_len: int = 1024

exp_cpmv = ExpArgs(
    # model_path="openbmb/MiniCPM-v-2_6",
    model_path=f"{abs_base}/../../../../modelbest/sync/checkpoint-1170",
    chat_template="minicpmv",
)

exp_janus = ExpArgs(
    model_path="deepseek-ai/Janus-Pro-1B",
    chat_template="janus-pro",
)

exp_qwen2vl = ExpArgs(
    model_path="Qwen/Qwen2.5-VL-3B-Instruct",
    chat_template="qwen2-vl",
)



def create_processor_components(exp_args: ExpArgs):
    """创建 processor 相关组件"""
    from sglang.srt.hf_transformers_utils import get_processor, get_tokenizer
    from sglang.srt.managers.multimodal_processor import (
        get_dummy_processor, get_mm_processor, import_processors)
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.multimodal.processors.base_processor import (
        BaseMultimodalProcessor, MultimodalSpecialTokens,)
    
    print(f"加载 SGLang processor: {exp_args.model_path}")
    
    # 获取 processor 和 tokenizer
    processor = get_processor(
        exp_args.model_path,
        trust_remote_code=True,
    )
    tokenizer = get_tokenizer(
        exp_args.model_path,
        trust_remote_code=True,
    )
    
    # 创建 ServerArgs 和 ModelConfig
    server_args = ServerArgs(
        model_path=exp_args.model_path,
        trust_remote_code=exp_args.trust_remote_code,
    )
    model_config = ModelConfig.from_server_args(server_args)
    
    # 导入 processors 并创建 mm_processor
    import_processors()
    _processor = get_processor(
        server_args.tokenizer_path,
        trust_remote_code=server_args.trust_remote_code,
    )
    
    # 创建 mm_processor
    mm_processor = get_mm_processor(
        model_config.hf_config, server_args, _processor
    )
    
    print(f"成功创建 mm_processor: {type(mm_processor)}")
    
    return mm_processor, processor, tokenizer


def create_dataset(exp_args: ExpArgs):
    """创建数据集"""
    # 获取测试数据
    vqa = VQADataset(chat_template=exp_args.chat_template)
    batch = vqa.get_batch(batch_size=exp_args.batch_size)
    input_text = [item["prompt"] for item in batch]
    image_data = [[item["image"]] for item in batch]
    one_req = (input_text[0], image_data[0])
    
    print(f"成功创建数据集，batch_size={exp_args.batch_size}")
    
    return vqa, batch, input_text, image_data, one_req


def call_processor(exp_args: ExpArgs, mm_processor, one_req):
    """调用 processor 进行处理"""
    from sglang.srt.multimodal.processors.base_processor import MultimodalSpecialTokens
    import asyncio
    import time
    
    if "qwen" in exp_args.chat_template.lower():
        multimodal_tokens=MultimodalSpecialTokens(
            image_token=mm_processor.IMAGE_TOKEN,
            image_token_regex=mm_processor.IMAGE_TOKEN_REGEX,
        )
    elif "minicpm" in exp_args.chat_template.lower():
        multimodal_tokens = MultimodalSpecialTokens(
            image_token=mm_processor.image_token, 
            audio_token=mm_processor.audio_token
        )
    else:
        raise ValueError(f"Unsupported model: {exp_args.model_path}")
    
    # 测试 load_mm_data
    mm_result = mm_processor.load_mm_data(
        one_req[0], 
        audio_data=[], 
        image_data=one_req[1], 
        max_req_input_len=exp_args.max_req_input_len, 
        multimodal_tokens=multimodal_tokens
    )

    # 直接使用 asyncio.run 调用异步方法
    print("开始异步调用 process_mm_data_async...")
    
    # 使用 lambda 创建 mock request_obj
    request_obj = lambda: None
    request_obj.audio_data = []
    
    async_begin = -time.time()
    sgl_result = asyncio.run(mm_processor.process_mm_data_async(
        image_data=one_req[1],
        audio_data=[],
        request_obj=request_obj,
        input_text=one_req[0],
        max_req_input_len=exp_args.max_req_input_len
    ))
    print(f"Async processor time: {time.time() + async_begin}")
    
    return mm_result, sgl_result


def test_sgl_processor(exp_args: ExpArgs):
    """测试 SGLang processor 的主函数"""
    print(f"开始测试 SGLang processor，batch_size={exp_args.batch_size}")
    
    # 1. 创建 processor 相关组件
    mm_processor, processor, tokenizer = create_processor_components(exp_args)
    
    # 2. 创建数据集
    vqa, batch, input_text, image_data, one_req = create_dataset(exp_args)
    
    # 3. 调用 processor
    mm_result, sgl_result = call_processor(exp_args, mm_processor, one_req)
    
    # 调试断点
    # import ipdb; ipdb.set_trace()
    print(f"mm_processor 已成功创建并可以使用")


def test_qwenvl_prec_processor(exp_args: ExpArgs):
    """测试 QwenVL 预计算特征处理"""
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from PIL import Image
    import requests
    from io import BytesIO
    import asyncio
    import time
    from sglang.srt.multimodal.processors.base_processor import MultimodalSpecialTokens
    
    # 测试图片 URL
    TEST_IMAGE_URL = "https://github.com/sgl-project/sglang/blob/main/test/lang/example_image.png?raw=true"
    
    print(f"开始测试 QwenVL 预计算特征处理: {exp_args.model_path}")
    
    # 1. 创建 processor 相关组件
    mm_processor, processor, tokenizer = create_processor_components(exp_args)
    
    # 2. 下载测试图片
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    response = requests.get(TEST_IMAGE_URL)
    main_image = Image.open(BytesIO(response.content))
    
    # 3. 使用 HF processor 处理输入
    text = "What's in this picture?"
    processor_output = processor(
        text=[text],
        images=[main_image],
        return_tensors="pt",
    ).to(device)
    
    print(f"HF processor 输出: {processor_output.keys()}")
    
    # 4. 创建预计算特征
    visual_model = (
        Qwen2_5_VLForConditionalGeneration.from_pretrained(
            exp_args.model_path, torch_dtype=torch.bfloat16
        )
        .eval()
        .visual.to(device)
    )
    
    with torch.inference_mode():
        precomputed_features = visual_model(
            processor_output["pixel_values"], 
            processor_output["image_grid_thw"]
        )
    
    print(f"预计算特征形状: {precomputed_features.shape}")
    
    # 5. 创建预计算图像数据
    precomputed_image_data = dict(
        modality="IMAGE",
        precomputed_features=precomputed_features,
    )
    
    # 6. 调用 mm_processor.process_mm_data_async
    multimodal_tokens = MultimodalSpecialTokens(
        image_token=mm_processor.IMAGE_TOKEN,
        image_token_regex=mm_processor.IMAGE_TOKEN_REGEX,
    )
    
    # 使用 lambda 创建 mock request_obj
    request_obj = lambda: None
    request_obj.audio_data = []
    
    print("开始异步调用 process_mm_data_async 处理预计算特征...")
    
    async_begin = -time.time()
    sgl_result = asyncio.run(mm_processor.process_mm_data_async(
        image_data=[precomputed_image_data],
        audio_data=[],
        request_obj=request_obj,
        input_text=processor_output["input_ids"][0].detach().cpu().tolist(),
        max_req_input_len=exp_args.max_req_input_len
    ))
    print(f"预计算特征处理时间: {time.time() + async_begin}")
    import ipdb; ipdb.set_trace()   
    print(f"处理结果: {type(sgl_result)}")
    if hasattr(sgl_result, 'input_ids'):
        print(f"输入 ID 长度: {len(sgl_result.input_ids)}")
    
    return sgl_result


if __name__ == "__main__":
    logging.info("start")
    
    # 测试不同的 processor
    print("="*50)
    print("测试 Processor 性能")
    print("="*50)
    
    # 测试 QwenVL 预计算特征处理
    test_qwenvl_prec_processor(exp_qwen2vl)
    print("\n" + "="*50 + "\n")
    
    test_sgl_processor(exp_qwen2vl)
    test_sgl_processor(exp_cpmv)
    print("\n" + "="*50 + "\n")

    






