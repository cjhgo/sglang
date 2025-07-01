#!/usr/bin/env python3
"""
简化的HF Processor调试脚本
用于快速调试processor部分，不加载模型
"""
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Union, Optional, Dict, Any
import sys

# 获取当前文件的绝对路径
abs_base = Path(__file__).resolve().parent

@dataclass
class PromptConfig:
    user_content: Union[str, List[Dict[str, Any]]] = field(default_factory=lambda: "")
    system_prompt: str = "You are a helpful assistant.请用中文回答问题"
    question: str = "What is in the image?"
    image_path: str = f"{abs_base}/../2.jpg"
    
    def get_prompt(self):
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_content}
        ]

def load_processor_only(model_id: str):
    """只加载tokenizer和processor，不加载模型"""
    print(f"Loading tokenizer and processor from: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return tokenizer, processor

def debug_processor_inputs(tokenizer, processor, prompt_config: PromptConfig, pop_keys: Optional[List[str]] = None):
    """调试processor的输入输出"""
    if pop_keys is None:
        pop_keys = []
    
    print("=" * 50)
    print("DEBUG: 处理输入数据")
    print("=" * 50)
    
    # 1. 加载图片
    print(f"Loading image from: {prompt_config.image_path}")
    image = Image.open(prompt_config.image_path).convert('RGB')
    print(f"Image size: {image.size}")
    
    # 2. 应用聊天模板
    print("\nApplying chat template...")
    raw_prompt = tokenizer.apply_chat_template(
        prompt_config.get_prompt(), 
        tokenize=False, 
        add_generation_prompt=True
    )
    print("Raw prompt:")
    print(raw_prompt)
    print()
    
    # 3. Processor处理
    print("Processing with processor...")
    import time
    begin = -time.time()
    model_inputs = processor(
        text=[raw_prompt, raw_prompt], 
        images=[[image], [image]], 
        videos=None, 
        return_tensors="pt"
    )
    print(f"Processor time: {time.time() + begin}")
    import ipdb; ipdb.set_trace()
    
    # 4. 显示processor输出的键
    print(f"Processor output keys: {list(model_inputs.keys())}")
    
    # 5. 显示每个键的shape
    for key, value in model_inputs.items():
        if hasattr(value, 'shape'):
            pass
            # print(f"  {key}: {value.shape}")
        else:
            pass
            # print(f"  {key}: {type(value)} - {value}")
    
    # 6. 移除指定的键
    if pop_keys:
        print(f"\nRemoving keys: {pop_keys}")
        for key in pop_keys:
            removed = model_inputs.pop(key, None)
            if removed is not None:
                print(f"  Removed {key}")
            else:
                print(f"  Key {key} not found")
    
    print(f"Final keys: {list(model_inputs.keys())}")
    return model_inputs

def main():
    if len(sys.argv) < 2:
        print("Usage: python tiny_hf.py <model_choice>")
        print("  0: Local checkpoint model")
        print("  1: Qwen2-VL-2B-Instruct")
        sys.exit(1)
    
    choose = int(sys.argv[1])
    
    # 模型配置
    model_id_0 = f"{abs_base}/../../../../modelbest/sync/checkpoint-1170"
    model_id_1 = "Qwen/Qwen2-VL-2B-Instruct"
    model_ids = [model_id_0, model_id_1]
    
    # pop_key配置
    pop_keys_configs = [
        ["image_sizes"],  # 对应model_id_0
        []                # 对应model_id_1
    ]
    
    # prompt配置
    prompts = [
        PromptConfig(
            user_content=f"<Question>What is in the image?</Question>\n：如图\n(<image>./</image>)"
        ),
        PromptConfig(
            user_content=[{"type":"image"}, {"type": "text", "text": "What is in the image?"}]
        )
    ]
    
    model_id = model_ids[choose]
    pop_keys = pop_keys_configs[choose]
    prompt_config = prompts[choose]
    
    print(f"选择的模型: {model_id}")
    print(f"Prompt配置: {choose}")
    
    try:
        # 只加载processor，不加载模型
        tokenizer, processor = load_processor_only(model_id)
        
        # 调试processor处理过程
        model_inputs = debug_processor_inputs(tokenizer, processor, prompt_config, pop_keys)
        
        print("\n" + "=" * 50)
        print("调试完成！")
        print("=" * 50)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 