import unittest
import logging
import time
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from vlm_data import VQADataset
logging.basicConfig(level=logging.INFO)

abs_base = Path(__file__).resolve().parent


from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from transformers import AutoProcessor, AutoTokenizer

class ProcessorTestBase(ABC):
    """Processor 测试基类"""
    
    model_path = None
    chat_template = None
    trust_remote_code = True
    batch_size = 10
    max_req_input_len = 1024
    
    vlm_dataset: VQADataset = None
    mm_processor: BaseMultimodalProcessor = None
    hf_processor: AutoProcessor = None
    tokenizer: AutoTokenizer = None

    @classmethod
    def setUpClass(cls):
        """类级别的初始化"""
        assert cls.model_path is not None, "Set model_path in subclass"
        assert cls.chat_template is not None, "Set chat_template in subclass"
        cls.vlm_dataset = VQADataset(chat_template=cls.chat_template)
        # 创建 processor 组件
        cls._create_processor_components()

    @classmethod
    def _create_processor_components(cls):
        from sglang.srt.hf_transformers_utils import get_processor, get_tokenizer
        from sglang.srt.managers.multimodal_processor import ( get_mm_processor, import_processors)
        from sglang.srt.configs.model_config import ModelConfig
        from sglang.srt.server_args import ServerArgs
            
        logging.info(f"加载 SGLang processor: {cls.model_path}")
            
        cls.hf_processor = get_processor( cls.model_path, trust_remote_code=True,)
        cls.tokenizer = get_tokenizer( cls.model_path, trust_remote_code=True,)
            
        cls.server_args = ServerArgs( model_path=cls.model_path, trust_remote_code=cls.trust_remote_code,)
        cls.model_config = ModelConfig.from_server_args(cls.server_args)
            
        import_processors()
        cls.mm_processor = get_mm_processor( cls.model_config.hf_config, cls.server_args, cls.hf_processor)
            
        logging.info(f"成功创建 mm_processor: {type(cls.mm_processor)}")

    def setUp(self):
        self.batch = self.vlm_dataset.get_batch(batch_size=self.batch_size)
        self.input_text = [item["prompt"] for item in self.batch]
        self.image_data = [[item["image"]] for item in self.batch]
        self.one_req = (self.input_text[0], self.image_data[0])

    def _get_multimodal_tokens(self):
        from sglang.srt.multimodal.processors.base_processor import MultimodalSpecialTokens
        
        return MultimodalSpecialTokens(
            image_token=getattr(self.mm_processor, 'IMAGE_TOKEN', '<image>'),
            image_token_regex=getattr(self.mm_processor, 'IMAGE_TOKEN_REGEX', None),
        )

    @unittest.skip("skip")
    def test_load_mm_data(self):
        multimodal_tokens = self._get_multimodal_tokens()
        
        mm_result = self.mm_processor.load_mm_data(
            self.one_req[0], 
            audio_data=[], 
            image_data=self.one_req[1], 
            max_req_input_len=self.max_req_input_len, 
            multimodal_tokens=multimodal_tokens
        )
        return mm_result
    
    def _call_async_process(self, text, image):
        request_obj = lambda: None
        request_obj.audio_data = []
        async_begin = -time.time()
        result = asyncio.run(self.mm_processor.process_mm_data_async(
            image_data=image,
            audio_data=[],
            request_obj=request_obj,
            input_text=text,
            max_req_input_len=self.max_req_input_len
        ))
        duration = time.time() + async_begin
        return result, duration

    def test_process_mm_data_async(self):
        text, image = self.one_req
        result, duration = self._call_async_process(text, image)
        print(f"process_mm_data_async 测试通过: {type(result)}, 耗时: {duration:.4f}s")
        return result

    # @unittest.skip("skip")
    def test_prec_process(self):
        input_ids, pixel_values_image_data = self.get_pixel_values()
        result, duration = self._call_async_process(input_ids, pixel_values_image_data)
        print(f"prec_process 测试通过: {type(result)}, 耗时: {duration:.4f}s")
        return result

    # @unittest.skip("skip")
    def test_hf_processor(self):
        hf_result = self.hf_processor(
            text=self.one_req[0],
            images=self.one_req[1],
            return_tensors="pt",
        )
        return hf_result

    # @unittest.skip("skip")
    def test_all_results(self):
        hf_result = self.test_hf_processor()
        sgl_raw_result = self.test_process_mm_data_async()
        sgl_prec_result = self.test_prec_process()
        
        # 比较两个结果
        self._compare_results(sgl_raw_result, sgl_prec_result)
        
        return hf_result, sgl_raw_result, sgl_prec_result
    
    def _compare_results(self, raw_result, prec_result):
        """比较原始处理和预计算处理的结果"""
        print("\n=== 开始比较 raw_result 和 prec_result ===")
        
        # 1. 比较 input_ids
        if raw_result['input_ids'] != prec_result['input_ids']:
            raise AssertionError(f"input_ids 不匹配:\nraw: {raw_result['input_ids'][:20]}...\nprec: {prec_result['input_ids'][:20]}...")
        print("✓ input_ids 匹配")
        
        # 2. 比较特殊 token IDs
        special_tokens = ['audio_start_id', 'audio_end_id', 'im_token_id', 
                         'im_start_id', 'im_end_id', 'slice_start_id', 'slice_end_id']
        for token in special_tokens:
            if raw_result.get(token) != prec_result.get(token):
                raise AssertionError(f"{token} 不匹配: raw={raw_result.get(token)}, prec={prec_result.get(token)}")
        print("✓ 所有特殊 token IDs 匹配")
        
        # 3. 比较 mm_items
        raw_items = raw_result.get('mm_items', [])
        prec_items = prec_result.get('mm_items', [])
        
        if len(raw_items) != len(prec_items):
            raise AssertionError(f"mm_items 数量不匹配: raw={len(raw_items)}, prec={len(prec_items)}")
        print(f"✓ mm_items 数量匹配: {len(raw_items)}")
        
        # 4. 详细比较每个 mm_item
        for i, (raw_item, prec_item) in enumerate(zip(raw_items, prec_items)):
            print(f"\n比较第 {i+1} 个 mm_item:")
            
            # 比较 modality
            if raw_item.modality != prec_item.modality:
                raise AssertionError(f"第 {i+1} 个 item 的 modality 不匹配")
            print(f"  ✓ modality: {raw_item.modality}")
            
            # 比较 image_offsets
            if raw_item.image_offsets != prec_item.image_offsets:
                raise AssertionError(f"第 {i+1} 个 item 的 image_offsets 不匹配")
            print(f"  ✓ image_offsets 匹配: {len(raw_item.image_offsets)} 个偏移")
            
            # 比较 tgt_size
            if hasattr(raw_item, 'tgt_size') and hasattr(prec_item, 'tgt_size'):
                if len(raw_item.tgt_size) != len(prec_item.tgt_size):
                    raise AssertionError(f"第 {i+1} 个 item 的 tgt_size 数量不匹配: raw={len(raw_item.tgt_size)}, prec={len(prec_item.tgt_size)}")
                
                for j, (raw_ts, prec_ts) in enumerate(zip(raw_item.tgt_size, prec_item.tgt_size)):
                    if not torch.equal(raw_ts, prec_ts):
                        raise AssertionError(f"第 {i+1} 个 item 的第 {j+1} 个 tgt_size 不匹配")
                print(f"  ✓ tgt_size 匹配: {len(raw_item.tgt_size)} 个尺寸")
            # 比较 pixel_values
            if raw_item.pixel_values is not None and prec_item.pixel_values is not None:
                if len(raw_item.pixel_values) != len(prec_item.pixel_values):
                    raise AssertionError(f"第 {i+1} 个 item 的 pixel_values 数量不匹配")
                
                # 比较每个 tensor
                for j, (raw_pv, prec_pv) in enumerate(zip(raw_item.pixel_values, prec_item.pixel_values)):
                    if not torch.allclose(raw_pv, prec_pv, rtol=1e-5, atol=1e-5):
                        raise AssertionError(f"第 {i+1} 个 item 的第 {j+1} 个 pixel_value 不匹配")
                print(f"  ✓ pixel_values 匹配: {len(raw_item.pixel_values)} 个张量")
            
        
        print("\n=== 所有比较通过！✓ ===\n")

    def get_pixel_values(self):
        text, image = self.one_req# image is [image]
        hf_out = self.hf_processor(
            text=[text],
            images=image,
            return_tensors="pt",
        )
        input_ids = hf_out["input_ids"][0].tolist()
        
        # 构建预计算数据，包含 MiniCPM 所需的所有字段
        pixel_values_image_data = dict(
            modality="IMAGE",
            pixel_values=hf_out["pixel_values"][0],
        )
        return input_ids, [pixel_values_image_data]


@unittest.skip("skip")
class TestGemmaProcessor(ProcessorTestBase, unittest.TestCase):
    
    model_path = "google/gemma-3-4b-it"
    chat_template = "gemma-it"
    

# @unittest.skip("skip")
class TestMiniCPMVProcessor(ProcessorTestBase, unittest.TestCase):
    """测试 MiniCPM-V processor"""
    
    model_path = f"{abs_base}/../../../../modelbest/sync/checkpoint-1170"
    chat_template = "minicpmv"
    
    def _get_multimodal_tokens(self):
        """获取 MiniCPM-V 模型的多模态特殊token"""
        from sglang.srt.multimodal.processors.base_processor import MultimodalSpecialTokens
        
        return MultimodalSpecialTokens(
            image_token=self.mm_processor.image_token, 
            image_token_regex=self.mm_processor.image_token_regex,
            audio_token=self.mm_processor.audio_token
        )

    def get_pixel_values(self):
        text, image = self.one_req# image is [image]
        hf_out = self.hf_processor(
            text=[text],
            images=image,
            return_tensors="pt",
        )
        input_ids = hf_out["input_ids"][0].tolist()
        
        # 构建预计算数据，包含 MiniCPM 所需的所有字段
        pixel_values_image_data = dict(
            modality="IMAGE",
            pixel_values=hf_out["pixel_values"],#[0],
            tgt_size=hf_out["tgt_sizes"], #[0],#name for sglang map
            # tgt_sizes=hf_out["tgt_sizes"],#[0],#name from hf processor
        )
        return input_ids, [pixel_values_image_data]

@unittest.skip("skip")
class TestQwen2VLProcessor(ProcessorTestBase, unittest.TestCase):
    """测试 Qwen2-VL processor"""
    
    model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
    chat_template = "qwen2-vl"
    
    def _get_multimodal_tokens(self):
        """获取 Qwen2-VL 模型的多模态特殊token"""
        from sglang.srt.multimodal.processors.base_processor import MultimodalSpecialTokens
        
        return MultimodalSpecialTokens(
            image_token=self.mm_processor.IMAGE_TOKEN,
            image_token_regex=self.mm_processor.IMAGE_TOKEN_REGEX,
        )




@unittest.skip("skip")
class TestJanusProcessor(ProcessorTestBase, unittest.TestCase):
    """测试 Janus processor"""
    
    model_path = "deepseek-ai/Janus-Pro-1B"
    chat_template = "janus-pro"
    
    def _get_multimodal_tokens(self):
        """获取 Janus 模型的多模态特殊token"""
        from sglang.srt.multimodal.processors.base_processor import MultimodalSpecialTokens
        
        return MultimodalSpecialTokens(
            image_token=self.mm_processor.IMAGE_TOKEN,
            image_token_regex=self.mm_processor.IMAGE_TOKEN_REGEX,
        )


if __name__ == "__main__":
    # 设置测试选项
    unittest.main(verbosity=2)
