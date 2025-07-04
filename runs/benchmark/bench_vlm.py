import json
import logging
import unittest
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
import datetime
import os

import torch
from transformers import AutoTokenizer, AutoProcessor
from dataclasses import dataclass

from sglang import Engine
from sglang.srt.conversation import generate_chat_conv
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.server_args import ServerArgs
from vlm_data import VQADataset

# 设置日志级别
logging.basicConfig(level=logging.INFO)

@dataclass
class BatchInput:
    prompt: list[Any]
    image_data: list[Any]
    answer: list[str]
    prompt_type: str = "text"

    def __len__(self):
        return len(self.prompt)
    
    @classmethod
    def from_vlm_dataset(cls, vlm_dataset:VQADataset, batch_size:int):
        batch = vlm_dataset.get_batch(batch_size=batch_size)
        return cls(
            prompt=[item["prompt"] for item in batch],
            prompt_type="text",
            image_data=[[item["image"]] for item in batch],#[[]] for batch
            answer=[item["answer"] for item in batch]
        )


class VLMTestBase(ABC):
    """VLM测试基类"""
    
    model_path = None
    chat_template = None
    processor:AutoProcessor = None
    tokenizer:AutoTokenizer = None
    vlm_dataset:VQADataset = None
    batch_size = [5, 1,2,4,8]

    @classmethod
    def setUpClass(cls):
        assert cls.model_path is not None, "Set model_path in subclass"
        assert cls.chat_template is not None, "Set chat_template in subclass"
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.processor = AutoProcessor.from_pretrained(
            cls.model_path, trust_remote_code=True, use_fast=True
        )
        cls.tokenizer = AutoTokenizer.from_pretrained(
            cls.model_path, trust_remote_code=True, use_fast=True
        )
        cls.vlm_dataset = VQADataset(chat_template=cls.chat_template)

    def setUp(self):
        """每个测试方法前的初始化"""
        # 创建Engine实例
        self.engine = Engine(
            model_path=self.model_path,
            chat_template=self.chat_template,
            device=self.device.type,
            trust_remote_code=True,
            enable_multimodal=True,#necessary for gemma-3-4b-it
            mem_fraction_static=0.6,
            mm_attention_backend="fa3",
            max_prefill_tokens=int(16384*5),
        )

    def tearDown(self):
        """每个测试方法后的清理"""
        if hasattr(self, 'engine'):
            self.engine.shutdown()

    def verify_response(self, batch_input: BatchInput, output):
        """验证响应结果"""
        print(f"Output{'='*100}, {output}")
        for idx, item in enumerate(output):
            logging.error(f"idx: {idx}, answer: {batch_input.answer[idx]}\n output: {item}")

    """
    parallel batch processor not fit prec
    def _get_processor_batchout(self, batch_input: BatchInput):
        # Process inputs using processor
        inputs = self.processor( text=batch_input.prompt, images=batch_input.image_data,
            return_tensors="pt",
        )
        return inputs
    def _processor_out2list(self, processor_output):
        input_ids_list = processor_output.input_ids.tolist()
        image_data_list = processor_output.pixel_values.tolist()
        return input_ids_list, image_data_list
    """

    def _get_processor_batchout(self, batch_input: BatchInput):
        processor_output = []
        for idx in range(len(batch_input)):
            prompt = batch_input.prompt[idx]
            image_data = batch_input.image_data[idx]
            inputs = self.processor( text=prompt, images=image_data,
                return_tensors="pt",
            )
            processor_output.append(inputs)
        return processor_output

    def _processor_out2list(self, processor_output):
        input_ids_list = []
        image_data_list = []
        for item in processor_output:
            input_ids_list.append(item["input_ids"][0].tolist())
            image_data_list.append(self._image_data_from_processor_output(item))
        return input_ids_list, image_data_list
    
    def _image_data_from_processor_output(self, processor_output):
        raise NotImplementedError

    def get_batch_ids_input(self, batch_input: BatchInput):
        processor_output = self._get_processor_batchout(batch_input)
        input_ids_list, image_data_list = self._processor_out2list(processor_output)
        batch_input = BatchInput(
            prompt=input_ids_list,
            image_data=image_data_list,
            prompt_type="id",
            answer=batch_input.answer,
        )
        return batch_input
    
    def _call_engine(self, batch_input: BatchInput):
        if batch_input.prompt_type == "text":
            return self.engine.generate(
                prompt=batch_input.prompt,
                image_data=batch_input.image_data,
                sampling_params=dict(temperature=0.0),
            )
        elif batch_input.prompt_type == "id":
            return self.engine.generate(
                input_ids=batch_input.prompt,
                image_data=batch_input.image_data,
                sampling_params=dict(temperature=0.0),
            )
        else:
            raise ValueError(f"Invalid prompt type: {batch_input.prompt_type}")


    @unittest.skip("skip")
    def test_batch_vqa_verify(self):
        batch_input = BatchInput.from_vlm_dataset(self.vlm_dataset, 5)
        import time
        begin_time = -time.time()
        out = self._call_engine(batch_input)
        duration = time.time() + begin_time
        logging.error(f"duration: {duration}")
        self.verify_response(batch_input, out)
    
    @unittest.skip("skip")
    def test_batch_pixel_verify(self):
        batch_input = BatchInput.from_vlm_dataset(self.vlm_dataset, 5)
        batch_input_ids = self.get_batch_ids_input(batch_input)
        out = self._call_engine(batch_input_ids)
        self.verify_response(batch_input, out)
    
    # @unittest.skip("skip")
    def test_batch_bench(self):
        e2e_time = []
        for batch_size in self.batch_size:
            torch.cuda.empty_cache()
            batch_input = BatchInput.from_vlm_dataset(self.vlm_dataset, batch_size)
            out = self._call_engine(batch_input)
            cur_e2e = out[-1]['meta_info']['e2e_latency']
            e2e_time.append(cur_e2e)
            logging.error(f"batch_size: {batch_size}, e2e_time: {cur_e2e}")
        for batch_size, e2e_time in zip(self.batch_size, e2e_time):
            logging.error(f"batch_size: {batch_size}, e2e_time: {e2e_time}")

        e2e_time = []
        for batch_size in self.batch_size:
            torch.cuda.empty_cache()
            batch_input = BatchInput.from_vlm_dataset(self.vlm_dataset, batch_size)
            batch_input_ids = self.get_batch_ids_input(batch_input)
            out = self._call_engine(batch_input_ids)
            cur_e2e = out[-1]['meta_info']['e2e_latency']
            e2e_time.append(cur_e2e)
            logging.error(f"batch_size: {batch_size}, e2e_time: {cur_e2e}")

        for batch_size, e2e_time in zip(self.batch_size, e2e_time):
            logging.error(f"batch_size: {batch_size}, e2e_time: {e2e_time}")


# @unittest.skip("skip")
class TestGemmaVLM(VLMTestBase, unittest.TestCase):

    model_path = "google/gemma-3-4b-it"
    chat_template = "gemma-it"
    batch_size = [20, 2, 4,  80, 120, 140, 160, 180, 200, 220, 240]

    def _image_data_from_processor_output(self, processor_output):
        pixel_values = processor_output["pixel_values"][0]
        return dict(modality="IMAGE", pixel_values=pixel_values)


@unittest.skip("skip")
class TestJanusProVLM(VLMTestBase, unittest.TestCase):
    """Janus Pro VLM 简化测试类，只测试基本的QA功能"""
    model_path = "deepseek-ai/Janus-Pro-1B"
    chat_template = "janus-pro"


# @unittest.skip("skip")
class TestMiniCPMV(VLMTestBase, unittest.TestCase):
    model_path = "openbmb/MiniCPM-v-2_6"
    chat_template = "minicpmv"

    def _image_data_from_processor_output(self, processor_output):
        pixel_values = processor_output["pixel_values"][0]
        tgt_sizes = processor_output["tgt_sizes"][0]
        return dict(modality="IMAGE", pixel_values=pixel_values, tgt_sizes=tgt_sizes)
    

@unittest.skip("skip")
class TestMiniCPMO(VLMTestBase, unittest.TestCase):
    model_path = "openbmb/MiniCPM-o-2_6"
    chat_template = "minicpmo"
    


if __name__ == "__main__":
    unittest.main()