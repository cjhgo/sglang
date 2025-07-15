import json
import logging
import unittest
from abc import ABC, abstractmethod
from io import BytesIO
from typing import Optional, List

import requests
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor

from sglang import Engine
from sglang.srt.conversation import generate_chat_conv
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.server_args import ServerArgs

# 设置日志级别
logging.basicConfig(level=logging.INFO)

# 测试图片URL
TEST_IMAGE_URL = "https://github.com/sgl-project/sglang/blob/main/test/lang/example_image.png?raw=true"

from pathlib import Path
abs_base = Path(__file__).resolve().parent

class VLMTestBase(ABC):
    """VLM测试基类"""
    
    model_path = None
    chat_template = None
    processor:AutoProcessor = None
    tokenizer:AutoTokenizer = None

    @classmethod
    def setUpClass(cls):
        assert cls.model_path is not None, "Set model_path in subclass"
        assert cls.chat_template is not None, "Set chat_template in subclass"
        cls.image_url = TEST_IMAGE_URL
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        response = requests.get(cls.image_url)
        cls.main_image = Image.open(BytesIO(response.content))
        cls.main_image2 = cls.main_image.rotate(90).crop((0,0, 300, 300))
        cls.processor = AutoProcessor.from_pretrained(
            cls.model_path, trust_remote_code=True, use_fast=True
        )
        cls.tokenizer = AutoTokenizer.from_pretrained(
            cls.model_path, trust_remote_code=True, use_fast=True
        )

    def setUp(self):
        """每个测试方法前的初始化"""
        # 创建Engine实例
        self.engine = Engine(
            model_path=self.model_path,
            chat_template=self.chat_template,
            device=self.device.type,
            trust_remote_code=True,
            enable_multimodal=True,#necessary for gemma-3-4b-it
            mem_fraction_static=0.7,
        )

    def tearDown(self):
        """每个测试方法后的清理"""
        if hasattr(self, 'engine'):
            self.engine.shutdown()

    def get_completion_request(self) -> ChatCompletionRequest:
        json_structure = {
            "model": self.model_path,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": self.image_url}},
                        {"type": "image_url", "image_url": {"url": self.image_url}},
                        {"type": "text", "text": "这两个图片内容一样吗?"},
                    ],
                }
            ],
        }
        json_str = json.dumps(json_structure)
        return ChatCompletionRequest.model_validate_json(json_str)

    def verify_response(self, output):
        """验证响应结果"""
        print(f"Output{'='*100}:\n {output}")

    def get_processor_output(self, req: Optional[ChatCompletionRequest] = None):
        if req is None:
            req = self.get_completion_request()
        conv = generate_chat_conv(req, template_name=self.chat_template)
        text = conv.get_prompt()

        # Process inputs using processor
        inputs = self.processor( text=[text], images=[[self.main_image, self.main_image2]], return_tensors="pt",)

        return inputs

    async def test_understands_image(self):
        req = self.get_completion_request()
        conv = generate_chat_conv(req, template_name=self.chat_template)
        text = conv.get_prompt()
        output = await self.engine.async_generate(
            prompt=[text],
            image_data=[[self.main_image, self.main_image2]],
            sampling_params=dict(temperature=0.0),
        )
        self.verify_response(output)

    def _pixel_values_image_data(self, processor_output):
        """Override in subclass to pass the correct set of arguments."""
        raise NotImplementedError

    async def test_understands_pixel_values(self):
        req = self.get_completion_request()
        processor_output = self.get_processor_output(req=req)
        tmp = self._pixel_values_image_data(processor_output)
        output = await self.engine.async_generate(
            input_ids=[processor_output["input_ids"][0].detach().cpu().tolist()],
            image_data=[tmp], #tmp is dict, which collapse mulit image... not [[tmp]]
            sampling_params=dict(temperature=0.0),
        )
        self.verify_response(output)


@unittest.skip("skip")
class TestGemmaVLM(VLMTestBase, unittest.IsolatedAsyncioTestCase):

    model_path = "google/gemma-3-4b-it"
    chat_template = "gemma-it"

    def _pixel_values_image_data(self, processor_output):
        return dict(
            modality="IMAGE",
            pixel_values=processor_output["pixel_values"][0],
        )

@unittest.skip("skip")
class TestJanusProVLM(VLMTestBase, unittest.IsolatedAsyncioTestCase):
    """Janus Pro VLM 简化测试类，只测试基本的QA功能"""
    model_path = "deepseek-ai/Janus-Pro-1B"
    chat_template = "janus-pro"


class TestMiniCPMV(VLMTestBase, unittest.IsolatedAsyncioTestCase):
    # model_path = f"{abs_base}/../../../../modelbest/sync/checkpoint-1170"
    model_path = "openbmb/MiniCPM-v-2_6"
    chat_template = "minicpmv"

    def _pixel_values_image_data(self, processor_output):
        image_slice_patch_size = len(processor_output.pixel_values[0])
        pre_dicts = []
        for i in range(image_slice_patch_size):
            pre_dicts.append(dict(
                modality="IMAGE",
                pixel_values=processor_output.pixel_values[0][i],
                tgt_size=processor_output.tgt_sizes[0][i],
            ))
        return pre_dicts

class TestMiniCPMO(VLMTestBase, unittest.IsolatedAsyncioTestCase):
    model_path = "openbmb/MiniCPM-o-2_6"
    chat_template = "minicpmo"
    def _pixel_values_image_data(self, processor_output):
        return dict(
            modality="IMAGE",
            pixel_values=processor_output["pixel_values"],#[0],
            tgt_size=processor_output["tgt_sizes"],#[0],#name for sglang map
        )
    

if __name__ == "__main__":
    unittest.main()