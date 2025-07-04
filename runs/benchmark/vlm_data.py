from pathlib import Path
from typing import Any, Dict, List
from PIL import Image
import base64
import io

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

    def to_prompt(self, item):
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