import logging
logging.basicConfig(level=logging.INFO)

def prompt_version():
    """使用 prompt 字符串版本的示例"""
    from sglang.srt.server_args import ServerArgs
    import sglang as sgl
    server_args = ServerArgs(
        model_path="deepseek-ai/Janus-Pro-1B",
        chat_template="janus-pro",
        trust_remote_code=True,
    )
    vlm = sgl.Engine(server_args=server_args)

    vlm.release_memory_occupation()
    from PIL import Image
    img = Image.open("../2.jpg").convert("RGB")
    from sglang.srt.conversation import chat_templates
    conv = chat_templates["janus-pro"].copy()
    image_token = chat_templates["janus-pro"].image_token
    user_prompt = f"{image_token}what is in image 或者说图片里有什么"
    conv.append_message(conv.roles[0], user_prompt)
    prompt = conv.get_prompt()
    print("=== Prompt Version ===")
    print("Input prompt:", prompt)

    sampling_params = {
        'n': 2, 'max_new_tokens': 2048, 
        'presence_penalty': 0.0, 'frequency_penalty': 0.0, 
        'repetition_penalty': 1.0, 'temperature': 1.0, 
        'top_k': -1, 'top_p': 1, 'ignore_eos': False
    }
    output = vlm.generate(prompt, image_data=img, sampling_params=sampling_params)
    print("Output:", output)
    print("===================\n")
    vlm.shutdown()

def inputid_version():
    """使用 input_ids 版本的示例"""
    from sglang.srt.server_args import ServerArgs
    import sglang as sgl
    from transformers import AutoTokenizer
    server_args = ServerArgs(
        model_path="deepseek-ai/Janus-Pro-1B",
        chat_template="janus-pro",
        trust_remote_code=True,
    )
    vlm = sgl.Engine(server_args=server_args)

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/Janus-Pro-1B", trust_remote_code=True)
    
    vlm.release_memory_occupation()
    from PIL import Image
    img = Image.open("../2.jpg").convert("RGB")
    from sglang.srt.conversation import chat_templates
    conv = chat_templates["janus-pro"].copy()
    image_token = chat_templates["janus-pro"].image_token
    user_prompt = f"{image_token}what is in image 或者说图片里有什么"
    conv.append_message(conv.roles[0], user_prompt)
    prompt = conv.get_prompt()
    
    # 使用 tokenizer 转换为 input_ids
    input_ids = tokenizer.encode(prompt, return_tensors="pt")[0].tolist()
    print("=== Input ID Version ===")
    print("Input prompt:", prompt)
    print("Input IDs:", input_ids)

    sampling_params = {
        'n': 2, 'max_new_tokens': 2048, 
        'presence_penalty': 0.0, 'frequency_penalty': 0.0, 
        'repetition_penalty': 1.0, 'temperature': 1.0, 
        'top_k': -1, 'top_p': 1, 'ignore_eos': False
    }
    output = vlm.generate(input_ids=input_ids, image_data=img, sampling_params=sampling_params)
    print("Output:", output)
    print("===================\n")
    vlm.shutdown()

if __name__ == "__main__":
    # 演示两种方式
    # inputid_version()
    prompt_version()
