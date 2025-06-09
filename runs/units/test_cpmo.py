import logging
logging.basicConfig(level=logging.INFO)

prompts = [
        "Find the perimeter of the figure. Round to the nearest tenth if necessary. You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}.",
        ")In the figure, $ \overline{JM} \cong \overline{PM}$ and $ \overline{ML} \cong \overline{PL}$. If $m \angle PLJ=34$, find $m \angle JPM$. You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}.",
        "The segment is tangent to the circle. Find $x$. You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}.",
        "Find the radius of $\odot K$. Round to the nearest hundredth. You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}."
    ]


def gen_raw_prompt_id(tokenizer, user_prompt):
    #apply_chat_template by hardcode, or by tokenizer.apply_chat_template, which is necessray!
    # prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n(<image>./</image>)\n{user_prompt}<|im_end|>\n<|im_start|>"
    prompt = f"<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\n(<image>./</image>)\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
    return tokenizer.encode(prompt, return_tensors="pt")[0].tolist()

def inputid_version():
    """使用 input_ids 版本的示例"""
    from sglang.srt.server_args import ServerArgs
    import sglang as sgl
    from transformers import AutoTokenizer
    server_args = ServerArgs(
        model_path="openbmb/MiniCPM-o-2_6",
        chat_template="minicpmo",
        trust_remote_code=True,
    )
    vlm = sgl.Engine(server_args=server_args)

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained("openbmb/MiniCPM-o-2_6", trust_remote_code=True)
    
    vlm.release_memory_occupation()
    from PIL import Image
    img2 = Image.open("../2.jpg").convert("RGB")
    img3 = Image.open("../3.png").convert("RGB")
    img4 = Image.open("../4.png").convert("RGB")

    input_ids = []
    images = []
    for prompt in prompts:
        raw_pid = gen_raw_prompt_id(tokenizer, prompt)
        input_ids.append(raw_pid)
        images.append(img2)

    # raw_pid2 = gen_raw_prompt_id(tokenizer, "what is in image 或者说图片里有什么")
    # raw_pid3 = gen_raw_prompt_id(tokenizer, "图片里有几个三角形")
    # raw_pid4 = gen_raw_prompt_id(tokenizer, "图片里有什么几何形状")
    # input_ids = [raw_pid2, raw_pid3, raw_pid4]
    # images = [img2, img3, img4]

    print("=== Input ID Version ===")
    print("Input IDs:", input_ids)
    # import ipdb; ipdb.set_trace()

    sampling_params = {
        'n': 2, 'max_new_tokens': 2048, 
        'presence_penalty': 0.0, 'frequency_penalty': 0.0, 
        'repetition_penalty': 1.0, 'temperature': 1.0, 
        'top_k': -1, 'top_p': 1, 'ignore_eos': False
    }
    output = vlm.generate(input_ids=input_ids, image_data=images, sampling_params=sampling_params)
    # print("Output:", output)
    print("===================\n")
    for out in output:
        print(out, end="\n\n")
    # import ipdb; ipdb.set_trace()
    vlm.shutdown()

if __name__ == "__main__":
    # 演示两种方式
    inputid_version()