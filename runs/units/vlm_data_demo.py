import gradio as gr
from datasets import load_dataset
from PIL import Image
import base64
import io

def base64_to_pil(base64_str):
    image_data = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(image_data))
    return image

# 加载数据集
#https://huggingface.co/datasets/OpenStellarTeam/Chinese-SimpleVQA
dataset = load_dataset("OpenStellarTeam/Chinese-SimpleVQA")

def show_sample(idx):
    item = dataset["train"][idx]
    image = base64_to_pil(item["image_base64"])
    # image = image.resize((512, 512))
    
    # 创建三列布局的内容
    col1 = f"主题: {item['Topic']}"
    col2 = f"识别问题: {item['recognition_question']}\n识别答案: {item['recognition_answer']}"
    col3 = f"最终问题: {item['final_question']}\n最终答案: {item['final_answer']}"
    
    return image, col1, col2, col3

# 创建 Gradio 界面
with gr.Blocks(title="Chinese-SimpleVQA 数据集展示") as demo:
    gr.Markdown("# Chinese-SimpleVQA 数据集展示")
    gr.Markdown("这是一个中文视觉问答数据集，包含图像识别和问答任务。")
    
    with gr.Row():
        index = gr.Number(label="样本索引", value=0, minimum=0, maximum=len(dataset["train"])-1)
        submit_btn = gr.Button("显示样本")
    
    with gr.Row():
        image_output = gr.Image(label="图像")
    
    with gr.Row():
        col1_output = gr.Textbox(label="主题信息")
        col2_output = gr.Textbox(label="识别问答")
        col3_output = gr.Textbox(label="最终问答")
    
    submit_btn.click(
        fn=show_sample,
        inputs=[index],
        outputs=[image_output, col1_output, col2_output, col3_output]
    )

if __name__ == "__main__":
    demo.launch() 