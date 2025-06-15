# test_model_loader.py
from transformers import PretrainedConfig
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.configs.load_config import LoadConfig, LoadFormat
from sglang.srt.model_loader.utils import get_model_architecture
from sglang.srt.model_loader.loader import get_model_loader
from sglang.srt.models.minicpmv import MiniCPMV, MiniCPMV2_6
from sglang.srt.server_args import ServerArgs

def main():
    # 1. 创建 ServerArgs
    server_args = ServerArgs(
        model_path="openbmb/MiniCPM-v-2_6",
        trust_remote_code=True,
        chat_template="minicpmv",
        # json_model_override_args='{"version": 2.6}'
    )
    
    # 2. 创建 ModelConfig
    model_config = ModelConfig.from_server_args(server_args)
    
    # 3. 创建 LoadConfig
    load_config = LoadConfig(
        load_format=LoadFormat.AUTO,
        download_dir="./test_download",
        ignore_patterns=None
    )

    # 4. 测试模型架构解析
    print("Step 1: Testing model architecture resolution")
    model_class, arch_name = get_model_architecture(model_config)
    print(f"Model class: {model_class.__name__}")
    print(f"Architecture name: {arch_name}")
    import ipdb; ipdb.set_trace()
    
    """
    # 5. 测试模型实例化
    print("\nStep 2: Testing model instantiation")
    model = model_class(config=model_config.hf_config, quant_config=None)
    print(f"Model type: {type(model).__name__}")
    print(f"Internal model type: {type(model.minicpmv).__name__}")
    
    # 6. 测试模型加载器
    print("\nStep 3: Testing model loader")
    loader = get_model_loader(load_config)
    print(f"Loader type: {type(loader).__name__}")
    
    # 7. 测试版本分发
    print("\nStep 4: Testing version dispatch")
    versions = [2.6, 2.5, 2.0]
    for version in versions:
        print(f"\nTrying version {version}")
        server_args.model_path = f"openbmb/MiniCPM-v-{version}"
        server_args.json_model_override_args = f'{{"version": {version}}}'
        try:
            model_config = ModelConfig.from_server_args(server_args)
            model = model_class(config=model_config.hf_config, quant_config=None)
            print(f"Successfully created model for version {version}")
            print(f"Internal model type: {type(model.minicpmv).__name__}")
        except Exception as e:
            print(f"Failed for version {version}: {str(e)}")

    """
if __name__ == "__main__":
    main()