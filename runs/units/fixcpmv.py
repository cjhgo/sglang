from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.server_args import PortArgs, ServerArgs
import logging
logging.basicConfig(level=logging.INFO)

def load_model(server_args, port_args, tp_rank):
    model_config = ModelConfig.from_server_args(server_args)
    model_runner = ModelRunner(
        model_config=model_config,
        mem_fraction_static=server_args.mem_fraction_static,
        gpu_id=tp_rank,
        tp_rank=tp_rank,
        tp_size=server_args.tp_size,
        pp_rank=0,
        pp_size=1,
        nccl_port=port_args.nccl_port,
        server_args=server_args
    )
    return model_runner

if __name__ == "__main__":
    server_args = ServerArgs(
        model_path="openbmb/MiniCPM-v-2_6",
        trust_remote_code=True,
        max_running_requests=33,
    )
    port_args = PortArgs.init_new(server_args)
    import ipdb; ipdb.set_trace()

    print("开始加载模型...")
    model_runner = load_model(server_args, port_args, tp_rank=0)
    import ipdb; ipdb.set_trace()
    logging.info(f"after loading,model_runner.model.load_weights={hasattr(model_runner.model, 'load_weights')}")
    assert hasattr(model_runner.model, "load_weights"), "model_runner.model.load_weights got lost"
    print("模型加载完成！")
