from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
import logging
import torch
logging.basicConfig(level=logging.INFO)

@torch.no_grad
def extend(reqs, model_runner):
    batch = ScheduleBatch.init_new(
        reqs=reqs,
        req_to_token_pool=model_runner.req_to_token_pool,
        token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
        tree_cache=None,
        model_config=model_runner.model_config,
        enable_overlap=False,
        spec_algorithm=SpeculativeAlgorithm.NONE,
        enable_custom_logit_processor=False,
    )
    batch.prepare_for_extend()
    # _maybe_prepare_dp_attn_batch(batch, model_runner)
    model_worker_batch = batch.get_model_worker_batch()
    forward_batch = ForwardBatch.init_new(model_worker_batch, model_runner)
    logits_output, _ = model_runner.forward(forward_batch)
    next_token_ids = model_runner.sample(logits_output, forward_batch)
    return next_token_ids, logits_output.next_token_logits, batch

@torch.no_grad
def decode(input_token_ids, batch, model_runner):
    batch.output_ids = input_token_ids
    batch.prepare_for_decode()
    # _maybe_prepare_dp_attn_batch(batch, model_runner)
    model_worker_batch = batch.get_model_worker_batch()
    forward_batch = ForwardBatch.init_new(model_worker_batch, model_runner)
    logits_output, _ = model_runner.forward(forward_batch)
    next_token_ids = model_runner.sample(logits_output, forward_batch)
    return next_token_ids, logits_output.next_token_logits

def prepare_inputs_for_correctness_test(cut_len, tokenizer):
    prompts = [
        "The capital of France is",
        "The capital of the United Kindom is",
        "Today is a sunny day and I like",
    ]
    input_ids = [tokenizer.encode(p) for p in prompts]
    sampling_params = SamplingParams(
        temperature=0,
        max_new_tokens=16,
    )

    reqs = []
    for i in range(len(prompts)):
        assert len(input_ids[i]) > cut_len

        tmp_input_ids = input_ids[i][: cut_len]
        req = Req(
            rid=i,
            origin_input_text=prompts[i],
            origin_input_ids=tmp_input_ids,
            sampling_params=sampling_params,
        )
        req.prefix_indices = []
        req.fill_ids = req.origin_input_ids
        req.extend_input_len = len(req.fill_ids) - len(req.prefix_indices)
        req.logprob_start_len = len(req.origin_input_ids) - 1
        reqs.append(req)

    return input_ids, reqs

def prepare_extend_inputs_for_correctness_test(
    cut_len, input_ids, reqs, model_runner
):
    for i in range(len(reqs)):
        req = reqs[i]
        req.fill_ids += input_ids[i][cut_len :]
        req.prefix_indices = model_runner.req_to_token_pool.req_to_token[
            i, : cut_len
        ]
        req.extend_input_len = len(req.fill_ids) - len(req.prefix_indices)
        req.logprob_start_len = len(req.origin_input_ids) - 1
    return reqs

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
    tokenizer = get_tokenizer(
        server_args.tokenizer_path,
        tokenizer_mode=server_args.tokenizer_mode,
        trust_remote_code=server_args.trust_remote_code,
    )
    return model_runner, tokenizer


def main():
    server_args = ServerArgs(
        # model_path="Qwen/Qwen2.5-0.5B-Instruct",
        model_path="deepseek-ai/Janus-Pro-1B",
        trust_remote_code=True,
    )
    port_args = PortArgs.init_new(server_args)

    print("开始加载模型...")
    model, tokenizer = load_model(server_args, port_args, tp_rank=0)
    input_ids, reqs = prepare_inputs_for_correctness_test(cut_len=4, tokenizer=tokenizer)
    next_token_ids, next_token_logits, batch = extend(reqs, model)
    reqs = prepare_extend_inputs_for_correctness_test(cut_len=4, input_ids=input_ids, reqs=reqs, model_runner=model)
    next_token_ids, logits_output, batch = extend(reqs, model)
    output_ids = [input_ids[i] + [next_token_ids[i]] for i in range(len(input_ids))]
    for _ in range(16):
        next_token_ids, _ = decode(next_token_ids, batch, model)
        next_token_ids_list = next_token_ids.tolist()
        for i in range(len(reqs)):
            output_ids[i].append(next_token_ids_list[i])
    #print output text
    for i in range(len(reqs)):
        print(f"========== Prompt {i} ==========")
        print(tokenizer.decode(output_ids[i]), "\n")

if __name__ == "__main__":
    main()
    # import ipdb; ipdb.set_trace()
