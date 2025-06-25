"""
VLM Benchmark - Extension of bench_one_batch.py for Vision-Language Models

This script extends the original bench_one_batch.py to support VLM (Vision-Language Model) benchmarking.
It handles multimodal inputs (text + images) and provides accurate performance measurements.

# Usage Examples:
## Correctness test with real images:
python bench_one_batch_vlm.py --model-path deepseek-ai/Janus-Pro-1B --correctness-test --image-dir ./images

## Performance test with synthetic data:
python bench_one_batch_vlm.py --model-path deepseek-ai/Janus-Pro-1B --batch-size 1 4 8 --input-len 256 512 --output-len 32 128

## Profile VLM performance:
python bench_one_batch_vlm.py --model-path deepseek-ai/Janus-Pro-1B --batch-size 4 --profile
"""

import argparse
import dataclasses
import itertools
import json
import logging
import multiprocessing
import os
import time
from typing import List, Optional, Tuple, Union
import glob
import random

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.distributed.parallel_state import destroy_distributed_environment
from sglang.srt.entrypoints.engine import _set_envs_and_config
from sglang.srt.hf_transformers_utils import get_tokenizer, get_processor
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch, MultimodalInputs, MultimodalDataItem, Modality
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import (
    configure_logger,
    get_bool_env_var,
    kill_process_tree,
    set_gpu_proc_affinity,
    suppress_other_loggers,
    load_image,
)
from sglang.srt.conversation import chat_templates


@dataclasses.dataclass
class VLMBenchArgs:
    run_name: str = "vlm_bench"
    batch_size: Tuple[int] = (1,)
    input_len: Tuple[int] = (256,)
    output_len: Tuple[int] = (32,)
    result_filename: str = "/tmp/vlm_result.jsonl"
    correctness_test: bool = False
    image_dir: str = "./images"
    synthetic_images: bool = True
    num_images_per_req: int = 1
    image_size: Tuple[int] = (224, 224)
    cut_len: int = 4
    log_decode_step: int = 0
    profile: bool = False
    profile_filename_prefix: str = "vlm_profile"

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument("--run-name", type=str, default=VLMBenchArgs.run_name)
        parser.add_argument("--batch-size", type=int, nargs="+", default=VLMBenchArgs.batch_size)
        parser.add_argument("--input-len", type=int, nargs="+", default=VLMBenchArgs.input_len)
        parser.add_argument("--output-len", type=int, nargs="+", default=VLMBenchArgs.output_len)
        parser.add_argument("--result-filename", type=str, default=VLMBenchArgs.result_filename)
        parser.add_argument("--correctness-test", action="store_true")
        parser.add_argument("--image-dir", type=str, default=VLMBenchArgs.image_dir)
        parser.add_argument("--synthetic-images", action="store_true", default=True)
        parser.add_argument("--num-images-per-req", type=int, default=VLMBenchArgs.num_images_per_req)
        parser.add_argument("--image-size", type=int, nargs=2, default=VLMBenchArgs.image_size)
        parser.add_argument("--cut-len", type=int, default=VLMBenchArgs.cut_len)
        parser.add_argument("--log-decode-step", type=int, default=VLMBenchArgs.log_decode_step)
        parser.add_argument("--profile", action="store_true")
        parser.add_argument("--profile-filename-prefix", type=str, default=VLMBenchArgs.profile_filename_prefix)

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        attrs = [(attr.name, type(attr.default)) for attr in dataclasses.fields(cls)]
        return cls(**{attr: getattr(args, attr) for attr, _ in attrs})


def load_model_and_processor(server_args, port_args, tp_rank):
    """Load VLM model and processor"""
    suppress_other_loggers()
    rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None

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
        server_args=server_args,
    )
    
    rank_print(f"max_total_num_tokens={model_runner.max_total_num_tokens}")
    
    # Load tokenizer and processor
    tokenizer = get_tokenizer(
        server_args.tokenizer_path,
        tokenizer_mode=server_args.tokenizer_mode,
        trust_remote_code=server_args.trust_remote_code,
    )
    
    processor = get_processor(
        server_args.tokenizer_path,
        trust_remote_code=server_args.trust_remote_code,
    )
    
    if server_args.tp_size > 1:
        dist.barrier()
    
    return model_runner, tokenizer, processor


def create_synthetic_image(size=(224, 224), seed=None):
    """Create a synthetic image for testing"""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Create random RGB image
    image_array = np.random.randint(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    return Image.fromarray(image_array)


def load_real_images(image_dir: str, num_images: int) -> List[Image.Image]:
    """Load real images from directory"""
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))
        image_files.extend(glob.glob(os.path.join(image_dir, ext.upper())))
    
    if not image_files:
        raise ValueError(f"No images found in {image_dir}")
    
    # Randomly sample images
    selected_files = random.choices(image_files, k=num_images)
    images = []
    for file_path in selected_files:
        try:
            image = Image.open(file_path).convert('RGB')
            images.append(image)
        except Exception as e:
            print(f"Failed to load {file_path}: {e}")
            # Fallback to synthetic image
            images.append(create_synthetic_image())
    
    return images


def prepare_vlm_inputs_for_correctness_test(bench_args, tokenizer, processor):
    """Prepare VLM inputs for correctness testing"""
    prompts = [
        "What is in this image?",
        "Describe the main objects in the image.",
        "What colors do you see in this image?",
    ]
    
    # Load or create images
    if bench_args.synthetic_images or not os.path.exists(bench_args.image_dir):
        images = [create_synthetic_image(bench_args.image_size, seed=i) for i in range(len(prompts))]
    else:
        images = load_real_images(bench_args.image_dir, len(prompts))
    
    # Get chat template
    chat_template = getattr(chat_templates, getattr(processor, 'chat_template', 'default'), None)
    if chat_template is None:
        # Fallback to simple template
        formatted_prompts = prompts
    else:
        formatted_prompts = []
        for prompt in prompts:
            conv = chat_template.copy()
            image_token = getattr(chat_template, 'image_token', '<image>')
            user_prompt = f"{image_token}{prompt}"
            conv.append_message(conv.roles[0], user_prompt)
            formatted_prompts.append(conv.get_prompt())
    
    # Process multimodal inputs
    reqs = []
    sampling_params = SamplingParams(temperature=0, max_new_tokens=bench_args.output_len[0])
    
    for i, (prompt, image) in enumerate(zip(formatted_prompts, images)):
        # Use processor to handle multimodal input
        try:
            if hasattr(processor, 'process_mm_data'):
                mm_result = processor.process_mm_data(input_text=prompt, images=[image])
                input_ids = mm_result.get('input_ids', tokenizer.encode(prompt))
                mm_items = mm_result.get('mm_items', [])
            else:
                # Fallback processing
                input_ids = tokenizer.encode(prompt)
                # Create basic multimodal item
                mm_items = [MultimodalDataItem(
                    pixel_values=torch.rand(1, 3, *bench_args.image_size),  # Dummy pixel values
                    modality=Modality.IMAGE
                )]
            
            if isinstance(input_ids, torch.Tensor):
                input_ids = input_ids.tolist()
            if isinstance(input_ids[0], list):
                input_ids = input_ids[0]
            
            # Truncate for correctness test
            if bench_args.cut_len > 0 and len(input_ids) > bench_args.cut_len:
                input_ids = input_ids[:bench_args.cut_len]
            
            multimodal_inputs = MultimodalInputs(mm_items=mm_items) if mm_items else None
            
            req = Req(
                rid=str(i),
                origin_input_text=prompt,
                origin_input_ids=input_ids,
                sampling_params=sampling_params,
                multimodal_inputs=multimodal_inputs,
            )
            req.prefix_indices = []
            req.fill_ids = req.origin_input_ids
            req.extend_input_len = len(req.fill_ids) - len(req.prefix_indices)
            req.logprob_start_len = len(req.origin_input_ids) - 1
            reqs.append(req)
            
        except Exception as e:
            print(f"Failed to process request {i}: {e}")
            # Fallback to text-only request
            input_ids = tokenizer.encode(prompt)[:bench_args.cut_len] if bench_args.cut_len > 0 else tokenizer.encode(prompt)
            req = Req(
                rid=str(i),
                origin_input_text=prompt,
                origin_input_ids=input_ids,
                sampling_params=sampling_params,
            )
            req.prefix_indices = []
            req.fill_ids = req.origin_input_ids
            req.extend_input_len = len(req.fill_ids)
            req.logprob_start_len = len(req.origin_input_ids) - 1
            reqs.append(req)
    
    return reqs, images


def prepare_synthetic_vlm_inputs_for_latency_test(batch_size, input_len, bench_args):
    """Prepare synthetic VLM inputs for latency testing"""
    # Generate synthetic text inputs
    text_input_ids = np.random.randint(0, 10000, (batch_size, input_len), dtype=np.int32)
    
    # Generate synthetic images
    images = [create_synthetic_image(bench_args.image_size, seed=i) for i in range(batch_size * bench_args.num_images_per_req)]
    
    sampling_params = SamplingParams(temperature=0, max_new_tokens=bench_args.output_len[0])
    
    reqs = []
    for i in range(batch_size):
        # Create multimodal items for this request
        mm_items = []
        for j in range(bench_args.num_images_per_req):
            img_idx = i * bench_args.num_images_per_req + j
            # Create synthetic pixel values
            pixel_values = torch.rand(1, 3, *bench_args.image_size)
            mm_items.append(MultimodalDataItem(
                pixel_values=pixel_values,
                modality=Modality.IMAGE
            ))
        
        multimodal_inputs = MultimodalInputs(mm_items=mm_items) if mm_items else None
        
        req = Req(
            rid=str(i),
            origin_input_text="",
            origin_input_ids=list(text_input_ids[i]),
            sampling_params=sampling_params,
            multimodal_inputs=multimodal_inputs,
        )
        req.prefix_indices = []
        req.fill_ids = req.origin_input_ids
        req.extend_input_len = len(req.fill_ids) - len(req.prefix_indices)
        req.logprob_start_len = len(req.origin_input_ids) - 1
        reqs.append(req)
    
    return reqs


@torch.no_grad
def extend_vlm(reqs, model_runner):
    """Extended version of extend function for VLM"""
    batch = ScheduleBatch.init_new(
        reqs=reqs,
        req_to_token_pool=model_runner.req_to_token_pool,
        token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
        tree_cache=getattr(model_runner, 'tree_cache', None),
        model_config=model_runner.model_config,
        enable_overlap=False,
        spec_algorithm=SpeculativeAlgorithm.NONE,
        enable_custom_logit_processor=False,
    )
    batch.prepare_for_extend()
    _maybe_prepare_dp_attn_batch(batch, model_runner)
    model_worker_batch = batch.get_model_worker_batch()
    forward_batch = ForwardBatch.init_new(model_worker_batch, model_runner)
    logits_output, _ = model_runner.forward(forward_batch)
    next_token_ids = model_runner.sample(logits_output, forward_batch)
    return next_token_ids, logits_output.next_token_logits, batch


@torch.no_grad
def decode_vlm(input_token_ids, batch, model_runner):
    """Extended version of decode function for VLM"""
    batch.output_ids = input_token_ids
    batch.prepare_for_decode()
    _maybe_prepare_dp_attn_batch(batch, model_runner)
    model_worker_batch = batch.get_model_worker_batch()
    forward_batch = ForwardBatch.init_new(model_worker_batch, model_runner)
    logits_output, _ = model_runner.forward(forward_batch)
    next_token_ids = model_runner.sample(logits_output, forward_batch)
    return next_token_ids, logits_output.next_token_logits


def _maybe_prepare_dp_attn_batch(batch: ScheduleBatch, model_runner):
    """Prepare DP attention batch if enabled"""
    if getattr(model_runner.server_args, 'enable_dp_attention', False):
        Scheduler.prepare_dp_attn_batch_raw(
            batch,
            dp_size=model_runner.server_args.dp_size,
            attn_tp_size=1,
            moe_dense_tp_size=getattr(model_runner.server_args, 'moe_dense_tp_size', 1),
            tp_cpu_group=model_runner.tp_group.cpu_group,
            get_idle_batch=None,
            disable_cuda_graph=model_runner.server_args.disable_cuda_graph,
            spec_algorithm=SpeculativeAlgorithm.NONE,
            speculative_num_draft_tokens=None,
        )


def vlm_correctness_test(server_args, port_args, bench_args, tp_rank):
    """VLM correctness test"""
    configure_logger(server_args, prefix=f" TP{tp_rank}")
    rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None

    # Load model and processor
    model_runner, tokenizer, processor = load_model_and_processor(server_args, port_args, tp_rank)

    # Prepare VLM inputs
    reqs, images = prepare_vlm_inputs_for_correctness_test(bench_args, tokenizer, processor)
    rank_print(f"\nPrepared {len(reqs)} VLM requests with {len(images)} images\n")

    # Extend (prefill)
    next_token_ids, next_token_logits, batch = extend_vlm(reqs, model_runner)
    rank_print(f"prefill logits: {next_token_logits} \n")

    # Decode
    output_ids = [[req.origin_input_ids[0] if req.origin_input_ids else 0] + [next_token_ids[i].item()] for i, req in enumerate(reqs)]
    for _ in range(bench_args.output_len[0] - 1):
        next_token_ids, _ = decode_vlm(next_token_ids, batch, model_runner)
        next_token_ids_list = next_token_ids.tolist()
        for i in range(len(reqs)):
            output_ids[i].append(next_token_ids_list[i])

    # Print output texts
    for i, req in enumerate(reqs):
        rank_print(f"========== VLM Prompt {i} ==========")
        rank_print(f"Input: {req.origin_input_text}")
        rank_print(f"Output: {tokenizer.decode(output_ids[i])}")
        rank_print()


def vlm_latency_test_run_once(
    run_name, model_runner, rank_print, reqs, batch_size, input_len, output_len, device, log_decode_step, profile, profile_filename_prefix, bench_args
):
    """VLM latency test for a single configuration"""
    # Estimate memory requirement (including image processing overhead)
    image_memory_overhead = bench_args.num_images_per_req * 0.1  # Rough estimate
    max_batch_size = int(model_runner.max_total_num_tokens // ((input_len + output_len) * (1 + image_memory_overhead)))
    
    if batch_size > max_batch_size:
        rank_print(f"skipping VLM ({batch_size}, {input_len}, {output_len}) due to max batch size limit")
        return

    # Clear the pools
    model_runner.req_to_token_pool.clear()
    model_runner.token_to_kv_pool_allocator.clear()

    measurement_results = {
        "run_name": run_name,
        "batch_size": batch_size,
        "input_len": input_len,
        "output_len": output_len,
        "num_images_per_req": bench_args.num_images_per_req,
        "image_size": bench_args.image_size,
    }

    tot_latency = 0
    profiler = None
    if profile:
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            with_stack=True,
        )
        profiler.start()

    # VLM Prefill
    def synchronize(device):
        torch.get_device_module(device).synchronize()
    
    synchronize(device)
    tic = time.time()
    next_token_ids, _, batch = extend_vlm(reqs, model_runner)
    synchronize(device)
    prefill_latency = time.time() - tic
    tot_latency += prefill_latency
    
    # Calculate throughput (including image processing)
    total_tokens = input_len * batch_size
    throughput = total_tokens / prefill_latency
    rank_print(f"VLM Prefill. latency: {prefill_latency:6.5f} s, throughput: {throughput:9.2f} token/s")
    measurement_results["prefill_latency"] = prefill_latency
    measurement_results["prefill_throughput"] = throughput

    # VLM Decode
    decode_latencies = []
    for i in range(output_len - 1):
        synchronize(device)
        tic = time.time()
        next_token_ids, _ = decode_vlm(next_token_ids, batch, model_runner)
        synchronize(device)
        latency = time.time() - tic
        tot_latency += latency
        throughput = batch_size / latency
        decode_latencies.append(latency)
        if i < 5 or (log_decode_step > 0 and i % log_decode_step == 0):
            rank_print(f"VLM Decode {i}. Batch size: {batch_size}, latency: {latency:6.5f} s, throughput: {throughput:9.2f} token/s")

    if profile:
        profiler.stop()
        profile_filename = f"{profile_filename_prefix}_vlm_batch{batch_size}_input{input_len}_output{output_len}.trace.json.gz"
        parent_dir = os.path.dirname(os.path.abspath(profile_filename))
        os.makedirs(parent_dir, exist_ok=True)
        profiler.export_chrome_trace(profile_filename)
        rank_print(f"VLM torch profiler chrome trace saved to {profile_filename}")

    # Record decode timing
    if output_len > 1:
        med_decode_latency = np.median(decode_latencies)
        med_decode_throughput = batch_size / med_decode_latency
        rank_print(f"VLM Decode. median latency: {med_decode_latency:6.5f} s, median throughput: {med_decode_throughput:9.2f} token/s")
        measurement_results["median_decode_latency"] = med_decode_latency
        measurement_results["median_decode_throughput"] = med_decode_throughput

    throughput = (input_len + output_len) * batch_size / tot_latency
    rank_print(f"VLM Total. latency: {tot_latency:6.3f} s, throughput: {throughput:9.2f} token/s")
    measurement_results["total_latency"] = tot_latency
    measurement_results["overall_throughput"] = throughput
    return measurement_results


def vlm_latency_test(server_args, port_args, bench_args, tp_rank):
    """VLM latency test"""
    if get_bool_env_var("SGLANG_SET_CPU_AFFINITY"):
        set_gpu_proc_affinity(server_args.tp_size, server_args.nnodes, tp_rank)

    configure_logger(server_args, prefix=f" TP{tp_rank}")
    rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None

    # Load model and processor
    model_runner, tokenizer, processor = load_model_and_processor(server_args, port_args, tp_rank)

    # Prepare inputs for warm up
    reqs = prepare_synthetic_vlm_inputs_for_latency_test(
        bench_args.batch_size[0], bench_args.input_len[0], bench_args
    )

    # Warm up
    rank_print("VLM Warmup ...")
    vlm_latency_test_run_once(
        bench_args.run_name, model_runner, rank_print, reqs,
        bench_args.batch_size[0], bench_args.input_len[0], min(32, bench_args.output_len[0]),
        server_args.device, log_decode_step=0, profile=False, profile_filename_prefix="", bench_args=bench_args
    )

    rank_print("VLM Benchmark ...")

    # Run the sweep
    result_list = []
    for bs, il, ol in itertools.product(bench_args.batch_size, bench_args.input_len, bench_args.output_len):
        reqs = prepare_synthetic_vlm_inputs_for_latency_test(bs, il, bench_args)
        ret = vlm_latency_test_run_once(
            bench_args.run_name, model_runner, rank_print, reqs, bs, il, ol,
            server_args.device, bench_args.log_decode_step,
            bench_args.profile if tp_rank == 0 else False, bench_args.profile_filename_prefix, bench_args
        )
        if ret is not None:
            result_list.append(ret)

    # Write results
    if tp_rank == 0 and bench_args.result_filename:
        with open(bench_args.result_filename, "a") as fout:
            for result in result_list:
                fout.write(json.dumps(result) + "\n")

    if server_args.tp_size > 1:
        destroy_distributed_environment()


def main(server_args, bench_args):
    """Main function for VLM benchmarking"""
    server_args.cuda_graph_max_bs = max(bench_args.batch_size)
    _set_envs_and_config(server_args)

    if server_args.model_path:
        if bench_args.correctness_test:
            work_func = vlm_correctness_test
        else:
            work_func = vlm_latency_test
    else:
        raise ValueError("Provide --model-path for running VLM tests")

    port_args = PortArgs.init_new(server_args)

    if server_args.tp_size == 1:
        work_func(server_args, port_args, bench_args, 0)
    else:
        workers = []
        for tp_rank in range(server_args.tp_size):
            proc = multiprocessing.Process(
                target=work_func,
                args=(server_args, port_args, bench_args, tp_rank),
            )
            proc.start()
            workers.append(proc)

        for proc in workers:
            proc.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    VLMBenchArgs.add_cli_args(parser)
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)
    bench_args = VLMBenchArgs.from_cli_args(args)

    logging.basicConfig(
        level=getattr(logging, server_args.log_level.upper()),
        format="%(message)s",
    )

    try:
        main(server_args, bench_args)
    finally:
        if server_args.tp_size != 1:
            kill_process_tree(os.getpid(), include_parent=False) 