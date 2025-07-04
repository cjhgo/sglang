import json
import logging
import datetime
import os
from typing import List, Dict, Any


def process_bench_result(batch_input, out, input_mode: str, model_path: str) -> dict:
    """处理基准测试结果，包括 token 统计和性能指标计算"""
    # 提取详细的性能数据
    meta_info = out[-1]['meta_info']
    cur_e2e = meta_info['e2e_latency']
    
    # 计算额外的性能指标
    num_prompts = len(batch_input)
    # 处理不同格式的 token 统计信息
    prompt_tokens = meta_info.get('prompt_tokens', 0)
    completion_tokens = meta_info.get('completion_tokens', 0)
    
    if isinstance(prompt_tokens, list):
        total_input_tokens = sum(prompt_tokens)
    else:
        total_input_tokens = prompt_tokens * num_prompts if prompt_tokens else 0
        
    if isinstance(completion_tokens, list):
        total_output_tokens = sum(completion_tokens)
    else:
        total_output_tokens = completion_tokens * num_prompts if completion_tokens else 0
        
    total_tokens = total_input_tokens + total_output_tokens
    
    # 构建结果记录
    result = {
        "model_path": model_path,
        "batch_size": len(batch_input),
        "input_mode": input_mode,
        "e2e_latency": cur_e2e,
        "num_prompts": num_prompts,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "throughput_token_per_s": total_tokens / cur_e2e if cur_e2e > 0 else 0,
        "throughput_prompt_per_s": num_prompts / cur_e2e if cur_e2e > 0 else 0,
        "meta_info": meta_info
    }
    
    logging.error(f"{input_mode} mode - batch_size: {result['batch_size']}, "
                 f"e2e_time: {cur_e2e:.4f}s, throughput: {result['throughput_token_per_s']:.2f} tokens/s")
    
    return result


def save_bench_results(results: List[Dict[str, Any]], model_path: str, prefix: str = "vlm_bench") -> str:
    """保存基准测试结果到 JSONL 文件"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = model_path.replace('/', '_') if model_path else "unknown_model"
    result_filename = f"runs/benchmark/results/{prefix}_{model_name}_{timestamp}.jsonl"
    
    # 确保结果目录存在
    os.makedirs(os.path.dirname(result_filename), exist_ok=True)
    
    # 将结果写入 JSONL 文件
    with open(result_filename, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    logging.info(f"Results saved to: {result_filename}")
    
    # 打印摘要
    logging.info("\n=== Performance Summary ===")
    for result in results:
        logging.info(f"{result['input_mode']} mode - batch_size: {result['batch_size']}, "
                    f"e2e_time: {result['e2e_latency']:.4f}s, "
                    f"throughput: {result['throughput_token_per_s']:.2f} tokens/s")
    
    return result_filename


def quick_report(results: List[Dict[str, Any]], title: str = "Performance Summary") -> None:
    """以表格格式快速显示性能数据摘要"""
    if not results:
        print("No results to display")
        return
    
    # 打印标题
    print(f"\n{'='*80}")
    print(f"{title:^80}")
    print(f"{'='*80}")
    
    # 准备表头
    headers = ["Batch Size", "Mode", "E2E Latency(s)", "Tokens/s", "Prompts/s", "Total Tokens"]
    col_widths = [12, 8, 15, 12, 12, 13]
    
    # 打印表头
    header_line = "|".join(f"{h:^{w}}" for h, w in zip(headers, col_widths))
    print(header_line)
    print("-" * len(header_line))
    
    # 打印数据行
    for result in results:
        row_data = [
            str(result['batch_size']),
            result['input_mode'],
            f"{result['e2e_latency']:.3f}",
            f"{result['throughput_token_per_s']:.1f}",
            f"{result['throughput_prompt_per_s']:.2f}",
            str(result['total_tokens'])
        ]
        row_line = "|".join(f"{d:^{w}}" for d, w in zip(row_data, col_widths))
        print(row_line)
    
    # 打印统计信息
    print(f"{'='*80}")
    
    # 计算平均值
    avg_latency = sum(r['e2e_latency'] for r in results) / len(results)
    avg_token_throughput = sum(r['throughput_token_per_s'] for r in results) / len(results)
    avg_prompt_throughput = sum(r['throughput_prompt_per_s'] for r in results) / len(results)
    
    print(f"Average E2E Latency: {avg_latency:.3f}s")
    print(f"Average Token Throughput: {avg_token_throughput:.1f} tokens/s")
    print(f"Average Prompt Throughput: {avg_prompt_throughput:.2f} prompts/s")
    
    # 找出最佳批次大小（基于 token 吞吐量）
    best_result = max(results, key=lambda x: x['throughput_token_per_s'])
    print(f"Best Batch Size: {best_result['batch_size']} (Throughput: {best_result['throughput_token_per_s']:.1f} tokens/s)")
    print(f"{'='*80}\n") 