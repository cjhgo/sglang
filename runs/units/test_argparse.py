#!/usr/bin/env python3
import argparse
import dataclasses
from typing import Tuple

@dataclasses.dataclass
class TestArgs:
    batch_size: Tuple[int] = (1,)
    input_len: int = 1024

def main():
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="测试 argparse 参数前缀匹配")
    
    # 添加参数
    parser.add_argument( "--batch-size", type=int, nargs="+", default=TestArgs.batch_size
        )
    parser.add_argument("-i", "--input-len", type=int, nargs="+", default=[1024],
                      help="输入长度")
    
    # 解析参数
    args = parser.parse_args()
    
    # 打印参数值
    print("参数值:")
    print(f"batch_size: {args.batch_size}")
    print(f"input_len: {args.input_len}")

if __name__ == "__main__":
    main() 