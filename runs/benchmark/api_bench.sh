CUDA_VISIBLE_DEVICES=4,5,6,7 \
python -m sglang.launch_server --model /home/test/test01/junhangc/modelbest/sync/checkpoint-1170 \
--chat-template minicpmv --trust-remote-code --port 30000 --tp-size 1 --enable-metrics

CUDA_VISIBLE_DEVICES=4,5,6,7 \
python -m sglang.launch_server --model openbmb/MiniCPM-o-2_6 \
--chat-template minicpmo --trust-remote-code --port 30000 --tp-size 1 --enable-metrics

CUDA_VISIBLE_DEVICES=4,5,6,7 \
python benchmark/mmmu/bench_sglang.py --port 30000 --concurrency 32