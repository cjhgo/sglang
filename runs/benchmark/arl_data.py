from pathlib import Path
abs_base=Path(__file__).resolve().parent
import site
site.addsitedir(f"{abs_base}")
from vlmt.data_utils import ArlDataset
import logging
logging.basicConfig(level=logging.INFO)



mb_base=f"{abs_base}/../../../../modelbest/sync/"
ds_base=f"{mb_base}/sft_data/"
ds_path=f"{ds_base}/multi_turn/multi_turn_merge_open_app.jsonl"

def load_arl_dataset():
    dataset = ArlDataset(jsonl_file_path=ds_path)
    return dataset

def test_arl_dataset():
    dataset = load_arl_dataset()
    batch = [dataset[i] for i in range(16)]  # 获取4个样本
    import ipdb;ipdb.set_trace()
    for item in batch:
        print(item)
        break



if __name__ == "__main__":
    test_arl_dataset()