import sglang as sgl
from sglang.srt.hf_transformers_utils import get_tokenizer

DEFAULT_SMALL_MODEL_NAME_FOR_TEST="Qwen/Qwen2.5-0.5B-Instruct"
def test_engine():
    # just to ensure there is no issue running multiple generate calls
    prompt = "Today is a sunny day and I like"
    model_path = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
    sampling_params = {"temperature": 0, "max_new_tokens": 8}

    engine = sgl.Engine(
        model_path=model_path, random_seed=42, disable_radix_cache=True
    )
    out1 = engine.generate(prompt, sampling_params)["text"]

    tokenizer = get_tokenizer(model_path)
    token_ids = tokenizer.encode(prompt)
    out2 = engine.generate(input_ids=token_ids, sampling_params=sampling_params)[
        "text"
    ]
    import ipdb; ipdb.set_trace()

    engine.shutdown()

    print("==== Answer 1 ====")
    print(out1)

    print("==== Answer 2 ====")
    print(out2)
if __name__ == "__main__":
    test_engine()
  