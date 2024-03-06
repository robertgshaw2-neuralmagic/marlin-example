import argparse, gc, shutil
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from datasets import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--model-id", type=str)
parser.add_argument("--save-dir", type=str)
parser.add_argument("--channelwise", action="store_true")
parser.add_argument("--num-samples", type=int, default=512)
parser.add_argument("--max-seq-len", type=int, default=2048)


def preprocess(example):
        return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}

if __name__ == "__main__":
    args = parser.parse_args()

    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:5%]")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    ds = dataset.shuffle().select(range(args.num_samples))
    ds = ds.map(preprocess)

    examples = [
        tokenizer(
            example["text"], padding=False, max_length=args.max_seq_len, truncation=True,
        ) for example in ds
    ]

    if args.channelwise:
        group_size = -1
    else:
        group_size = 128

    quantize_config = BaseQuantizeConfig(
        bits=4,                         # Only support 4 bit
        group_size=group_size,          # Set to g=128 or -1 (for channelwise)
        desc_act=False,                 # Marlin does not suport act_order=True
        model_file_base_name="model"    # Name of the model.safetensors when we call save_pretrained
    )

    model = AutoGPTQForCausalLM.from_pretrained(
        args.model_id, 
        quantize_config, 
        device_map="auto")
    model.quantize(examples)
    
    gptq_save_dir = "./tmp-gptq"
    print(f"Saving gptq model to {gptq_save_dir}")
    model.save_pretrained(gptq_save_dir)
    tokenizer.save_pretrained(gptq_save_dir)

    del model
    gc.collect()

    print("Reloading in marlin format")
    
    marlin_model = AutoGPTQForCausalLM.from_quantized(
        gptq_save_dir, 
        use_marlin=True, 
        device_map="auto")

    print("Saving in marlin format")
    marlin_model.save_pretrained(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)

    shutil.rmtree(gptq_save_dir)
