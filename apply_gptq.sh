#!/bin/bash
model_repo="HuggingFaceH4"
model_name="zephyr-7b-beta"
hf_namespace="robertgshaw2"
channelwise=true
push_to_hub=true

model_id="$model_repo/$model_name"

if [ "$channelwise" = true ]; then
    gptq_name=$model_name-channelwise-gptq
    marlin_name=$model_name-channelwise-marlin
    gptq_save_dir="./$gptq_name"
    marlin_save_dir="./$marlin_name"

    # python3 apply_gptq_save_marlin.py \
    #     --model-id $model_id \
    #     --gptq-save-dir $gptq_save_dir \
    #     --marlin-save-dir $marlin_save_dir \
    #     --channelwise

else
    gptq_name=$model_name-g128-gptq
    marlin_name=$model_name-g128-marlin
    gptq_save_dir="./$gptq_name"
    marlin_save_dir="./$marlin_name"

    # python3 apply_gptq_save_marlin.py \
    #     --model-id $model_id \
    #     --gptq-save-dir $gptq_save_dir \
    #     --marlin-save-dir $marlin_save_dir
fi

# rm $gptq_save_dir/autogptq_model.safetensors

if [ "$push_to_hub" = true ]; then
    echo "Pushing to HF hub ..."
    python3 push_to_hub.py --model $gptq_name --hf-namespace $hf_namespace
    python3 push_to_hub.py --model $marlin_name --hf-namespace $hf_namespace
fi
