#!/bin/bash

GPU_IDS="0"

accelerate launch --gpu_ids $GPU_IDS train.py \
  --pretrained_model_name_or_path /root/autodl-fs/models/CogVideoX-2b \
  --cache_dir /root/autodl-tmp/cache \
  --instance_data_root /root/autodl-tmp/nuscenes/all \
  --dataset_name nuscenes \
  --validation_prompt "turn left:::wait:::turn left:::turn right:::go straight:::drive slowly:::drive fast:::speed up:::slow down:::maintain speed" \
  --validation_images "/root/diffusers/examples/cogvideo/n015-2018-07-18-11-41-49+0800__CAM_FRONT__1531885320012463.jpg:::/root/diffusers/examples/cogvideo/n015-2018-07-18-11-41-49+0800__CAM_FRONT__1531885326412466.jpg" \
  --validation_prompt_separator ::: \
  --num_validation_videos 1 \
  --validation_epochs 10 \
  --seed 42 \
  --rank 128 \
  --lora_alpha 64 \
  --mixed_precision fp16 \
  --output_dir /root/autodl-fs/ckpt/cogvideo/cogvideox-lora \
  --height 480 --width 720 --fps 8 --max_num_frames 33 \
  --train_batch_size 1 \
  --num_train_epochs 30 \
  --checkpointing_steps 1000 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-3 \
  --lr_scheduler cosine_with_restarts \
  --lr_warmup_steps 200 \
  --lr_num_cycles 1 \
  --enable_slicing \
  --enable_tiling \
  --optimizer Adam \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --max_grad_norm 1.0 \
  --report_to wandb \
  --gradient_checkpointing \
  # --dataloader_num_workers 16 \