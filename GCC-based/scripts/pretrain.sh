#!/bin/bash

python -u train.py \
  --exp Pretrain \
  --model-path saved \
  --tb-path tensorboard \
  --gpu 0 \
  --moco \
  --dataset "collab" \
  --nce-k 16384 >result_pretrain1.out &
python -u train.py \
  --exp Pretrain \
  --model-path saved \
  --tb-path tensorboard \
  --gpu 1 \
  --moco \
  --dataset "imdb-multi" \
  --nce-k 16384 >result_pretrain2.out &
python -u train.py \
  --exp Pretrain \
  --model-path saved \
  --tb-path tensorboard \
  --gpu 5 \
  --moco \
  --dataset "rdt-5k" \
  --nce-k 16384 >result_pretrain3.out &
python -u train.py \
  --exp Pretrain \
  --model-path saved \
  --tb-path tensorboard \
  --gpu 6 \
  --moco \
  --dataset "rdt-b" \
  --nce-k 16384 >result_pretrain4.out

#load_path1="/home/syf/workspace/GCC_ori/GCC_old/saved/FT_moco_False_usa_airport_gin_layer_5_lr_0.005_decay_1e-05_bsz_32_hid_64_samples_2000_nce_t_0.07_nce_k_32_rw_hops_256_restart_prob_0.8_aug_1st_ft_True_deg_16_pos_32_momentum_0.999/current.pth"

#declare -A epochs=(["usa_airport"]=30 ["h-index"]=30 ["imdb-binary"]=30 ["imdb-multi"]=30 ["collab"]=30 ["rdt-b"]=100 ["rdt-5k"]=100)

#python -u train.py --exp FT --model-path saved --tb-path tensorboard --tb-freq 5 --gpu 1 --dataset "imdb-binary" --finetune --epochs 30 --resume "$load_path1/current.pth" --cv >result_f1.out &
#python -u train.py --exp FT --model-path saved --tb-path tensorboard --tb-freq 5 --gpu 2 --dataset "imdb-binary" --finetune --epochs 30 --resume "$load_path2/current.pth" --cv >result_f2.out &
#python -u train.py --exp FT --model-path saved --tb-path tensorboard --tb-freq 5 --gpu 5 --dataset "imdb-binary" --finetune --epochs 30 --resume "$load_path3/current.pth" --cv >result_f3.out &
#python -u train.py --exp FT --model-path saved --tb-path tensorboard --tb-freq 5 --gpu 6 --dataset "imdb-binary" --finetune --epochs 30 --resume "$load_path4/current.pth" --cv >result_f4.out &
