#!/usr/bin/env bash
trap "exit" INT TERM ERR
trap "kill 0" EXIT

python main.py --env=hopper --with_embed --robot_dir=../xml/gen_xmls/hopper --save_dir=hopper/with_embed --seed=55326218 &

python main.py --env=hopper --robot_dir=../xml/gen_xmls/hopper --save_dir=hopper/no_embed --seed=55326218 &

python main.py --env=hopper --with_kin --with_dyn --robot_dir=../xml/gen_xmls/hopper --save_dir=hopper/kin_dyn --seed=55326218 &
#python main.py --embed_dim=16 --env=inv_pen --robot_dir=/home/taochen/PycharmProjects/walker/robot_dataset/embed_dataset/inv_pen/1000_train_medium_dyn --save_dir=inv_pen/train_embed_16/seed_71230811 --seed=71230811 --nsteps=1000 &
#
#python main.py --embed_dim=8 --env=inv_pen --robot_dir=/home/taochen/PycharmProjects/walker/robot_dataset/embed_dataset/inv_pen/1000_train_medium_dyn --save_dir=inv_pen/train_embed_8/seed_71230811 --seed=71230811 --nsteps=1000 &
#
#python main.py --embed_dim=4 --env=inv_pen --robot_dir=/home/taochen/PycharmProjects/walker/robot_dataset/embed_dataset/inv_pen/1000_train_medium_dyn --save_dir=inv_pen/train_embed_4/seed_71230811 --seed=71230811 --nsteps=1000 &


#python main.py --robot_num=100 --with_embed --resume --new_robot --env=hopper --robot_dir=/home/taochen/PycharmProjects/walker/robot_dataset/embed_dataset/hopper/100_eval --pretrain_dir=/home/taochen/PycharmProjects/walker/train/hopper_transfer/train_embed/pretrain/hopper --save_dir=/home/taochen/Desktop/result/hopper_embed_finetune/finetune/seed_71230811 --seed=71230811 &



#python main.py --robot_num=1000 --env=hopper --robot_dir=/home/tao/PycharmProjects/robot_dataset/hopper/1000_torso_mass --save_dir=/home/tao/Desktop/result/hopper/basic/seed_55326218 --seed=55326218 &
#
#python main.py --robot_num=1000 --with_embed --env=hopper --robot_dir=/home/tao/PycharmProjects/robot_dataset/hopper/1000_torso_mass --save_dir=/home/tao/Desktop/result/hopper/train_embed_32/seed_55326218 --seed=55326218 &
#
#python main.py --embed_dim=8 --robot_num=1000 --with_embed --env=hopper --robot_dir=/home/tao/PycharmProjects/robot_dataset/hopper/1000_torso_mass --save_dir=/home/tao/Desktop/result/hopper/train_embed_8/seed_55326218 --seed=55326218 &

# python main.py --embed_dim=8 --robot_num=1000 --with_embed --env=hopper --robot_dir=/home/tao/PycharmProjects/robot_dataset/hopper/1000_torso_mass2 --save_dir=/home/tao/Desktop/result/hopper2/train_embed_8/seed_938691614 --seed=938691614 &

# python main.py --embed_dim=8 --robot_num=1000 --with_embed --env=hopper --robot_dir=/home/tao/PycharmProjects/robot_dataset/hopper/1000_torso_mass2 --save_dir=/home/tao/Desktop/result/hopper2/train_embed_8/seed_770355848 --seed=770355848 &
#
# python main.py --embed_dim=8 --robot_num=1000 --with_embed --env=hopper --robot_dir=/home/tao/PycharmProjects/robot_dataset/hopper/1000_torso_mass2 --save_dir=/home/tao/Desktop/result/hopper2/train_embed_8/seed_265317347 --seed=265317347 &





#python main.py --with_kin --with_dyn --env=hopper --robot_dir=/home/taochen/PycharmProjects/walker/robot_dataset/embed_dataset/hopper/1000_train --save_dir=/home/taochen/Desktop/result/hopper/kin_dyn/seed_55326218 --seed=55326218 &

#python main.py --with_kin --env=hopper --robot_dir=/home/taochen/PycharmProjects/walker/robot_dataset/embed_dataset/hopper/1000_train --save_dir=/home/taochen/Desktop/result/hopper/kin/seed_55326218 --seed=55326218 &

#
#python main.py --resume --new_robot --env=hopper --robot_dir=/home/taochen/PycharmProjects/walker/robot_dataset/embed_dataset/hopper/100_eval --pretrain_dir=/home/taochen/PycharmProjects/walker/train/hopper_transfer/train_embed/pretrain/hopper --save_dir=/home/taochen/Desktop/paper/hopper_embed_finetune/finetune/seed_55326218 --seed=55326218 &
#
#python main.py --resume --new_robot --env=hopper --robot_dir=/home/taochen/PycharmProjects/walker/robot_dataset/embed_dataset/hopper/100_eval --pretrain_dir=/home/taochen/PycharmProjects/walker/train/hopper_transfer/train_embed/pretrain/hopper --save_dir=/home/taochen/Desktop/paper/hopper_embed_finetune/finetune/seed_31202383 --seed=31202383 &
#
#python main.py --resume --new_robot --env=hopper --robot_dir=/home/taochen/PycharmProjects/walker/robot_dataset/embed_dataset/hopper/100_eval --pretrain_dir=/home/taochen/PycharmProjects/walker/train/hopper_transfer/train_embed/pretrain/hopper --save_dir=/home/taochen/Desktop/paper/hopper_embed_finetune/finetune/seed_33379960 --seed=33379960 &
#
#
# python main.py --env=hopper --robot_dir=/home/tao/project/robot_dataset/hopper/300_damping_eval --save_dir=/home/tao/project/result/hopper_damping/seed_71230811 --seed=71230811 &

# python main.py --env=hopper --robot_dir=/home/tao/project/robot_dataset/hopper/300_damping_eval --save_dir=/home/tao/project/result/hopper_damping/seed_55326218 --seed=55326218 &

#python main.py --env=hopper --robot_dir=/home/tao/project/robot_dataset/hopper/100_eval --save_dir=/home/tao/project/result/hopper_embed_finetune/scratch/seed_31202383 --seed=31202383 &

#python main.py --env=hopper --robot_dir=/home/tao/project/robot_dataset/hopper/100_eval --save_dir=/home/tao/project/result/hopper_embed_finetune/scratch/seed_33379960 --seed=33379960 &
#python main.py --resume --new_robot --env=hopper --robot_dir=/home/taochen/PycharmProjects/walker/robot_dataset/embed_dataset/hopper/100_eval --pretrain_dir=/home/taochen/PycharmProjects/walker/train/hopper_transfer/train_embed/pretrain/hopper --save_dir=/home/taochen/Desktop/paper/hopper_embed_finetune/fintune/seed_67819713 --seed=67819713 &

#python main.py --env=walker2d --robot_dir=/home/taoc1/software/train_embed/embed_dataset/walker/1000_train --save_dir=/scratch/taoc1/walker2d/train_embed/seed_71230811 --seed=71230811 &
#
#python main.py --env=walker2d --robot_dir=/home/taoc1/software/train_embed/embed_dataset/walker/1000_train --save_dir=/scratch/taoc1/walker2d/train_embed/seed_55326218 --seed=55326218 &
#
#python main.py --env=walker2d --robot_dir=/home/taoc1/software/train_embed/embed_dataset/walker/1000_train --save_dir=/scratch/taoc1/walker2d/train_embed/seed_31202383 --seed=31202383 &
#
#python main.py --env=walker2d --robot_dir=/home/taoc1/software/train_embed/embed_dataset/walker/1000_train --save_dir=/scratch/taoc1/walker2d/train_embed/seed_33379960 --seed=33379960 &
#
#python main.py --env=walker2d --robot_dir=/home/taoc1/software/train_embed/embed_dataset/walker/1000_train --save_dir=/scratch/taoc1/walker2d/train_embed/seed_67819713 --seed=67819713 &
#
#
#
#python main.py --env=hopper --robot_dir=/home/taoc1/software/train_embed/embed_dataset/hopper/1000_train --save_dir=/scratch/taoc1/hopper/train_embed/seed_55326218 --seed=55326218 &
#
#python main.py --env=hopper --robot_dir=/home/taoc1/software/train_embed/embed_dataset/hopper/1000_train --save_dir=/scratch/taoc1/hopper/train_embed/seed_31202383 --seed=31202383 &
#
#python main.py --env=hopper --robot_dir=/home/taoc1/software/train_embed/embed_dataset/hopper/1000_train --save_dir=/scratch/taoc1/hopper/train_embed/seed_33379960 --seed=33379960 &
#
#python main.py --env=hopper --robot_dir=/home/taoc1/software/train_embed/embed_dataset/hopper/1000_train --save_dir=/scratch/taoc1/hopper/train_embed/seed_67819713 --seed=67819713 &

#python main.py --robot_dir=/home/taoc1/software/hopper/1000_eval_small_range  --with_embed --with_kin --save_dir=/scratch/taoc1/hopper_embed_with_body_pos/embed_kin &
#
#python main.py --robot_dir=/home/taoc1/software/hopper/1000_eval_small_range --with_kin --save_dir=/scratch/taoc1/hopper_embed_with_body_pos/kin &
#
#python main.py --robot_dir=/home/taoc1/software/hopper/1000_eval_small_range  --with_embed --save_dir=/scratch/taoc1/hopper_embed_with_body_pos/embed &

#python main.py --resume --new_task --backward --robot_dir=/home/taochen/PycharmProjects/walker/robot_dataset/cross_small_range/1000_eval --save_dir=backward_pretrain &
#
#python main.py --backward --robot_dir=/home/taochen/PycharmProjects/walker/robot_dataset/cross_small_range/1000_eval --save_dir=backward_scratch &
#
#python main.py --resume --new_robot --robot_dir=/home/taochen/PycharmProjects/walker/robot_dataset/cross_small_range/100_eval --save_dir=new_robot_forward_pretrain_fixed &

#python main.py --robot_dir=/home/taochen/PycharmProjects/walker/robot_dataset/cross_small_range/100_eval --save_dir=new_robot_forward_scratch &

#python main.py --robot_dir=/home/taochen/PycharmProjects/walker/robot_dataset/cross_small_range/300_eval  --with_embed --with_kin --save_dir=hang_small_range_100/embed_kin --train_ratio=0.67 &
#python main.py --robot_dir=/home/taochen/PycharmProjects/walker/robot_dataset/cross_small_range/300_eval   --with_kin --save_dir=hang_small_range_100/kin --train_ratio=0.67 &

#python main.py --robot_dir=/home/taochen/PycharmProjects/walker/robot_dataset/limit_kin/1000_eval  --with_embed --with_kin --save_dir=hang/embed_kin &

#python main.py --robot_dir=/home/taochen/PycharmProjects/walker/robot_dataset/limit_kin/1000_eval  --with_embed --save_dir=embed

#python main.py --robot_dir=/home/taochen/PycharmProjects/walker/robot_dataset/limit_kin/1000_eval  --with_kin --save_dir=hang/kin

#python main.py --robot_dir=/home/taochen/PycharmProjects/walker/robot_dataset/1000_eval_small_range  --with_embed --with_kin --save_dir=embed_kin &
#python main.py --robot_dir=/home/taochen/PycharmProjects/walker/robot_dataset/1000_eval_small_range  --with_embed --with_kin --with_dyn --save_dir=embed_kin_dyn &
#python main.py --robot_dir=/home/taochen/PycharmProjects/walker/robot_dataset/1000_eval_small_range  --with_embed --save_dir=embed &
#python main.py --robot_dir=/home/taochen/PycharmProjects/walker/robot_dataset/1000_eval_small_range  --with_kin --with_dyn --save_dir=kin_dyn &
#python main.py --robot_dir=/home/taochen/PycharmProjects/walker/robot_dataset/1000_eval_small_range  --with_kin --save_dir=kin &
#python main.py --robot_dir=/home/taochen/PycharmProjects/walker/robot_dataset/1000_eval_small_range --save_dir=basic &
#CUDA_VISIBLE_DEVICES=3 nice -19 python main.py --seed=187372311 --ent_coef=0.015 --noptepochs=5 --lr=0.0001 --save_dir=/scratch/taoc1/rand_seed_187372311 &
#CUDA_VISIBLE_DEVICES=3 nice -19 python main.py --seed=94412880 --ent_coef=0.015 --noptepochs=5 --lr=0.0001 --save_dir=/scratch/taoc1/rand_seed_94412880 &
#CUDA_VISIBLE_DEVICES=3 nice -19 python main.py --seed=146764631 --ent_coef=0.015 --noptepochs=5 --lr=0.0001 --save_dir=/scratch/taoc1/rand_seed_146764631 &
#CUDA_VISIBLE_DEVICES=3 nice -19 python main.py --seed=6155814 --ent_coef=0.015 --noptepochs=5 --lr=0.0001 --save_dir=/scratch/taoc1/rand_seed_6155814 &
# =========================
#good:
# CUDA_VISIBLE_DEVICES=1 nice -19 python main.py --ent_coef=0.015 --lr=0.0001 --noptepochs=5 --hid2_dim=256 --save_dir=/scratch/taoc1/nopt_5_ent_015_hid2_256_lr_0_0001 &
# CUDA_VISIBLE_DEVICES=3 nice -19 python main.py --ent_coef=0.015 --noptepochs=5 --lr=0.0001 --save_dir=/scratch/taoc1/ent_015_nopt_5_lr_0_0001 &
# =========================


# CUDA_VISIBLE_DEVICES=0 nice -19 python main.py --save_dir=/scratch/taoc1/base &
#CUDA_VISIBLE_DEVICES=0 nice -19 python main.py --hid2_dim=256 --noptepochs=5 --save_dir=/scratch/taoc1/hid2_256_noptepochs_5 &
#CUDA_VISIBLE_DEVICES=0 nice -19 python main.py --ent_coef=0.015 --noptepochs=5 --save_dir=/scratch/taoc1/ent_015_nopt_5 &
#CUDA_VISIBLE_DEVICES=1 nice -19 python main.py --lr=0.0001 --hid2_dim=256 --save_dir=/scratch/taoc1/hid2_256_lr_0_0001 &

#CUDA_VISIBLE_DEVICES=2 nice -19 python main.py --ent_coef=0.015 --hid2_dim=256 --save_dir=/scratch/taoc1/hid2_256_ent_coef_0_015 &
#CUDA_VISIBLE_DEVICES=2 nice -19 python main.py --ent_coef=0.015 --lr=0.0001 --hid2_dim=256 --save_dir=/scratch/taoc1/ent_015_hid2_256_lr_0_0001 &
#CUDA_VISIBLE_DEVICES=3 nice -19 python main.py --ent_coef=0.015 --noptepochs=5 --hid2_dim=256 --save_dir=/scratch/taoc1/nopt_5_ent_015_hid2_256 &

# python main.py --nminibatches=80 --save_dir=./nminibatches_80 &

#python main.py --hid1_dim=256 --save_dir=./hid1_256 &
#python main.py --hid2_dim=256 --save_dir=./hid2_256 &
#python main.py --hid1_dim=256 --hid2_dim=256 --save_dir=./hid12_256 &


# python main.py --lr=0.0001 --save_dir=./lr_0_0001 &
# python main.py --ent_coef=0.001 --save_dir=./ent_0_001 &
# python main.py --ent_coef=0.005 --save_dir=./ent_0_005 &
# python main.py --ent_coef=0.012 --save_dir=./ent_0_012 &
# python main.py --max_grad_norm=3.0 --save_dir=./grad_3 &
# python main.py --max_grad_norm=10.0 --save_dir=./grad_10 &
wait
