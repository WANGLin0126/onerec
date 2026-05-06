# coding: utf-8

"""
Main entry
# UPDATED: 2022-Feb-15
##########################
"""
import ast
import os
import argparse
from utils.quick_start import quick_start
os.environ['NUMEXPR_MAX_THREADS'] = '48'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="DRAGON", help="name of model")
    parser.add_argument("--dataset", "-d", type=str, default="baby_sparse", help="name of dataset")

    # Ablation 开关参数（True / False）
    parser.add_argument("--use_homogeneity", type=ast.literal_eval, default=True, help="whether to use homogeneity info (True/False)")
    parser.add_argument("--use_diversity", type=ast.literal_eval, default=True, help="whether to use diversity info (True/False)")
    parser.add_argument("--use_align_loss", type=ast.literal_eval, default=True, help="whether to use visual-text alignment loss (True/False)")
    parser.add_argument("--use_residual", type=ast.literal_eval, default=True, help="whether to use residual connection (True/False)")
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")


    # ============ 修改 1:新增 ckpt_tag 参数 ============
    parser.add_argument("--ckpt_tag", type=str, default="base",
                        help="checkpoint filename tag, e.g. base / full / ablation_xxx")
    # ===================================================

    args = parser.parse_args()

    if args.dataset == 'baby_sparse':
        config_dict = {
            'use_homogeneity': args.use_homogeneity,
            'use_diversity': args.use_diversity,
            'use_align_loss': args.use_align_loss,
            'use_residual': args.use_residual,
            'ckpt_tag': args.ckpt_tag,        # <<< 修改 2:塞进 config_dict
            'learning_rate': [0.05],
            'mix_bpr_weight_loss': [0.8],
            'dragon_bpr_weight': [0.001],
            'align_weight_loss': [0.1],
            'diver_weight_loss': [0.05],
            'seed': [999],
        }
    elif args.dataset == 'clothing_sparse':
        config_dict = {
            'use_homogeneity': args.use_homogeneity,
            'use_diversity': args.use_diversity,
            'use_align_loss': args.use_align_loss,
            'use_residual': args.use_residual,
            'ckpt_tag': args.ckpt_tag,        # <<< 修改 2
            'learning_rate': [0.05],
            'mix_bpr_weight_loss': [0.1],
            'dragon_bpr_weight': [0.001],
            'align_weight_loss': [0.001],
            'diver_weight_loss': [0.001],
            'seed': [999],
        }
    elif args.dataset == 'sports_sparse':
        config_dict = {
            'use_homogeneity': args.use_homogeneity,
            'use_diversity': args.use_diversity,
            'use_align_loss': args.use_align_loss,
            'use_residual': args.use_residual,
            'ckpt_tag': args.ckpt_tag,        # <<< 修改 2
            'learning_rate': [0.05],
            'mix_bpr_weight_loss': [0.01],
            'dragon_bpr_weight': [0.001],
            'align_weight_loss': [0.05],
            'diver_weight_loss': [0.05],
            'seed': [999],
        }


    args, _ = parser.parse_known_args()
    print("config_dict", config_dict)
    quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=True)


# nohup python main.py --dataset baby >log/test_add_dragonv3.log 2>&1 &
# nohup python main.py \
#     --model DRAGON \
#     --dataset baby_sparse \
#     --use_homogeneity False \
#     --use_diversity False \
#     --use_align_loss False \
#     --use_residual False \
#     --ckpt_tag base \
#     --gpu_id 0 \
#     > log/baby_base.log 2>&1 &



# nohup python main.py \
#     --model DRAGON \
#     --dataset clothing_sparse \
#     --use_homogeneity False \
#     --use_diversity False \
#     --use_align_loss False \
#     --use_residual False \
#     --ckpt_tag base \
#     --gpu_id 0 \
#     > log/clothing_base.log 2>&1 &


# nohup python main.py \
#     --model DRAGON \
#     --dataset sports_sparse \
#     --use_homogeneity False \
#     --use_diversity False \
#     --use_align_loss False \
#     --use_residual False \
#     --ckpt_tag base \
#     --gpu_id 0 \
#     > log/sports_base.log 2>&1 &



# nohup python main.py --model DRAGON --dataset baby_sparse \
#     --use_homogeneity True --use_diversity True \
#     --use_align_loss True --use_residual True \
#     --ckpt_tag full --gpu_id 0 \
#     > log/baby_full.log 2>&1 &


# nohup python main.py --model DRAGON --dataset clothing_sparse \
#     --use_homogeneity True --use_diversity True \
#     --use_align_loss True --use_residual True \
#     --ckpt_tag full --gpu_id 0 \
#     > log/clothing_full.log 2>&1 &


# nohup python main.py --model DRAGON --dataset sports_sparse \
#     --use_homogeneity True --use_diversity True \
#     --use_align_loss True --use_residual True \
#     --ckpt_tag full --gpu_id 0 \
#     > log/sports_full.log 2>&1 &


# tail -f log/baby_full.log
# tail -f log/baby_base.log

# tail -f log/clothing_full.log
# tail -f log/clothing_base.log

# tail -f log/sports_full.log
# tail -f log/sports_base.log
