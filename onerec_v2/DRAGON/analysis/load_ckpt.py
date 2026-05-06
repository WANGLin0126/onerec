# analysis/load_ckpt.py
import os
import sys
import torch

# 把项目根目录加到 path
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJ_ROOT)

from utils.configurator import Config
from utils.dataset import RecDataset
from utils.dataloader import TrainDataLoader, EvalDataLoader
from utils.utils import init_seed, get_model


# checkpoint 路径
CKPT_PATHS = {
    'baby_sparse': {
        'base': './saved/ckpt_DRAGON_baby_sparse_base.pth',
        'full': './saved/ckpt_DRAGON_baby_sparse_full.pth',
    },
    'clothing_sparse': {
        'base': './saved/ckpt_DRAGON_clothing_sparse_base.pth',
        'full': './saved/ckpt_DRAGON_clothing_sparse_full.pth',
    },
    'sports_sparse': {
        'base': './saved/ckpt_DRAGON_sports_sparse_base.pth',
        'full': './saved/ckpt_DRAGON_sports_sparse_full.pth',
    },
}


# 跟 main.py 里一致的 config_dict (用于重建模型超参)
def _config_dict_for(dataset, tag):
    base_dict = {
        'baby_sparse': {
            'learning_rate': 0.05,
            'mix_bpr_weight_loss': 0.8,
            'dragon_bpr_weight': 0.001,
            'align_weight_loss': 0.1,
            'diver_weight_loss': 0.05,
            'seed': 999,
        },
        'clothing_sparse': {
            'learning_rate': 0.05,
            'mix_bpr_weight_loss': 0.1,
            'dragon_bpr_weight': 0.001,
            'align_weight_loss': 0.001,
            'diver_weight_loss': 0.001,
            'seed': 999,
        },
        'sports_sparse': {
            'learning_rate': 0.05,
            'mix_bpr_weight_loss': 0.01,
            'dragon_bpr_weight': 0.001,
            'align_weight_loss': 0.05,
            'diver_weight_loss': 0.05,
            'seed': 999,
        },
    }
    cfg = dict(base_dict[dataset])
    if tag == 'base':
        cfg.update({
            'use_homogeneity': False, 'use_diversity': False,
            'use_align_loss': False, 'use_residual': False,
        })
    else:
        cfg.update({
            'use_homogeneity': True, 'use_diversity': True,
            'use_align_loss': True, 'use_residual': True,
        })
    cfg['ckpt_tag'] = tag
    return cfg


def load_model_and_data(dataset, tag, device='cpu', model_name='DRAGON'):
    """
    返回: (model, config, train_data, valid_data, test_data, ckpt)
    """
    print(f'[load_ckpt] dataset={dataset}, tag={tag}')

    # 1. 构造 config
    config_dict = _config_dict_for(dataset, tag)
    config = Config(model_name, dataset, config_dict)

    # gpu device 覆盖
    config['device'] = device
    config['gpu_id'] = int(device.split(':')[-1]) if ':' in device else 0

    # 2. seed
    init_seed(config['seed'])

    # 3. 数据集
    full_dataset = RecDataset(config)
    train_dataset, valid_dataset, test_dataset = full_dataset.split()
    
    # 触发 __str__ 来初始化 inter_num 等属性 (照抄 quick_start 的逻辑)
    _ = str(train_dataset)
    _ = str(valid_dataset)
    _ = str(test_dataset)

    train_data = TrainDataLoader(
        config, train_dataset,
        batch_size=config['train_batch_size'], shuffle=True
    )
    valid_data = EvalDataLoader(
        config, valid_dataset, additional_dataset=train_dataset,
        batch_size=config['eval_batch_size']
    )
    test_data = EvalDataLoader(
        config, test_dataset, additional_dataset=train_dataset,
        batch_size=config['eval_batch_size']
    )
    train_data.pretrain_setup()

    # 4. 模型
    model = get_model(model_name)(config, train_data).to(device)

    # 5. 加载 ckpt
    ckpt_path = CKPT_PATHS[dataset][tag]
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'checkpoint 不存在: {ckpt_path}')
    
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:    print(f'[load_ckpt] missing keys: {missing[:5]}{"..." if len(missing)>5 else ""}')
    if unexpected: print(f'[load_ckpt] unexpected keys: {unexpected[:5]}{"..." if len(unexpected)>5 else ""}')
    
    model.eval()
    print(f'[load_ckpt] ✓ loaded {ckpt_path}')
    print(f'[load_ckpt] best_valid_score = {ckpt.get("best_valid_score", "N/A")}')

    return model, config, train_data, valid_data, test_data, ckpt


if __name__ == '__main__':
    # 单独测试
    model, config, train, valid, test, ckpt = load_model_and_data(
        'baby_sparse', 'base', device='cpu')
    print('model:', type(model).__name__)
    print('n_users:', model.n_users, 'n_items:', model.n_items)
    print('valid result:', ckpt.get('valid_result'))