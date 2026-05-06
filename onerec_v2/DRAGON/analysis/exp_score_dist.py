# analysis/exp_score_dist.py
import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from scipy.stats import wasserstein_distance

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_ckpt import load_model_and_data
from dragon_utils import warmup_dragon

# ============ 配置 ============
DATASET = 'clothing_sparse'
DEVICE  = 'cpu'
N_NEG   = 99
SEED    = 42
SPARSE_THR = (1, 5)
MEDIUM_THR = (5, 20)
OUT_DIR = 'analysis/results'    
# ==============================


def collect_train_inter(model):
    """从 model.interaction_matrix 拿训练交互"""
    res = defaultdict(set)
    inter = model.interaction_matrix.tocoo()
    for u, i in zip(inter.row, inter.col):
        res[int(u)].add(int(i))
    return res


def collect_test_inter(test_data):
    """从 EvalDataLoader 拿测试集 user -> pos items"""
    res = {}
    eval_u = test_data.eval_u
    eval_users = eval_u.cpu().numpy() if hasattr(eval_u, 'cpu') else eval_u
    eval_items = test_data.get_eval_items()
    for u, items in zip(eval_users, eval_items):
        res[int(u)] = set(int(i) for i in items)
    return res


@torch.no_grad()
def score_one_user(model, u, device):
    """DRAGON full_sort_predict 接收 interaction = [user_tensor, mask_matrix]
    我们这里只关心 user, mask 不传 (置空)"""
    u_tensor = torch.LongTensor([u]).to(device)
    # 给 mask_matrix 一个空 tensor (避免后续可能的 indexing 出错)
    empty_mask = torch.zeros((2, 0), dtype=torch.long).to(device)
    interaction = [u_tensor, empty_mask]
    s = model.full_sort_predict(interaction)
    return s.detach().squeeze().cpu().numpy()


def collect_scores(model, train_inter, test_inter, n_items,
                   n_neg=99, seed=42, device='cuda:0'):
    warmup_dragon(model)
    rng = np.random.RandomState(seed)
    rows = []

    test_users = list(test_inter.keys())
    n_total = len(test_users)
    for idx, u in enumerate(test_users):
        if idx % 200 == 0:
            print(f'  scoring {idx}/{n_total}')

        pos_items = list(test_inter[u])
        if len(pos_items) == 0:
            continue

        seen = train_inter.get(u, set()) | set(pos_items)
        cand = np.setdiff1d(np.arange(n_items),
                            np.array(list(seen), dtype=np.int64))
        if len(cand) == 0:
            continue
        n_take = min(n_neg, len(cand))
        neg_items = rng.choice(cand, n_take, replace=False)

        scores = score_one_user(model, u, device)

        for i in pos_items:
            if i < len(scores):
                rows.append((u, 'pos', float(scores[i])))
        for i in neg_items:
            rows.append((u, 'neg', float(scores[i])))

    return pd.DataFrame(rows, columns=['user', 'label', 'score'])


def normalize_per_user(df):
    df = df.copy()
    df['score_norm'] = df.groupby('user')['score'].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-8))
    return df


def compute_metrics(df, user_set=None):
    if user_set is not None:
        df = df[df['user'].isin(user_set)]
    if len(df) == 0 or df.label.nunique() < 2:
        return {'gap': np.nan, 'auc': np.nan, 'wdist': np.nan, 'n_users': 0}

    df_n = normalize_per_user(df)
    gap = (df_n[df_n.label == 'pos']['score_norm'].mean() -
           df_n[df_n.label == 'neg']['score_norm'].mean())

    aucs = []
    for u, g in df.groupby('user'):
        if g.label.nunique() < 2:
            continue
        try:
            aucs.append(roc_auc_score(
                (g.label == 'pos').astype(int).values, g['score'].values))
        except Exception:
            pass
    avg_auc = float(np.mean(aucs)) if aucs else np.nan

    pos_s = df_n[df_n.label == 'pos']['score_norm'].values
    neg_s = df_n[df_n.label == 'neg']['score_norm'].values
    wdist = (float(wasserstein_distance(pos_s, neg_s))
             if len(pos_s) and len(neg_s) else np.nan)
    return {'gap': float(gap), 'auc': avg_auc, 'wdist': wdist,
            'n_users': df['user'].nunique()}


def split_buckets(train_inter, test_users):
    sparse, medium, dense = set(), set(), set()
    for u in test_users:
        c = len(train_inter.get(u, set()))
        if   SPARSE_THR[0] <= c < SPARSE_THR[1]:    sparse.add(u)
        elif MEDIUM_THR[0] <= c < MEDIUM_THR[1]:    medium.add(u)
        elif c >= MEDIUM_THR[1]:                    dense.add(u)
    return {'sparse': sparse, 'medium': medium, 'dense': dense}


def plot_kde(df_base, df_full, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, tag, df in zip(axes, ['Base', 'Full'], [df_base, df_full]):
        d = normalize_per_user(df)
        sns.kdeplot(data=d[d.label == 'pos'], x='score_norm',
                    label='Positive', fill=True, alpha=0.5, ax=ax, color='#2ecc71')
        sns.kdeplot(data=d[d.label == 'neg'], x='score_norm',
                    label='Negative', fill=True, alpha=0.5, ax=ax, color='#e74c3c')
        gap = (d[d.label == 'pos']['score_norm'].mean() -
               d[d.label == 'neg']['score_norm'].mean())
        ax.set_title(f'{tag}: gap = {gap:.3f}', fontsize=13)
        ax.set_xlabel('Normalized score (per-user z)')
        ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', dpi=200)
    plt.savefig(out_path.replace('.pdf', '.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print(f'  ✓ {out_path}')


def plot_kde_by_bucket(df_base, df_full, buckets, out_path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharex=True, sharey=True)
    for col, bname in enumerate(['sparse', 'medium', 'dense']):
        uset = buckets[bname]
        for row, (tag, df) in enumerate([('Base', df_base), ('Full', df_full)]):
            ax = axes[row, col]
            sub = df[df['user'].isin(uset)]
            if len(sub) == 0:
                ax.set_visible(False)
                continue
            d = normalize_per_user(sub)
            sns.kdeplot(data=d[d.label == 'pos'], x='score_norm',
                        label='Pos', fill=True, alpha=0.5, ax=ax, color='#2ecc71')
            sns.kdeplot(data=d[d.label == 'neg'], x='score_norm',
                        label='Neg', fill=True, alpha=0.5, ax=ax, color='#e74c3c')
            gap = (d[d.label == 'pos']['score_norm'].mean() -
                   d[d.label == 'neg']['score_norm'].mean())
            ax.set_title(f'{tag} | {bname} (gap={gap:.3f})')
            if row == 1: ax.set_xlabel('Score (z-norm)')
            if col == 0: ax.set_ylabel('Density')
            ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', dpi=200)
    plt.savefig(out_path.replace('.pdf', '.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print(f'  ✓ {out_path}')


def main():
    os.makedirs(f'{OUT_DIR}/tables', exist_ok=True)
    os.makedirs(f'{OUT_DIR}/figures', exist_ok=True)

    print('=== Loading base ===')
    base_model, config, train_data, _, test_data, _ = load_model_and_data(
        DATASET, 'base', device=DEVICE)
    n_items = base_model.n_items
    print(f'n_items = {n_items}')

    train_inter = collect_train_inter(base_model)
    test_inter  = collect_test_inter(test_data)
    print(f'#train_users = {len(train_inter)}, #test_users = {len(test_inter)}')

    print('Scoring base ...')
    df_base = collect_scores(base_model, train_inter, test_inter, n_items,
                             n_neg=N_NEG, seed=SEED, device=DEVICE)
    df_base.to_csv(f'{OUT_DIR}/tables/scores_{DATASET}_base.csv', index=False)
    del base_model
    torch.cuda.empty_cache()

    print('\n=== Loading full ===')
    full_model, *_ = load_model_and_data(DATASET, 'full', device=DEVICE)
    print('Scoring full ...')
    df_full = collect_scores(full_model, train_inter, test_inter, n_items,
                             n_neg=N_NEG, seed=SEED, device=DEVICE)
    df_full.to_csv(f'{OUT_DIR}/tables/scores_{DATASET}_full.csv', index=False)
    del full_model
    torch.cuda.empty_cache()

    # ===== 整体 + 分桶 =====
    print('\n========== Metrics ==========')
    rows = []
    for tag, df in [('base', df_base), ('full', df_full)]:
        m = compute_metrics(df)
        m.update({'variant': tag, 'bucket': 'overall'})
        rows.append(m)

    buckets = split_buckets(train_inter, list(test_inter.keys()))
    print(f'Bucket sizes: ' + ', '.join(f'{k}={len(v)}' for k, v in buckets.items()))
    for bname, uset in buckets.items():
        for tag, df in [('base', df_base), ('full', df_full)]:
            m = compute_metrics(df, user_set=uset)
            m.update({'variant': tag, 'bucket': bname})
            rows.append(m)

    dfm = pd.DataFrame(rows)[['bucket', 'variant', 'gap', 'auc', 'wdist', 'n_users']]
    csv = f'{OUT_DIR}/tables/score_dist_metrics_{DATASET}.csv'
    dfm.to_csv(csv, index=False)
    print(f'✓ {csv}')
    print('\n' + dfm.to_string(index=False))

    print('\n========== Improvement (full - base) ==========')
    print(f'{"Bucket":<10}{"gap_b":>10}{"gap_f":>10}{"Δgap":>10}'
          f'{"auc_b":>10}{"auc_f":>10}{"Δauc":>10}')
    for bname in ['overall', 'sparse', 'medium', 'dense']:
        sub = dfm[dfm.bucket == bname]
        if len(sub) < 2:
            continue
        b = sub[sub.variant == 'base'].iloc[0]
        f = sub[sub.variant == 'full'].iloc[0]
        if pd.isna(b.gap) or pd.isna(f.gap):
            continue
        print(f'{bname:<10}{b.gap:>10.4f}{f.gap:>10.4f}{f.gap-b.gap:>+10.4f}'
              f'{b.auc:>10.4f}{f.auc:>10.4f}{f.auc-b.auc:>+10.4f}')

    # ===== 图 =====
    plot_kde(df_base, df_full, f'{OUT_DIR}/figures/score_kde_{DATASET}.pdf')
    plot_kde_by_bucket(df_base, df_full, buckets,
                       f'{OUT_DIR}/figures/score_kde_bybucket_{DATASET}.pdf')


if __name__ == '__main__':
    main()


# python analysis/exp_score_dist.py 2>&1 | tee log/exp_score_dist_clothing.log