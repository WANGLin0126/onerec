# analysis/exp_tsne_cluster.py
import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_ckpt import load_model_and_data           # 你自己实现的
from dragon_utils import extract_dragon_embeddings

# ============ 配置 ============
DATASET           = 'sports_sparse'
DEVICE            = 'cpu'
K_LIST            = [10, 20, 50]
N_SEEDS           = 3
EMB_TYPES         = ['item_id', 'item_cf', 'item_fused_proxy',
                     'v_item', 't_item', 'v_homo', 'v_diver']
TSNE_K            = 10
SAMPLE_N          = 5000
PCA_DIM           = 50
TOP_M_PER_CLUSTER = 100        # 每簇只画最接近中心的 M 个点；设为 None 或 <=0 关闭
OUT_DIR           = '/home/wanglin/Projects/onerec/onerec_v2/DRAGON/analysis/results'
# ==============================


def cluster_metrics(emb, K, seed=42):
    # 退化检测: embedding 全零或近常数 -> 返回 NaN
    if emb.std() < 1e-8:
        return {
            'r_bar': np.nan, 'd_bar': np.nan, 'd_over_r': np.nan,
            'sil': np.nan, 'db': np.nan,
            'labels': np.zeros(len(emb), dtype=int),
        }

    km = KMeans(n_clusters=K, random_state=seed, n_init=10).fit(emb)
    labels, centers = km.labels_, km.cluster_centers_

    # 退化检测2: 实际只聚出 < 2 个簇
    if len(np.unique(labels)) < 2:
        return {
            'r_bar': np.nan, 'd_bar': np.nan, 'd_over_r': np.nan,
            'sil': np.nan, 'db': np.nan,
            'labels': labels,
        }

    radii = []
    for k in range(K):
        m = emb[labels == k]
        if len(m) == 0:
            continue
        radii.append(np.linalg.norm(m - centers[k], axis=1).mean())
    avg_r = float(np.mean(radii))

    inter = []
    for i in range(K):
        for j in range(i + 1, K):
            inter.append(np.linalg.norm(centers[i] - centers[j]))
    avg_d = float(np.mean(inter))

    sample_size = min(5000, len(emb))
    sil = float(silhouette_score(emb, labels, sample_size=sample_size, random_state=seed))
    db  = float(davies_bouldin_score(emb, labels))

    return {
        'r_bar': avg_r,
        'd_bar': avg_d,
        'd_over_r': avg_d / (avg_r + 1e-12),
        'sil': sil,
        'db': db,
        'labels': labels,
    }


def metrics_multi_seed(emb, K, n_seeds=3):
    keys = ['r_bar', 'd_bar', 'd_over_r', 'sil', 'db']
    accum = {k: [] for k in keys}
    last_labels = None
    for s in range(n_seeds):
        r = cluster_metrics(emb, K, seed=42 + s)
        for k in keys:
            accum[k].append(r[k])
        last_labels = r['labels']
    out = {k: float(np.mean(v)) for k, v in accum.items()}
    out.update({k + '_std': float(np.std(v)) for k, v in accum.items()})
    out['labels'] = last_labels
    return out


def _keep_topm_per_cluster(emb, labels, m_per_cluster):
    """在原始 embedding 空间里，每个簇保留离其中心最近的 m 个点。

    Returns
    -------
    idx : np.ndarray
        被保留的样本在原 emb 中的下标（升序）。
    """
    if m_per_cluster is None or m_per_cluster <= 0:
        return np.arange(len(emb))

    keep = []
    for k in np.unique(labels):
        idx_k = np.where(labels == k)[0]
        if len(idx_k) == 0:
            continue
        center = emb[idx_k].mean(axis=0, keepdims=True)
        dists = np.linalg.norm(emb[idx_k] - center, axis=1)
        take = min(m_per_cluster, len(idx_k))
        # argpartition: O(n) partial sort
        sel = idx_k[np.argpartition(dists, take - 1)[:take]]
        keep.append(sel)
    if not keep:
        return np.arange(len(emb))
    return np.sort(np.concatenate(keep))


def plot_tsne_compare(b_emb, b_lab, f_emb, f_lab, title_prefix, out_path,
                      sample_n=5000, pca_dim=50, top_m_per_cluster=None):
    b_deg = b_emb.std() < 1e-8
    f_deg = f_emb.std() < 1e-8
    if b_deg or f_deg:
        which = []
        if b_deg: which.append('base')
        if f_deg: which.append('full')
        print(f'  ⚠ skip t-SNE for {title_prefix}: {"+".join(which)} degenerate')
        return

    rng = np.random.RandomState(42)

    def reduce(emb, labels):
        # ① 先按簇筛：每簇只保留离中心最近的 top_m_per_cluster 个
        if top_m_per_cluster is not None and top_m_per_cluster > 0:
            idx = _keep_topm_per_cluster(emb, labels, top_m_per_cluster)
            emb, labels = emb[idx], labels[idx]
            print(f'    kept top-{top_m_per_cluster}/cluster -> {len(emb)} pts')

        # ② 再做随机子采样（防止单簇过大仍然爆 t-SNE）
        n = len(emb)
        if n > sample_n:
            sub = rng.choice(n, sample_n, replace=False)
            emb, labels = emb[sub], labels[sub]

        # ③ PCA + t-SNE
        if emb.shape[1] > pca_dim:
            emb = PCA(n_components=pca_dim, random_state=42).fit_transform(emb)
        proj = TSNE(n_components=2, perplexity=30, init='pca',
                    random_state=42, n_iter=1000).fit_transform(emb)
        return proj, labels

    print('  t-SNE base ...'); pb, lb = reduce(b_emb, b_lab)
    print('  t-SNE full ...'); pf, lf = reduce(f_emb, f_lab)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (proj, lab, name) in zip(
            axes, [(pb, lb, 'Base (w/o ours)'), (pf, lf, 'Full (ours)')]):
        ax.scatter(proj[:, 0], proj[:, 1], c=lab, s=4,
                   cmap='tab20', alpha=0.6)
        ax.set_title(f'{title_prefix} — {name}', fontsize=13)
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', dpi=200)
    # plt.savefig(out_path.replace('.pdf', '.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print(f'  ✓ {out_path}')


def main():
    os.makedirs(f'{OUT_DIR}/tables', exist_ok=True)
    os.makedirs(f'{OUT_DIR}/figures', exist_ok=True)

    print('Loading base ...')
    base_model, *_ = load_model_and_data(DATASET, 'base', device=DEVICE)
    base_embs = extract_dragon_embeddings(base_model)
    del base_model; torch.cuda.empty_cache()

    print('Loading full ...')
    full_model, *_ = load_model_and_data(DATASET, 'full', device=DEVICE)
    full_embs = extract_dragon_embeddings(full_model)
    del full_model; torch.cuda.empty_cache()

    print('\n--- Embedding diagnostics ---')
    print(f'{"name":20s} | {"base std":>10s} | {"full std":>10s} | status')
    for n in EMB_TYPES:
        b_std = base_embs[n].std() if n in base_embs else None
        f_std = full_embs[n].std() if n in full_embs else None
        status = []
        if b_std is None: status.append('base missing')
        elif b_std < 1e-8: status.append('base degenerate')
        if f_std is None: status.append('full missing')
        elif f_std < 1e-8: status.append('full degenerate')
        if not status: status.append('OK')
        print(f'{n:20s} | {b_std if b_std is not None else "N/A":>10} | '
              f'{f_std if f_std is not None else "N/A":>10} | {", ".join(status)}')

    available = [n for n in EMB_TYPES if n in base_embs and n in full_embs]
    print(f'\nAnalyzing: {available}')

    rows = []
    for name in available:
        b, f = base_embs[name], full_embs[name]
        print(f'\n=== {name} ===  base{b.shape} full{f.shape}')
        for K in K_LIST:
            print(f'  K={K}')
            mb = metrics_multi_seed(b, K, N_SEEDS)
            mf = metrics_multi_seed(f, K, N_SEEDS)
            for tag, m in [('base', mb), ('full', mf)]:
                rows.append({
                    'emb': name, 'K': K, 'variant': tag,
                    'r_bar':    round(m['r_bar'], 4),
                    'd_bar':    round(m['d_bar'], 4),
                    'd_over_r': round(m['d_over_r'], 4),
                    'sil':      round(m['sil'], 4),
                    'db':       round(m['db'], 4),
                    'r_bar_std': round(m['r_bar_std'], 4),
                    'sil_std':   round(m['sil_std'], 4),
                })
            if K == TSNE_K and name == 'item_fused_proxy':
                plot_tsne_compare(
                    b, mb['labels'], f, mf['labels'],
                    title_prefix=f'{name} (K={K})',
                    out_path=f'{OUT_DIR}/figures/tsne_{DATASET}_{name}_K{K}_top{TOP_M_PER_CLUSTER}.pdf',
                    sample_n=SAMPLE_N, pca_dim=PCA_DIM,
                    top_m_per_cluster=TOP_M_PER_CLUSTER)

    df = pd.DataFrame(rows)
    csv = f'{OUT_DIR}/tables/cluster_metrics_{DATASET}.csv'
    df.to_csv(csv, index=False)
    print(f'\n✓ {csv}')

    # 打印对比
    print('\n========= Comparison =========')
    for name in available:
        sub = df[df.emb == name]
        print(f'\n--- {name} ---')
        for K in K_LIST:
            b = sub[(sub.K == K) & (sub.variant == 'base')].iloc[0]
            f = sub[(sub.K == K) & (sub.variant == 'full')].iloc[0]
            if pd.isna(b.r_bar) or pd.isna(f.r_bar):
                print(f'  K={K:>2} | (degenerate, skipped)')
                continue
            print(f'  K={K:>2} | '
                  f'r̄ {b.r_bar:.3f}->{f.r_bar:.3f}({(f.r_bar-b.r_bar)/b.r_bar*100:+.1f}%) | '
                  f'd̄/r̄ {b.d_over_r:.3f}->{f.d_over_r:.3f}({(f.d_over_r-b.d_over_r)/b.d_over_r*100:+.1f}%) | '
                  f'Sil {b.sil:+.3f}->{f.sil:+.3f}({f.sil-b.sil:+.3f}) | '
                  f'DB {b.db:.3f}->{f.db:.3f}({(f.db-b.db)/b.db*100:+.1f}%)')


if __name__ == '__main__':
    main()


# python analysis/exp_tsne_cluster.py 2>&1 | tee log/exp_tsne_cluster_clothing.log