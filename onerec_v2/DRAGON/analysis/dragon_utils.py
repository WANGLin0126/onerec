# analysis/dragon_utils.py
import torch


@torch.no_grad()
def warmup_dragon(model, n_items_to_use=2):
    """
    DRAGON 加载 checkpoint 后:
      - self.masked_adj 是 None
      - self.v_item_rep / v_homo / v_diver / t_* 这些都不存在
    必须先跑一次 pre_epoch_processing + forward_v2 来"激活"它们。
    """
    model.eval()
    
    # 1) 设置 masked_adj 等 (eval 时用 norm_adj)
    if model.masked_adj is None:
        model.masked_adj = model.norm_adj
    
    # 2) topk_sample 需要的 user_weight_matrix
    if not hasattr(model, 'user_weight_matrix') or model.user_weight_matrix is None:
        model.epoch_user_graph, model.user_weight_matrix = model.topk_sample(model.k)
        model.user_weight_matrix = model.user_weight_matrix.to(model.device)
    
    # 3) 跑一次 forward_v2,塞 dummy interaction
    dummy = [
        torch.LongTensor([0]).to(model.device),                       # user
        torch.LongTensor([0]).to(model.device),                       # pos
        torch.LongTensor([min(1, model.n_items - 1)]).to(model.device) # neg
    ]
    _ = model.forward_v2(dummy)


@torch.no_grad()
def extract_dragon_embeddings(model):
    """
    返回 dict: {name: ndarray [n_items, d]}
    """
    warmup_dragon(model)
    embs = {}

    # --- 静态 (state_dict 里的) ---
    embs['item_id'] = model.item_id_embedding.weight.detach().cpu().numpy()
    
    if model.v_feat is not None:
        embs['item_v_raw'] = model.image_embedding.weight.detach().cpu().numpy()
    if model.t_feat is not None:
        embs['item_t_raw'] = model.text_embedding.weight.detach().cpu().numpy()

    # --- CF 主干 forward 出的 item rep (= i_g_embeddings + h) ---
    u_g, i_g = model.forward(model.norm_adj)
    embs['item_cf'] = i_g.detach().cpu().numpy()  # 这是 LightGCN+mm_adj 后的 item rep

    # --- forward_v2 里塞进 model 的临时属性 ---
    if hasattr(model, 'v_item_rep'):
        embs['v_item'] = model.v_item_rep.detach().cpu().numpy()
    if hasattr(model, 't_item'):
        pass
    if hasattr(model, 't_item_rep'):
        embs['t_item'] = model.t_item_rep.detach().cpu().numpy()

    # 同质 / 多样性 (item 维度: [n_items, d])
    if hasattr(model, 'v_homo') and isinstance(model.v_homo, torch.Tensor):
        embs['v_homo'] = model.v_homo.detach().cpu().numpy()
    if hasattr(model, 'v_diver') and isinstance(model.v_diver, torch.Tensor):
        embs['v_diver'] = model.v_diver.detach().cpu().numpy()
    if hasattr(model, 't_homo') and isinstance(model.t_homo, torch.Tensor):
        embs['t_homo'] = model.t_homo.detach().cpu().numpy()
    if hasattr(model, 't_diver') and isinstance(model.t_diver, torch.Tensor):
        embs['t_diver'] = model.t_diver.detach().cpu().numpy()

    # --- "完整 fused" 用 CF + (v_item + t_item)/2 作为代理 (因为 QKV 是 user-specific) ---
    fused = embs['item_cf'].copy()
    if 'v_item' in embs and 't_item' in embs:
        fused = fused + 0.5 * (embs['v_item'] + embs['t_item'])
    embs['item_fused_proxy'] = fused

    return embs