import torch
import torch.nn.functional as F

def contrastive_loss(
    query_embed: torch.Tensor,
    abst_embed: torch.Tensor,
    logit_scale: float,
    device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    '''
    対照損失を計算する
    :param query_embed: クエリの埋め込みベクトル
    :param abst_embed: Abstの埋め込みベクトル
    :param logit_scale: スケーリング係数
    :param device: デバイス
    :return: 対照損失, クエリからAbatへの類似度行列, Abstからクエリへの類似度行列
    '''
    # 正規化
    normalized_query_embed = query_embed / query_embed.norm(dim=1, keepdim=True)
    normalized_abst_embed = abst_embed / abst_embed.norm(dim=1, keepdim=True)

    # 類似度行列を計算
    sim_query2abst = torch.matmul(normalized_query_embed, normalized_abst_embed.T)
    sim_abst2query = torch.matmul(normalized_abst_embed, normalized_query_embed.T)
    
    # スケーリング
    sim_query2abst = torch.nn.Parameter(torch.ones([]) * logit_scale).exp() * sim_query2abst
    sim_abst2query = torch.nn.Parameter(torch.ones([]) * logit_scale).exp() * sim_abst2query

    # 正解ラベルを生成
    targets = torch.arange(abst_embed.size(0), dtype=torch.long).to(device)
    
    # lossを計算
    inter_loss = (F.cross_entropy(sim_query2abst, targets) + F.cross_entropy(sim_abst2query, targets)) / 2
    
    return inter_loss, sim_query2abst, sim_abst2query
