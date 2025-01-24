import torch.nn as nn
from sentence_transformers import SentenceTransformer
from transformers import AutoModel
import torch
import torch.nn.functional as F

class ContrastiveModel(nn.Module):
    def __init__(self, model_name):
        """
        BERTベースの対照学習モデル
        :param model_name: HuggingFaceモデル名
        """
        super(ContrastiveModel, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        """
        埋め込みを生成
        :param input_ids: トークナイズされた入力
        :param attention_mask: アテンションマスク
        :return: 埋め込みベクトル
        """
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = mean_pooling(output, attention_mask)
        embed = F.normalize(pooled_output, p=2, dim=1)
        return embed


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
