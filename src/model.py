import torch.nn as nn
from transformers import BertModel
from sentence_transformers import SentenceTransformer

class ContrastiveModel(nn.Module):
    def __init__(self, model_name):
        """
        BERTベースの対照学習モデル
        :param model_name: HuggingFaceモデル名
        """
        super(ContrastiveModel, self).__init__()
        # self.bert = BertModel.from_pretrained(model_name)
        self.bert = SentenceTransformer(model_name)
        self.linear = nn.Linear(self.bert.config.hidden_size, 256)

    # def forward(self, input_ids, attention_mask):
    def forward(self, sentences):
        """
        埋め込みを生成
        :param input_ids: トークナイズされた入力
        :param attention_mask: アテンションマスク
        :return: 埋め込みベクトル
        """
        
        return self.bert.encode(sentences, convert_to_tensor=True)
        
        # outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # pooled_output = outputs.pooler_output
        # return self.linear(pooled_output)
