import random
from torch.utils.data import Dataset

class ContrastiveDataset(Dataset):
    def __init__(self, queries, papers, tokenizer, max_len):
        """
        クエリとポジティブペーパーを返すデータセット
        :param queries: クエリのリスト（query.json）
        :param papers: 論文のリスト（paper.json）
        :param tokenizer: トークナイザ
        :param max_len: トークナイズ後の最大トークン長
        """
        self.queries = queries
        self.papers = papers
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        # クエリを取得
        query = self.queries.iloc[idx]["query"]
        category = self.queries.iloc[idx]["category"]

        # ポジティブペーパーをカテゴリに基づいてランダムに選択
        positive_papers = self.papers[self.papers["categories"].apply(lambda x: category in x)]
        positive_paper = positive_papers.sample(1).iloc[0]
        abst = positive_paper["abstract"]

        # 正規化
        query = query.lower()
        query = " ".join(query.split())
        abst = abst.lower()
        abst = " ".join(abst.split())

        # トークナイズ
        query_enc = self.tokenizer(
            query, truncation=True, max_length=self.max_len, padding="max_length", return_tensors="pt"
        )
        abst_enc = self.tokenizer(
            abst, truncation=True, max_length=self.max_len, padding="max_length", return_tensors="pt"
        )

        return {
            "query_input_ids": query_enc["input_ids"].squeeze(0),
            "query_attention_mask": query_enc["attention_mask"].squeeze(0),
            "abst_input_ids": abst_enc["input_ids"].squeeze(0),
            "abst_attention_mask": abst_enc["attention_mask"].squeeze(0),
            "category": category
        }
