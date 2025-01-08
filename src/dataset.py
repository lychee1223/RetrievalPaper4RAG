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

# コピペ from https://github.com/IyatomiLab/Abst2GA_with_LongCLIP/blob/main/src/custom_dataset.py
class DomainConsistentBatchSampler(Sampler):
    def __init__(self, data_source, subjects, batch_size):
        self.data_source = data_source
        self.subjects = subjects
        self.batch_size = batch_size

        # 研究分野ごとにインデックスを分類
        self.subject2idx = defaultdict(list)
        for idx, subject in enumerate(self.subjects):
            self.subject2idx[subject].append(idx)

    def __iter__(self) -> Iterator[int]:
        # バッチサイズ個分の研究分野から一つずつ取ってくる

        # ランダムな分野をまず1個選択
        # →その分野から1個持ってくる
        # →重複しない分野をバッチサイズ-1個ランダムに選択
        # →TF-IDFが高いやつを1個ずつ持ってくる


        # shuffled_idxs = []
        # for subject, idxs in self.subject2idx.items():
        #     random.shuffle(idxs)
        #     # バッチサイズの整数倍になるように調整
        #     if len(idxs) % self.batch_size != 0:
        #         remainder = self.batch_size - (len(idxs) % self.batch_size)
        #         idxs.extend(random.choices(range(len(self.data_source)), k=remainder))
        #     shuffled_idxs.extend(idxs)
        
        # # シャッフルされたインデックスをバッチごとにグループ化する
        # batches = [
        #     shuffled_idxs[i:i + self.batch_size]
        #     for i in range(0, len(shuffled_idxs), self.batch_size)
        # ]
        # random.shuffle(batches)

        # # インデックスを1つずつ返す
        # all_idx = [i for batch in batches for i in batch]
        # yield from all_idx

    def __len__(self):
        return len(self.data_source)
