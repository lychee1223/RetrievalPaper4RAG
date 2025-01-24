import random
from torch.utils.data import Dataset, Sampler
from collections import defaultdict
from typing import Iterator, Tuple, List, Dict
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import torch


class ContrastiveDataset(Dataset):
    '''
    ベースライン
    '''
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
        encoded_query = self.tokenizer(
            query,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        encoded_abst = self.tokenizer(
            abst,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "query_input_ids": encoded_query["input_ids"].squeeze(0),
            "query_attention_mask": encoded_query["attention_mask"].squeeze(0),
            "abst_input_ids": encoded_abst["input_ids"].squeeze(0),
            "abst_attention_mask": encoded_abst["attention_mask"].squeeze(0),
            "category": category
        }

class TrainDataset(Dataset):
    '''
    提案手法
    '''
    def __init__(self, queries, papers, tokenizer, max_len, batch_size=64):
        """
        クエリとポジティブペーパーを返すデータセット
        :param queries: クエリのリスト（query.json）
        :param papers: 論文のリスト（paper.json）
        :param batch_size: バッチサイズ
        """
        self.queries = queries
        self.papers = papers
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.batch_size = batch_size

        # アブストラクトのTF-IDF行列
        self.abstracts = self.papers['abstract'].tolist()
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.abstracts)

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        batch_idxes = []
        used_paper_idx = []

        # クエリを取得
        p_query = self.queries.iloc[idx]["query"]
        p_category = self.queries.iloc[idx]["category"]

        # ポジティブペア (Abst) を取得
        positive_papers = self.papers[self.papers["categories"].apply(lambda x: p_category in x)]
        positive_paper = positive_papers.sample(1).iloc[0]
        p_abst = positive_paper["abstract"]

        # ネガティブペアとなり得る論文を取得
        negative_papers = self.papers[self.papers["categories"].apply(lambda x: p_category not in x)]
        p_idx = positive_paper.name
        n_idxes = negative_papers.index
        similarities = cosine_similarity(self.tfidf_matrix[p_idx].reshape(1, -1), self.tfidf_matrix[n_idxes, :])
        sorted_indices = similarities.argsort()[0][::-1]

        used_paper_idx.append(p_idx)

        # ハードネガティブ (Abst) を取得
        for sorted_idx in sorted_indices:            
            if sorted_idx in used_paper_idx:
                continue
            batch_idxes.append(sorted_idx)
            used_paper_idx.append(sorted_idx)
            if len(batch_idxes) >= int(self.batch_size/2):
                break

        # イージーネガティブ (Abst) を取得
        while len(batch_idxes) < self.batch_size - 1:
            n_paper_rand = negative_papers.sample(1).iloc[0]
            rand_idx = n_paper_rand.name
            if rand_idx in used_paper_idx:
                continue
            batch_idxes.append(rand_idx)
            used_paper_idx.append(rand_idx)

        # queryを順にエンコード
        encoded_query = torch.empty(0, dtype=torch.long)
        encoded_abst = torch.empty(0, dtype=torch.long)

        query_ids_tensor = torch.empty(0, dtype=torch.long)
        query_attn_mask_tensor = torch.empty(0, dtype=torch.long)
        abst_ids_tensor = torch.empty(0, dtype=torch.long)
        abst_attn_mask_tensor = torch.empty(0, dtype=torch.long)
        category_list = [p_category]

        # query_texts = [p_query]

        # ポジティブペア(query)を取得・埋め込み
        p_query = p_query.lower()
        p_query = " ".join(p_query.split())
        encoded_p_query = self.tokenizer(
            p_query, truncation=True, max_length=self.max_len, padding="max_length", return_tensors="pt"
        )
        query_ids_tensor = torch.cat([query_ids_tensor, encoded_p_query["input_ids"].squeeze(0).unsqueeze(0)], dim=0)
        query_attn_mask_tensor = torch.cat([query_attn_mask_tensor, encoded_p_query["attention_mask"].squeeze(0).unsqueeze(0)], dim=0)
        
        # ネガティブペア(query)を取得・埋め込み
        for i in batch_idxes:
            paper = self.papers.iloc[i]
            n_category = random.choice(paper['categories'])
            category_list.append(n_category)

            n_query = random.choice(self.queries[self.queries["category"].apply(lambda x: x == n_category)]['category'].tolist())
            n_query = n_query.lower()
            n_query = " ".join(n_query.split())

            encoded_n_query = self.tokenizer(
                n_query, truncation=True, max_length=self.max_len, padding="max_length", return_tensors="pt"
            )
            encoded_query = torch.cat([encoded_query, encoded_n_query["input_ids"].squeeze(0)], dim=0)
            query_ids_tensor = torch.cat([query_ids_tensor, encoded_n_query["input_ids"].squeeze(0).unsqueeze(0)], dim=0)
            query_attn_mask_tensor = torch.cat([query_attn_mask_tensor, encoded_n_query["attention_mask"].squeeze(0).unsqueeze(0)], dim=0)

            # query_texts.append(n_query)

        # abst_texts = [p_abst]

        # ポジティブペア(Abst)を取得・埋め込み
        p_abst = p_abst.lower()
        p_abst = " ".join(p_abst.split())
        encoded_p_abst = self.tokenizer(
            p_abst, truncation=True, max_length=self.max_len, padding="max_length", return_tensors="pt"
        )
        abst_ids_tensor = torch.cat([abst_ids_tensor, encoded_p_abst["input_ids"].squeeze(0).unsqueeze(0)], dim=0)
        abst_attn_mask_tensor = torch.cat([abst_attn_mask_tensor, encoded_p_abst["attention_mask"].squeeze(0).unsqueeze(0)], dim=0)

        # ネガティブペア(Abst)を取得・埋め込み 
        for i in batch_idxes:
            paper = self.papers.iloc[i]
            n_abst = paper["abstract"]
            n_abst = n_abst.lower()
            n_abst = " ".join(n_abst.split())
            encoded_n_abst = self.tokenizer(
                n_abst, truncation=True, max_length=self.max_len, padding="max_length", return_tensors="pt"
            )
            encoded_abst = torch.cat([encoded_abst, encoded_n_abst["input_ids"].squeeze(0)], dim=0)
            abst_ids_tensor = torch.cat([abst_ids_tensor, encoded_n_abst["input_ids"].squeeze(0).unsqueeze(0)], dim=0)
            abst_attn_mask_tensor = torch.cat([abst_attn_mask_tensor, encoded_n_abst["attention_mask"].squeeze(0).unsqueeze(0)], dim=0)

        return{
            "query_input_ids": query_ids_tensor,
            "query_attention_mask": query_attn_mask_tensor,
            "abst_input_ids": abst_ids_tensor,
            "abst_attention_mask": abst_attn_mask_tensor,
            "category": category_list
        }

def custom_train_collate_fn(batch):
    batch_dict = {
        "query_input_ids": torch.stack([item["query_input_ids"] for item in batch]).squeeze(0),
        "query_attention_mask": torch.stack([item["query_attention_mask"] for item in batch]).squeeze(0),
        "abst_input_ids": torch.stack([item["abst_input_ids"] for item in batch]).squeeze(0),
        "abst_attention_mask": torch.stack([item["abst_attention_mask"] for item in batch]).squeeze(0),
        "category": [category for item in batch for category in item["category"]]
    }

    return batch_dict
