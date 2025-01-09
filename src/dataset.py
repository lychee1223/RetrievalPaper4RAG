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
    def __init__(self, queries, papers):
        """
        クエリとポジティブペーパーを返すデータセット
        :param queries: クエリのリスト（query.json）
        :param papers: 論文のリスト（paper.json）
        :param tokenizer: トークナイザ
        :param max_len: トークナイズ後の最大トークン長
        """
        self.queries = queries
        self.papers = papers


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

        return query, abst, category

# class TrainDataset(Dataset):
#     def __init__(self, queries, papers, tokenizer, max_len, batch_size=64):
#         """
#         クエリとポジティブペーパーを返すデータセット
#         :param queries: クエリのリスト（query.json）
#         :param papers: 論文のリスト（paper.json）
#         :param tokenizer: トークナイザ
#         :param max_len: トークナイズ後の最大トークン長
#         """
#         self.queries = queries
#         self.papers = papers
#         self.tokenizer = tokenizer
#         self.max_len = max_len
#         self.batch_size = batch_size
        
#         self.abstracts = self.papers['abstract'].tolist()
#         self.vectorizer = TfidfVectorizer()
#         self.tfidf_matrix = self.vectorizer.fit_transform(self.abstracts)
        


#     def __len__(self):
#         return len(self.queries)


#     def __getitem__(self, idx)->Tuple[torch.Tensor, torch.Tensor, list]:
#         batch_idxes = []
#         used_paper_idx = []
#         # クエリを取得
#         query = self.queries.iloc[idx]["query"]
#         category = self.queries.iloc[idx]["category"]


#         # ポジティブペーパーをカテゴリに基づいてランダムに選択
#         positive_papers = self.papers[self.papers["categories"].apply(lambda x: category in x)]
#         positive_paper = positive_papers.sample(1).iloc[0]
#         abst = positive_paper["abstract"]
#         idx = positive_paper.name
#         batch_idxes.append(idx)
#         used_paper_idx.append(idx)

#         # ネガティブペーパー
#         negative_papers = self.papers[self.papers["categories"].apply(lambda x: category not in x)]
#         n_idxes = negative_papers.index
#         similarities = cosine_similarity(self.tfidf_matrix[idx].reshape(1, -1), self.tfidf_matrix[n_idxes, :])
#         sorted_indices = similarities.argsort()[0][::-1]
#         for sorted_idx in sorted_indices:            
#             if sorted_idx in used_paper_idx:
#                 continue
#             batch_idxes.append(sorted_idx)
#             used_paper_idx.append(sorted_idx)
#             if len(batch_idxes) >= int(self.batch_size/2):
#                 break
#         while len(batch_idxes)<self.batch_size:
#             n_paper_rand = positive_papers.sample(1).iloc[0]
#             rand_idx = n_paper_rand.name
#             if rand_idx in used_paper_idx:
#                 continue
#             batch_idxes.append(rand_idx)
#             used_paper_idx.append(rand_idx)
        
#         category_list = []
#         query_ids_tensor = torch.empty(0, self.max_len, dtype=torch.long)  # 空のテンソルを初期化
#         query_attn_mask_tensor = torch.empty(0, self.max_len, dtype=torch.long) 
#         for idx in batch_idxes:
#             paper = self.papers.iloc[idx]
#             category = random.choice(paper['categories'])
#             category_list.append(category)
#             query = random.choice(self.queries[self.queries["category"].apply(lambda x: x == category)]['category'].tolist())
#             query = query.lower()
#             query = " ".join(query.split())
#             query_enc = self.tokenizer(
#                 query, truncation=True, max_length=self.max_len, padding="max_length", return_tensors="pt"
#             )
#             query_ids_tensor = torch.cat([query_ids_tensor, query_enc["input_ids"].squeeze(0).unsqueeze(0)], dim=0)
#             query_attn_mask_tensor = torch.cat([query_attn_mask_tensor, query_enc["attention_mask"].squeeze(0).unsqueeze(0)], dim=0)
        
#         abst_ids_tensor = torch.empty(0, self.max_len, dtype=torch.long) 
#         abst_attn_mask_tensor = torch.empty(0, self.max_len, dtype=torch.long)
#         for idx in batch_idxes:
#             paper = self.papers.iloc[idx]
#             abst = paper["abstract"]
#             abst = abst.lower()
#             abst = " ".join(abst.split())
#             abst_enc = self.tokenizer(
#                 abst, truncation=True, max_length=self.max_len, padding="max_length", return_tensors="pt"
#             )
#             abst_ids_tensor = torch.cat([abst_ids_tensor, abst_enc["input_ids"].squeeze(0).unsqueeze(0)], dim=0)
#             abst_attn_mask_tensor = torch.cat([abst_attn_mask_tensor, abst_enc["attention_mask"].squeeze(0).unsqueeze(0)], dim=0)
        
#         return{
#             "query_input_ids": query_ids_tensor,
#             "query_attention_mask": query_attn_mask_tensor,
#             "abst_input_ids": abst_ids_tensor,
#             "abst_attention_mask": abst_attn_mask_tensor,
#             "category": category_list
#         }


class TrainDataset(Dataset):
    def __init__(self, queries, papers, batch_size=64):
        """
        クエリとポジティブペーパーを返すデータセット
        :param queries: クエリのリスト（query.json）
        :param papers: 論文のリスト（paper.json）
        :param batch_size: バッチサイズ
        """
        self.queries = queries
        self.papers = papers
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

        # ポジティブペーパーをカテゴリに基づいてランダムに選択
        positive_papers = self.papers[self.papers["categories"].apply(lambda x: p_category in x)]
        positive_paper = positive_papers.sample(1).iloc[0]
        p_abst = positive_paper["abstract"]
        
        # 前処理
        p_query = p_query.lower()
        p_query = " ".join(p_query.split())
        
        p_abst = p_abst.lower()
        p_abst = " ".join(p_abst.split())

        # ネガティブペーパー
        negative_papers = self.papers[self.papers["categories"].apply(lambda x: p_category not in x)]
        p_idx = positive_paper.name
        n_idxes = negative_papers.index
        similarities = cosine_similarity(self.tfidf_matrix[p_idx].reshape(1, -1), self.tfidf_matrix[n_idxes, :])
        sorted_indices = similarities.argsort()[0][::-1]
        for sorted_idx in sorted_indices:            
            if sorted_idx in used_paper_idx:
                continue
            batch_idxes.append(sorted_idx)
            used_paper_idx.append(sorted_idx)
            if len(batch_idxes) >= int(self.batch_size/2):
                break
        while len(batch_idxes) < self.batch_size - 1:
            n_paper_rand = negative_papers.sample(1).iloc[0]
            rand_idx = n_paper_rand.name
            if rand_idx in used_paper_idx:
                continue
            batch_idxes.append(rand_idx)
            used_paper_idx.append(rand_idx)

        category_list = [p_category]
        query_texts = [p_query]
        for i in batch_idxes:
            paper = self.papers.iloc[i]
            n_category = random.choice(paper['categories'])
            category_list.append(n_category)
            n_query = random.choice(self.queries[self.queries["category"].apply(lambda x: x == n_category)]['category'].tolist())
            n_query = n_query.lower()
            n_query = " ".join(n_query.split())
            query_texts.append(n_query)

        abst_texts = [p_abst]
        for i in batch_idxes:
            paper = self.papers.iloc[i]
            n_abst = paper["abstract"]
            n_abst = n_abst.lower()
            n_abst = " ".join(n_abst.split())
            abst_texts.append(n_abst)

        # return query_texts, abst_texts, category_list
        return {
            "query": query_texts,  # 文字列で返す
            "abst": abst_texts,    # 文字列で返す
            "category": category_list
        }


def custom_collate_fn(batch):
    """
    バッチ内のデータをリストとしてまとめるカスタムコレート関数
    """
    query_batch = [item[0] for item in batch]
    abst_batch = [item[1] for item in batch]
    category_batch = [item[2] for item in batch]

    return query_batch, abst_batch, category_batch


def custom_train_collate_fn(batch):
    """
    バッチ内の辞書を適切にまとめるカスタムコレート関数
    """
    # 辞書を作成して、各キーごとにリストにまとめる
    batch_dict = {
        'query': [item['query'] for item in batch],
        'abst': [item['abst'] for item in batch],
        'category': [item['category'] for item in batch],
    }

    return batch_dict
    