import random
from torch.utils.data import Dataset, Sampler
from collections import defaultdict
from typing import Iterator, Tuple
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import torch


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


class TrainDataset(Dataset):
    def __init__(self, queries, papers, tokenizer, max_len, batch_size=64):
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
        self.batch_size = batch_size
        
        self.abstracts = self.papers['abstract'].tolist()
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.abstracts)
        


    def __len__(self):
        return len(self.queries)


    def __getitem__(self, idx)->Tuple[torch.Tensor, torch.Tensor, list]:
        batch_idxes = []
        used_paper_idx = []
        # クエリを取得
        query = self.queries.iloc[idx]["query"]
        category = self.queries.iloc[idx]["category"]


        # ポジティブペーパーをカテゴリに基づいてランダムに選択
        positive_papers = self.papers[self.papers["categories"].apply(lambda x: category in x)]
        positive_paper = positive_papers.sample(1).iloc[0]
        abst = positive_paper["abstract"]
        idx = positive_paper.name
        batch_idxes.append(idx)
        used_paper_idx.append(idx)

        # ネガティブペーパー
        negative_papers = self.papers[self.papers["categories"].apply(lambda x: category not in x)]
        n_idxes = negative_papers.index
        similarities = cosine_similarity(self.tfidf_matrix[idx].reshape(1, -1), self.tfidf_matrix[n_idxes, :])
        sorted_indices = similarities.argsort()[0][::-1]
        for sorted_idx in sorted_indices:            
            if sorted_idx in used_paper_idx:
                continue
            batch_idxes.append(sorted_idx)
            used_paper_idx.append(sorted_idx)
            if len(batch_idxes) >= int(self.batch_size/2):
                break
        while len(batch_idxes)<self.batch_size:
            n_paper_rand = positive_papers.sample(1).iloc[0]
            rand_idx = n_paper_rand.name
            if rand_idx in used_paper_idx:
                continue
            batch_idxes.append(rand_idx)
            used_paper_idx.append(rand_idx)
        
        category_list = []
        query_ids_tensor = torch.empty(0, self.max_len, dtype=torch.long)  # 空のテンソルを初期化
        query_attn_mask_tensor = torch.empty(0, self.max_len, dtype=torch.long) 
        for idx in batch_idxes:
            paper = self.papers.iloc[idx]
            category = random.choice(paper['categories'])
            category_list.append(category)
            query = random.choice(self.queries[self.queries["category"].apply(lambda x: x == category)]['category'].tolist())
            query = query.lower()
            query = " ".join(query.split())
            query_enc = self.tokenizer(
                query, truncation=True, max_length=self.max_len, padding="max_length", return_tensors="pt"
            )
            query_ids_tensor = torch.cat([query_ids_tensor, query_enc["input_ids"].squeeze(0).unsqueeze(0)], dim=0)
            query_attn_mask_tensor = torch.cat([query_attn_mask_tensor, query_enc["attention_mask"].squeeze(0).unsqueeze(0)], dim=0)
        
        abst_ids_tensor = torch.empty(0, self.max_len, dtype=torch.long) 
        abst_attn_mask_tensor = torch.empty(0, self.max_len, dtype=torch.long)
        for idx in batch_idxes:
            paper = self.papers.iloc[idx]
            abst = paper["abstract"]
            abst = abst.lower()
            abst = " ".join(abst.split())
            abst_enc = self.tokenizer(
                abst, truncation=True, max_length=self.max_len, padding="max_length", return_tensors="pt"
            )
            abst_ids_tensor = torch.cat([abst_ids_tensor, abst_enc["input_ids"].squeeze(0).unsqueeze(0)], dim=0)
            abst_attn_mask_tensor = torch.cat([abst_attn_mask_tensor, abst_enc["attention_mask"].squeeze(0).unsqueeze(0)], dim=0)
        
        # breakpoint()
        
        return{
            "query_input_ids": query_ids_tensor,
            "query_attention_mask": query_attn_mask_tensor,
            "abst_input_ids": abst_ids_tensor,
            "abst_attention_mask": abst_attn_mask_tensor,
            "category": category_list
        }



# コピペ from https://github.com/IyatomiLab/Abst2GA_with_LongCLIP/blob/main/src/custom_dataset.py
# class DomainConsistentBatchSampler(Sampler):
#     def __init__(self, data_source, subjects, batch_size):
#         self.data_source = data_source
#         self.subjects = subjects
#         self.batch_size = batch_size

#         # 研究分野ごとにインデックスを分類
#         self.subject2idx = defaultdict(list)
#         for idx, subject in enumerate(self.subjects):
#             self.subject2idx[subject].append(idx)
        
#         # TF-IDFの計算
#         self.abstracts = [paper['abstract'] for paper in self.data_source]
#         self.vectorizer = TfidfVectorizer()
#         self.tfidf_matrix = self.vectorizer.fit_transform(self.abstracts)
        
#     def __iter__(self) -> Iterator[dict]:
#         batch = []
#         used_papers = set()  # 追加：選ばれた論文を追跡

#         while len(used_papers) < len(self.data_source):  # すべての論文を使い切るまで続ける
#             if len(batch) >= self.batch_size:
#                 yield batch  # バッチを返す
#                 batch = []  # バッチをリセット

#             # ランダムにターゲットカテゴリを選択
#             random_subject = random.choice(list(self.subjects))
#             filtered_papers_by_subject = [
#                 idx for idx, paper in enumerate(self.data_source)
#                 if random_subject in paper['categories'] and idx not in used_papers
#             ]
            
#             if filtered_papers_by_subject:
#                 # ランダムにターゲット文書を選択
#                 target_index = random.choice(filtered_papers_by_subject)  # ターゲット文書を選択
#                 batch.append(self.data_source[target_index])  # ターゲット文書をバッチに追加
#                 used_papers.add(target_index)  # 選ばれたターゲット文書を追跡

#                 # 残りの文書に関して、ターゲットとの類似度で文書を選ぶ
#                 filtered_papers_by_subject.remove(target_index)  # ターゲット文書を除外

#                 # 異なる分野から類似度が高い文書を選択
#                 remaining_papers = [
#                     idx for idx, paper in enumerate(self.data_source)
#                     if paper['categories'] != random_subject and idx not in used_papers
#                 ]
                
#                 if remaining_papers:
#                     # ターゲット文書との類似度計算
#                     similarities = cosine_similarity(self.tfidf_matrix[target_index], self.tfidf_matrix[remaining_papers])

#                     # 最も類似度が高い順に文書を選択
#                     sorted_indices = similarities.argsort()[0][::-1]

#                     # sorted_indicesは類似度が高い順にインデックスが並んだ配列
#                     for sorted_idx in sorted_indices:
#                         paper_idx = remaining_papers[sorted_idx]
                        
#                         if paper_idx in used_papers:
#                             continue

#                         batch.append(self.data_source[paper_idx])
#                         used_papers.add(paper_idx)
                        
#                         if len(batch) >= self.batch_size:
#                             break

#             if len(batch) < self.batch_size:
#                 continue

#         if len(batch) > 0:
#             yield batch
class DomainConsistentBatchSampler:
    def __init__(self, data_source: pd.DataFrame, subjects: list, batch_size: int):
        self.data_source = data_source
        self.subjects = subjects
        self.batch_size = batch_size
        self.abstracts = self.data_source['abstract'].tolist()  # 'abstract' 列をリストに変換

        # TF-IDFベクトル化
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.abstracts)

    def __iter__(self) -> Iterator[dict]:
        batch = []
        used_papers = set()  # 追加：選ばれた論文を追跡

        while len(used_papers) < len(self.data_source):  # すべての論文を使い切るまで続ける
            if len(batch) >= self.batch_size:
                yield batch  # バッチを返す
                batch = []  # バッチをリセット

            # ランダムにターゲットカテゴリを選択
            random_subject = random.choice(list(self.subjects))
            
            # ターゲットカテゴリに属する論文を選択
            filtered_papers_by_subject = [
                idx for idx, row in self.data_source.iterrows()
                if random_subject in row['categories'] and idx not in used_papers
            ]
            
            if filtered_papers_by_subject:
                # ランダムにターゲット文書を選択
                target_index = random.choice(filtered_papers_by_subject)  # ターゲット文書を選択
                batch.append(self.data_source.iloc[target_index])  # ターゲット文書をバッチに追加
                used_papers.add(target_index)  # 選ばれたターゲット文書を追跡

                # 残りの文書に関して、ターゲットとの類似度で文書を選ぶ
                filtered_papers_by_subject.remove(target_index)  # ターゲット文書を除外

                # 異なる分野から類似度が高い文書を選択
                remaining_papers = [
                    idx for idx, row in self.data_source.iterrows()
                    if random_subject not in row['categories'] and idx not in used_papers
                ]
                
                if remaining_papers:
                    # ターゲット文書との類似度計算
                    similarities = cosine_similarity(self.tfidf_matrix[target_index].reshape(1, -1), self.tfidf_matrix[remaining_papers])

                    # 最も類似度が高い順に文書を選択
                    sorted_indices = similarities.argsort()[0][::-1]

                    # sorted_indicesは類似度が高い順にインデックスが並んだ配列
                    for sorted_idx in sorted_indices:
                        paper_idx = remaining_papers[sorted_idx]
                        
                        if paper_idx in used_papers:
                            continue

                        batch.append(self.data_source.iloc[paper_idx])  # DataFrameから行を取得
                        used_papers.add(paper_idx)
                        
                        if len(batch) >= self.batch_size:
                            break

            if len(batch) < self.batch_size:
                continue

        if len(batch) > 0:
            yield batch



    def __len__(self):
        return len(self.data_source)

# 例: データをロード
# source_path = "/home/sharp/NIMA/RetrievalPaper4RAG/dataset/paper.json"
# with open(source_path, 'r', encoding='utf-8') as file:
#     data_source = json.load(file)

# # カテゴリの抽出
# subject = set()
# for data in data_source:
#     subject.update(data["categories"])

# # サンプラーのインスタンスを作成
# dc = DomainConsistentBatchSampler(data_source=data_source, subjects=subject, batch_size=64)

# paper_idx = []
# for batch in dc:
#     print(f"バッチの長さ: {len(batch)}")
#     # バッチを処理
#     # 例: バッチ内の各論文IDを表示
#     for data in batch:
#         paper_idx.append(data['id'])
        
# print(len(paper_idx))
# print(len(paper_idx) == len(set(paper_idx)))