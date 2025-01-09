import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from tqdm import tqdm
import json
import argparse
import pandas as pd
import numpy as np
import os
import random
import dataset, model, loss_functions, metrix
from sentence_transformers import SentenceTransformer

class Experiment:
    def __init__(self, args):
        """
        実験の初期化
        """
        self.args = args
        self.device = torch.device(f'cuda:{self.args.device}' if torch.cuda.is_available() else 'cpu')

        # モデルと最適化
        self.model = model.ContrastiveModel(self.args.model_name).to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained(self.args.model_name)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        self.logit_scale = np.log(1 / self.args.temperature)
        self.best_valid_loss = float('inf')

        # データセットの準備
        self.paper_df = self.load_data(self.args.paper_path)
        train_query_df = self.load_data(self.args.train_query_path)
        valid_query_df = self.load_data(self.args.valid_query_path)
        test_query_df = self.load_data(self.args.test_query_path)

        train_query_df = train_query_df.explode("query").reset_index(drop=True)
        valid_query_df = valid_query_df.explode("query").reset_index(drop=True)
        test_query_df = test_query_df.explode("query").reset_index(drop=True)
        
        if self.args.is_using_my_sampler:
            train_dataset = dataset.TrainDataset(train_query_df, self.paper_df, batch_size=self.args.batch_size)
            self.train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=dataset.custom_train_collate_fn)
        else:
            train_dataset = dataset.ContrastiveDataset(train_query_df, self.paper_df)
            self.train_dataloader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, collate_fn=dataset.custom_collate_fn)

        val_dataset = dataset.ContrastiveDataset(valid_query_df, self.paper_df)         
        self.valid_dataloader = DataLoader(val_dataset, batch_size=self.args.batch_size, shuffle=False)
        
        test_dataset = dataset.ContrastiveDataset(test_query_df, self.paper_df) 
        self.test_dataloader = DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False)

        # 保存先のディレクトリを作成
        os.makedirs(self.args.save_path, exist_ok=True)

    def load_data(self, path):
        """
        データをロードする
        :param path: JSONデータのパス
        :return: ロードしたデータ
        """
        with open(path, "r") as f:
            return pd.DataFrame(json.load(f))

    def train(self):
        """
        学習フェーズ
        """
        for epoch in range(self.args.epochs):
            self.model.train()
            total_loss = 0.0

            for batch in tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{self.args.epochs}"):
                # breakpoint()
                if self.args.is_using_my_sampler:
                    query = batch["query"]
                    abst = batch["abst"]
                    category = batch["category"]
                else:
                    query = batch[0]
                    abst = batch[1]
                    category = batch[2] 
                # Forward
                # query_embed = self.model(query_input_ids, query_attention_mask)
                # abst_embed = self.model(abst_input_ids, abst_attention_mask)
                query_embed = self.model(query, self.device).to(self.device)
                abst_embed = self.model(abst, self.device).to(self.device)

                # 対照損失を計算
                loss, _, _ = loss_functions.contrastive_loss(query_embed, abst_embed, self.logit_scale, self.device)

                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(self.train_dataloader)
            (f"Train Loss: {avg_loss:.4f}")

            # 検証フェーズ
            loss, _, _ = self.evaluate(self.valid_dataloader)
            print(f"Valid Loss: {loss:.4f}")
        
            if loss < self.best_valid_loss:
                self.best_valid_loss = loss
                os.makedirs(self.args.save_path, exist_ok=True)
                torch.save(self.model.state_dict(), os.path.join(self.args.save_path, "best_model.pth"))
        
        torch.save(self.model.state_dict(), os.path.join(self.args.save_path, "final_model.pth"))

    @torch.no_grad()
    def evaluate(self, dataloader):
        """
        検証またはテストフェーズ
        :param dataloader: DataLoader
        :return: 損失, 評価指標
        """
        self.model.eval()

        total_loss = 0.0

        top_k = [10, 50, 100, 200, 500, 1000, len(self.paper_df)]
        recall_k = {k: 0.0 for k in top_k}
        presicion_k = {k: 0.0 for k in top_k}
        ndcg_k = {k: 0.0 for k in top_k}
        map = 0.0
        mrr = 0.0

        # 全ての論文のAbstractを埋め込む
        abst_embeds = []
        batch_size = self.args.batch_size
        absts = self.paper_df["abstract"].tolist()

        for i in tqdm(range(0, len(absts), batch_size), desc="Abst Eembed"):
            abst = absts[i:i + batch_size]
            abst_embed = self.model(abst, self.device)
            abst_embeds.append(abst_embed.cpu().numpy())

        abst_embeds = np.concatenate(abst_embeds, axis=0)

        # 各クエリで論文を検索
        for batch in tqdm(dataloader, desc=f"Test"):

            query = batch[0]
            abst = batch[1]
            categories = batch[2] 

            # Forward
            # query_embed = self.model(query_input_ids, query_attention_mask)
            # abst_embed = self.model(abst_input_ids, abst_attention_mask)
            query_embeds = self.model(query, self.device).to(self.device)
            abst_embed = self.model(abst, self.device).to(self.device)

            # 対照損失を計算
            loss, _, _ = loss_functions.contrastive_loss(query_embeds, abst_embed, self.logit_scale, self.device)
            total_loss += loss.item()

            # sim(query, abst)を計算
            for i, query_embed in enumerate(query_embeds):
                category = categories[i]

                # sim(query, abst)を計算
                sim_query2abst = np.dot(abst_embeds, query_embed.cpu().numpy())
                preds = np.argsort(sim_query2abst)[::-1]

                # 同カテゴリの論文をGTとする
                GT = [
                    i for i, c in enumerate(self.paper_df["categories"])
                    if category in c
                ]
                
                # Top-kの評価指標を計算
                for k in top_k:
                    recall_k[k] += metrix.recall_at_k(preds, GT, k)
                    presicion_k[k] += metrix.precision_at_k(preds, GT, k)
                    ndcg_k[k] += metrix.ndcg_at_k(preds, GT, k)
                
                map += metrix.map(preds, GT)
                mrr += metrix.mrr(preds, GT)

        # 評価指標を計算
        for k in top_k:
            recall_k[k] /= len(dataloader.dataset)
            presicion_k[k] /= len(dataloader.dataset)
            ndcg_k[k] /= len(dataloader.dataset)
            print(f"Recall@{k}: {recall_k[k]:.4f}, Precision@{k}: {presicion_k[k]:.4f}, nDCG@{k}: {ndcg_k[k]:.4f}")
        map /= len(dataloader.dataset)
        mrr /= len(dataloader.dataset)
        print(f"mAP: {map:.4f}, MRR: {mrr:.4f}")

        avg_loss = total_loss / len(dataloader)

        return avg_loss, recall_k, ndcg_k

    def run(self):
        """
        実験の実行
        """
        print("Starting training...")
        self.train()
        print("Starting testing...")
        # # 重みをロードしてテストする場合は以下のコメントアウトを外す
        # self.model = model.ContrastiveModel(self.args.model_name).to(self.device)
        # self.model.load_state_dict(torch.load(os.path.join(self.args.save_path, "best_model.pth")))
        loss, _, _ = self.evaluate(self.test_dataloader)
        print(f"Test Loss: {loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Contrastive Learning Training, Validation, and Test")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--device", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--paper_path", type=str, required=True, help="Path to paper dataset JSON")
    parser.add_argument("--train_query_path", type=str, required=True, help="Path to train dataset JSON")
    parser.add_argument("--valid_query_path", type=str, required=True, help="Path to validation dataset JSON")
    parser.add_argument("--test_query_path", type=str, required=True, help="Path to test dataset JSON")
    parser.add_argument("--save_path", type=str, default="checkpoint/", help="Path to save the model")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="HuggingFace model name")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--max_len", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--temperature", type=float, default=0.07, help="Temperature parameter for contrastive loss")
    parser.add_argument("--is_using_my_sampler", action='store_true', default=False, help="Whether to use my sampler")
    args = parser.parse_args()

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    experiment = Experiment(args)
    experiment.run()
