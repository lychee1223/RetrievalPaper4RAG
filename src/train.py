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
import dataset, model, loss_functions, metrics
from sentence_transformers import SentenceTransformer
from transformers import AutoModel
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

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
            train_dataset = dataset.TrainDataset(train_query_df, self.paper_df, self.tokenizer, self.args.max_len, batch_size=self.args.batch_size)
            self.train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=dataset.custom_train_collate_fn)
        else:
            train_dataset = dataset.ContrastiveDataset(train_query_df, self.paper_df, self.tokenizer, self.args.max_len)
            self.train_dataloader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True)

        val_dataset = dataset.ContrastiveDataset(valid_query_df, self.paper_df, self.tokenizer, self.args.max_len)      
        self.valid_dataloader = DataLoader(val_dataset, batch_size=self.args.batch_size, shuffle=False)
        
        test_dataset = dataset.ContrastiveDataset(test_query_df, self.paper_df, self.tokenizer, self.args.max_len)
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
        loss_history = {'train': [], 'valid': []}
        if self.args.is_using_my_sampler:
            epochs = 1
        else:
            epochs = self.args.epochs
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0

            for i, batch in enumerate(tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{self.args.epochs}")):
                query_input_ids = batch["query_input_ids"].to(self.device)
                query_attention_mask = batch["query_attention_mask"].to(self.device)
                abst_input_ids = batch["abst_input_ids"].to(self.device)
                abst_attention_mask = batch["abst_attention_mask"].to(self.device)
                category = batch["category"]
                # # Forward
                query_embed = self.model(query_input_ids, query_attention_mask)
                abst_embed = self.model(abst_input_ids, abst_attention_mask)

                # 対照損失を計算
                loss, _, _ = loss_functions.contrastive_loss(query_embed, abst_embed, self.logit_scale, self.device)

                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

                if self.args.is_using_my_sampler:
                    if int(int(len(self.train_dataloader) / int(self.args.batch_size))*self.args.epochs) == int(i + 1):
                        break

                    if int(int(i+1) % (len(self.train_dataloader) / int(self.args.batch_size))) == 0:
                        avg_loss = total_loss / len(self.train_dataloader)
                        loss_history['train'].append(avg_loss)
                        print(f"Train Loss: {avg_loss:.4f}")

                        # 検証フェーズ
                        print(f"エポック : {int(int(i+1) / (len(self.train_dataloader) / int(self.args.batch_size)))}")
                        avg_loss, _, _ = self.evaluate(self.test_dataloader)
                        loss_history['valid'].append(avg_loss)
                        print(f"Valid Loss: {avg_loss:.4f}")
                        self.model.train()
                        total_loss = 0.0
                    

            if self.args.is_using_my_sampler:
                avg_loss = total_loss / (len(self.train_dataloader) % self.args.batch_size)
                loss_history['train'].append(avg_loss)
                print(f"Train Loss: {avg_loss:.4f}")

            else:
                avg_loss = total_loss / len(self.train_dataloader)
                loss_history['train'].append(avg_loss)
                print(f"Train Loss: {avg_loss:.4f}")

            # 検証フェーズ
            avg_loss, _, _ = self.evaluate(self.test_dataloader)
            loss_history['valid'].append(avg_loss)
            print(f"Valid Loss: {avg_loss:.4f}")


        plt.plot(range(1, len(loss_history['train']) + 1), loss_history['train'], label="Train Loss")
        plt.plot(range(1, len(loss_history['valid']) + 1), loss_history['valid'], label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training and Validation Loss")
        
        if self.args.is_using_my_sampler:
            plt.savefig(os.path.join(self.args.save_path, f"loss_original_sampler_{self.args.batch_size}.png"))
            torch.save(self.model.state_dict(), os.path.join(self.args.save_path, f"final_model_original_sampler_{self.args.batch_size}.pth"))
        else:
            plt.savefig(os.path.join(self.args.save_path, f"loss_{self.args.batch_size}.png"))
            torch.save(self.model.state_dict(), os.path.join(self.args.save_path, "final_model_{self.args.batch_size}..pth"))

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
        map_k = {k: 0.0 for k in top_k}
        mrr = 0.0

        # 全ての論文のAbstractを埋め込む
        all_abst_embeds = []
        all_absts = self.paper_df["abstract"].tolist()

        for i in tqdm(range(0, len(all_absts), self.args.batch_size), desc="Abst Eembed"):
            abst = all_absts[i:i + self.args.batch_size]
            abst_enc = self.tokenizer(
                abst, truncation=True, max_length=self.args.max_len, padding="max_length", return_tensors="pt"
            )
            abst_input_ids = abst_enc["input_ids"].to(self.device)
            abst_attention_mask = abst_enc["attention_mask"].to(self.device)
            abst_embeds = self.model(abst_input_ids, abst_attention_mask)
            all_abst_embeds.append(abst_embeds.cpu().numpy())

        all_abst_embeds = np.concatenate(all_abst_embeds, axis=0)

        # 各クエリで論文を検索
        for batch in tqdm(dataloader, desc=f"Test"):
            query_input_ids = batch["query_input_ids"].to(self.device)
            query_attention_mask = batch["query_attention_mask"].to(self.device)
            abst_input_ids = batch["abst_input_ids"].to(self.device)
            abst_attention_mask = batch["abst_attention_mask"].to(self.device)
            categories = batch["category"]

            # Forward
            query_embeds = self.model(query_input_ids, query_attention_mask)
            abst_embeds = self.model(abst_input_ids, abst_attention_mask)

            # 対照損失を計算
            loss, _, _ = loss_functions.contrastive_loss(query_embeds, abst_embeds, self.logit_scale, self.device)
            total_loss += loss.item()

            # sim(query, abst)を計算
            for i, query_embed in enumerate(query_embeds):
                category = categories[i]

                # sim(query, abst)を計算
                sim_query2abst = np.dot(all_abst_embeds, query_embed.cpu().numpy())
                preds = np.argsort(sim_query2abst)[::-1]

                # 同カテゴリの論文をGTとする
                GT = [
                    i for i, c in enumerate(self.paper_df["categories"])
                    if category in c
                ]
                
                # Top-kの評価指標を計算
                for k in top_k:
                    recall_k[k] += metrics.recall_at_k(preds, GT, k)
                    map_k[k] += metrics.map_at_k(preds, GT, k)
                    presicion_k[k] += metrics.precision_at_k(preds, GT, k)
                    ndcg_k[k] += metrics.ndcg_at_k(preds, GT, k)
                
                mrr += metrics.mrr(preds, GT)

        # 評価指標を計算
        for k in top_k:
            recall_k[k] /= len(dataloader.dataset)
            presicion_k[k] /= len(dataloader.dataset)
            map_k[k] /= len(dataloader.dataset)
            ndcg_k[k] /= len(dataloader.dataset)
            print(f"Recall@{k}: {recall_k[k]:.4f}, Precision@{k}: {presicion_k[k]:.4f}, nDCG@{k}: {ndcg_k[k]:.4f}, mAP@{k}: {map_k[k]:.4f}")
        mrr /= len(dataloader.dataset)
        print(f"MRR: {mrr:.4f}")

        avg_loss = total_loss / len(dataloader)

        return avg_loss, recall_k, ndcg_k

    def run(self):
        """
        実験の実行
        """
        print("Starting training...")
        self.train()

        print("Starting testing...")
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
