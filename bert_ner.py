#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle
import os

class BERTNER(nn.Module):
    """基于BERT的命名实体识别模型"""
    
    def __init__(self, bert_model_name='bert-base-chinese', num_classes=9, dropout=0.1):
        super(BERTNER, self).__init__()
        
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
        # 状态标签
        self.states = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-GPE', 'I-GPE']
        self.state_to_idx = {state: idx for idx, state in enumerate(self.states)}
        self.idx_to_state = {idx: state for state, idx in self.state_to_idx.items()}
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        # BERT前向传播
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # 获取序列输出
        sequence_output = outputs[0]  # [batch_size, seq_len, hidden_size]
        
        # 应用dropout
        sequence_output = self.dropout(sequence_output)
        
        # 分类层
        logits = self.classifier(sequence_output)  # [batch_size, seq_len, num_classes]
        
        return logits

class BERTNERDataset(Dataset):
    """BERT NER数据集类"""
    
    def __init__(self, sequences, labels, tokenizer, max_len=128):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # 状态标签映射
        self.states = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-GPE', 'I-GPE']
        self.state_to_idx = {state: idx for idx, state in enumerate(self.states)}
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        labels = self.labels[idx]
        
        # 将字符序列转换为字符串
        text = ''.join(sequence)
        
        # 使用BERT tokenizer
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        
        # 处理标签
        label_ids = []
        for i, label in enumerate(labels):
            if i >= self.max_len - 2:  # 考虑[CLS]和[SEP]
                break
            label_ids.append(self.state_to_idx.get(label, 0))
        
        # 添加特殊token的标签
        label_ids = [0] + label_ids + [0]  # [CLS]和[SEP]的标签
        
        # 填充标签
        while len(label_ids) < self.max_len:
            label_ids.append(0)
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'labels': torch.tensor(label_ids, dtype=torch.long)
        }

class BERTNERModel:
    """BERT NER模型包装类"""
    
    def __init__(self, bert_model_name='bert-base-chinese', learning_rate=2e-5, 
                 warmup_steps=0, weight_decay=0.01):
        self.bert_model_name = bert_model_name
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        
        self.states = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-GPE', 'I-GPE']
        self.state_to_idx = {state: idx for idx, state in enumerate(self.states)}
        self.idx_to_state = {idx: state for state, idx in self.state_to_idx.items()}
        
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_data(self, file_path):
        """加载数据并转换为序列标注格式"""
        print("正在加载数据...")
        
        sequences = []
        labels = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="处理数据"):
                data = json.loads(line.strip())
                text = data['text']
                entities = data['entities']
                
                # 将文本转换为字符序列
                char_seq = list(text)
                label_seq = ['O'] * len(char_seq)
                
                # 标注实体
                for entity in entities:
                    start_idx = entity['start_idx']
                    end_idx = entity['end_idx']
                    entity_label = entity['entity_label']
                    
                    # 转换为BIO标注
                    if start_idx < len(char_seq):
                        label_seq[start_idx] = f'B-{entity_label}'
                        for i in range(start_idx + 1, min(end_idx, len(char_seq))):
                            label_seq[i] = f'I-{entity_label}'
                
                sequences.append(char_seq)
                labels.append(label_seq)
        
        return sequences, labels
    
    def create_data_loaders(self, sequences, labels, batch_size=16, max_len=128):
        """创建数据加载器"""
        # 初始化tokenizer
        if self.tokenizer is None:
            self.tokenizer = BertTokenizer.from_pretrained(self.bert_model_name)
        
        # 划分数据集
        train_seqs, test_seqs, train_labels, test_labels = train_test_split(
            sequences, labels, test_size=0.2, random_state=42
        )
        
        # 创建数据集
        train_dataset = BERTNERDataset(train_seqs, train_labels, self.tokenizer, max_len)
        test_dataset = BERTNERDataset(test_seqs, test_labels, self.tokenizer, max_len)
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, test_loader, train_seqs, test_seqs, train_labels, test_labels
    
    def train(self, sequences, labels, epochs=3, batch_size=16, max_len=128):
        """训练BERT模型"""
        print("正在训练BERT模型...")
        
        # 创建数据加载器
        train_loader, test_loader, train_seqs, test_seqs, train_labels, test_labels = \
            self.create_data_loaders(sequences, labels, batch_size, max_len)
        
        # 初始化模型
        self.model = BERTNER(
            bert_model_name=self.bert_model_name,
            num_classes=len(self.states),
            dropout=0.1
        ).to(self.device)
        
        # 优化器和调度器
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps
        )
        
        # 损失函数
        criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略PAD标签
        
        # 训练循环
        train_losses = []
        test_f1_scores = []
        
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            total_loss = 0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # 前向传播
                logits = self.model(input_ids, attention_mask, token_type_ids)
                
                # 计算损失
                loss = criterion(logits.view(-1, len(self.states)), labels.view(-1))
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            train_losses.append(avg_loss)
            
            # 评估阶段
            f1_score = self.evaluate_model(test_loader)
            test_f1_scores.append(f1_score)
            
            print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, F1 = {f1_score:.4f}")
        
        print("BERT模型训练完成!")
        
        # 保存训练历史
        self.training_history = {
            'train_losses': train_losses,
            'test_f1_scores': test_f1_scores
        }
        
        return train_losses, test_f1_scores
    
    def evaluate_model(self, test_loader):
        """评估模型"""
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                logits = self.model(input_ids, attention_mask, token_type_ids)
                predictions = torch.argmax(logits, dim=-1)
                
                # 收集预测和标签（忽略特殊token）
                for i in range(len(predictions)):
                    pred_seq = predictions[i].cpu().numpy()
                    label_seq = labels[i].cpu().numpy()
                    mask = label_seq != 0  # 忽略PAD标签
                    
                    all_predictions.extend(pred_seq[mask])
                    all_labels.extend(label_seq[mask])
        
        # 计算F1分数
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        return f1
    
    def predict(self, sequences):
        """预测多个序列"""
        if self.model is None:
            raise ValueError("模型未训练")
        
        self.model.eval()
        predictions = []
        
        for seq in tqdm(sequences, desc="预测"):
            # 将字符序列转换为字符串
            text = ''.join(seq)
            
            # 使用BERT tokenizer
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=128,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            token_type_ids = encoding['token_type_ids'].to(self.device)
            
            with torch.no_grad():
                logits = self.model(input_ids, attention_mask, token_type_ids)
                pred_indices = torch.argmax(logits, dim=-1)[0].cpu().numpy()
            
            # 转换为标签（跳过特殊token）
            pred_labels = []
            for i, idx in enumerate(pred_indices[1:-1]):  # 跳过[CLS]和[SEP]
                if i < len(seq):
                    pred_labels.append(self.idx_to_state[idx])
            
            predictions.append(pred_labels)
        
        return predictions
    
    def predict_single(self, sequence):
        """预测单个序列"""
        return self.predict([sequence])[0]
    
    def evaluate(self, true_labels, pred_labels):
        """评估模型性能"""
        print("正在评估BERT模型...")
        
        # 展平标签
        flat_true = []
        flat_pred = []
        
        for true_seq, pred_seq in zip(true_labels, pred_labels):
            min_len = min(len(true_seq), len(pred_seq))
            flat_true.extend(true_seq[:min_len])
            flat_pred.extend(pred_seq[:min_len])
        
        # 计算分类报告
        report = classification_report(flat_true, flat_pred, 
                                     target_names=self.states, 
                                     zero_division=0)
        
        # 计算F1分数
        f1_score_val = f1_score(flat_true, flat_pred, average='weighted')
        
        print("分类报告:")
        print(report)
        print(f"加权F1分数: {f1_score_val:.4f}")
        
        return report, f1_score_val
    
    def save_model(self, file_path):
        """保存模型"""
        model_data = {
            'model_state_dict': self.model.state_dict(),
            'tokenizer': self.tokenizer,
            'states': self.states,
            'state_to_idx': self.state_to_idx,
            'idx_to_state': self.idx_to_state,
            'model_config': {
                'bert_model_name': self.bert_model_name,
                'learning_rate': self.learning_rate,
                'warmup_steps': self.warmup_steps,
                'weight_decay': self.weight_decay
            }
        }
        
        torch.save(model_data, file_path)
        print(f"模型已保存到: {file_path}")
    
    def load_model(self, file_path):
        """加载模型"""
        model_data = torch.load(file_path, map_location=self.device)
        
        # 恢复配置
        config = model_data['model_config']
        self.bert_model_name = config['bert_model_name']
        self.learning_rate = config['learning_rate']
        self.warmup_steps = config['warmup_steps']
        self.weight_decay = config['weight_decay']
        
        # 恢复tokenizer和标签
        self.tokenizer = model_data['tokenizer']
        self.states = model_data['states']
        self.state_to_idx = model_data['state_to_idx']
        self.idx_to_state = model_data['idx_to_state']
        
        # 重建模型
        self.model = BERTNER(
            bert_model_name=self.bert_model_name,
            num_classes=len(self.states),
            dropout=0.1
        ).to(self.device)
        
        # 加载权重
        self.model.load_state_dict(model_data['model_state_dict'])
        
        print(f"模型已从 {file_path} 加载")

def main():
    """主函数"""
    # 初始化模型
    model = BERTNERModel(
        bert_model_name='bert-base-chinese',
        learning_rate=2e-5,
        warmup_steps=0,
        weight_decay=0.01
    )
    
    # 加载数据
    sequences, labels = model.load_data('data/ccfbdci.jsonl')
    
    # 训练模型
    train_losses, test_f1_scores = model.train(sequences, labels, epochs=3)
    
    # 创建测试数据加载器进行评估
    _, test_loader, _, test_seqs, _, test_labels = model.create_data_loaders(
        sequences, labels, batch_size=16
    )
    
    # 评估模型
    f1_score = model.evaluate_model(test_loader)
    print(f"最终F1分数: {f1_score:.4f}")
    
    # 预测测试集
    pred_labels = model.predict(test_seqs)
    
    # 详细评估
    model.evaluate(test_labels, pred_labels)
    
    # 保存模型
    model.save_model('models/bert_ner_model.pth')
    
    print("BERT NER模型训练和评估完成!")

if __name__ == "__main__":
    main() 