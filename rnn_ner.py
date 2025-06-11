#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle
import os

class RNNNER(nn.Module):
    """基于RNN的命名实体识别模型"""
    
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2, 
                 num_classes=9, dropout=0.5, bidirectional=True):
        super(RNNNER, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # RNN层
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # 输出层
        rnn_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(rnn_output_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        
        # 状态标签
        self.states = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-GPE', 'I-GPE']
        self.state_to_idx = {state: idx for idx, state in enumerate(self.states)}
        self.idx_to_state = {idx: state for state, idx in self.state_to_idx.items()}
        
    def forward(self, x, lengths):
        # x: [batch_size, seq_len]
        batch_size, seq_len = x.size()
        
        # 嵌入
        embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        
        # 打包序列
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True)
        
        # RNN前向传播
        output, hidden = self.rnn(packed)
        
        # 解包序列
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        
        # 应用dropout
        output = self.dropout(output)
        
        # 全连接层
        logits = self.fc(output)  # [batch_size, seq_len, num_classes]
        
        return logits

class NERDataset(Dataset):
    """NER数据集类"""
    
    def __init__(self, sequences, labels, word_to_idx, max_len=100):
        self.sequences = sequences
        self.labels = labels
        self.word_to_idx = word_to_idx
        self.max_len = max_len
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        labels = self.labels[idx]
        
        # 截断或填充序列
        if len(sequence) > self.max_len:
            sequence = sequence[:self.max_len]
            labels = labels[:self.max_len]
        
        # 转换为索引
        seq_indices = [self.word_to_idx.get(char, self.word_to_idx['<UNK>']) for char in sequence]
        label_indices = [self.state_to_idx.get(label, 0) for label in labels]
        
        # 填充
        seq_len = len(seq_indices)
        while len(seq_indices) < self.max_len:
            seq_indices.append(0)  # PAD token
            label_indices.append(0)  # PAD label
        
        return {
            'input_ids': torch.tensor(seq_indices, dtype=torch.long),
            'labels': torch.tensor(label_indices, dtype=torch.long),
            'length': seq_len
        }

class RNNNERModel:
    """RNN NER模型包装类"""
    
    def __init__(self, embedding_dim=128, hidden_dim=256, num_layers=2, 
                 dropout=0.5, bidirectional=True, learning_rate=0.001):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.learning_rate = learning_rate
        
        self.states = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-GPE', 'I-GPE']
        self.state_to_idx = {state: idx for idx, state in enumerate(self.states)}
        self.idx_to_state = {idx: state for state, idx in self.state_to_idx.items()}
        
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.word_count = 0
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
    
    def build_vocabulary(self, sequences, min_freq=2):
        """构建词汇表"""
        print("正在构建词汇表...")
        
        word_freq = Counter()
        for seq in sequences:
            word_freq.update(seq)
        
        # 过滤低频词
        valid_words = {word for word, freq in word_freq.items() if freq >= min_freq}
        valid_words.add('<UNK>')  # 未知词标记
        valid_words.add('<PAD>')  # 填充标记
        
        self.word_to_idx = {word: idx for idx, word in enumerate(valid_words)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        self.word_count = len(self.word_to_idx)
        
        print(f"词汇表大小: {self.word_count}")
    
    def create_data_loaders(self, sequences, labels, batch_size=32, max_len=100):
        """创建数据加载器"""
        # 划分数据集
        train_seqs, test_seqs, train_labels, test_labels = train_test_split(
            sequences, labels, test_size=0.2, random_state=42
        )
        
        # 创建数据集
        train_dataset = NERDataset(train_seqs, train_labels, self.word_to_idx, max_len)
        test_dataset = NERDataset(test_seqs, test_labels, self.word_to_idx, max_len)
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, test_loader, train_seqs, test_seqs, train_labels, test_labels
    
    def train(self, sequences, labels, epochs=10, batch_size=32, max_len=100):
        """训练RNN模型"""
        print("正在训练RNN模型...")
        
        # 创建数据加载器
        train_loader, test_loader, train_seqs, test_seqs, train_labels, test_labels = \
            self.create_data_loaders(sequences, labels, batch_size, max_len)
        
        # 初始化模型
        self.model = RNNNER(
            vocab_size=self.word_count,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_classes=len(self.states),
            dropout=self.dropout,
            bidirectional=self.bidirectional
        ).to(self.device)
        
        # 损失函数和优化器
        criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略PAD标签
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # 训练循环
        train_losses = []
        test_f1_scores = []
        
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            total_loss = 0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                lengths = batch['length']
                
                # 前向传播
                logits = self.model(input_ids, lengths)
                
                # 计算损失
                loss = criterion(logits.view(-1, len(self.states)), labels.view(-1))
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            train_losses.append(avg_loss)
            
            # 评估阶段
            f1_score = self.evaluate_model(test_loader)
            test_f1_scores.append(f1_score)
            
            print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, F1 = {f1_score:.4f}")
        
        print("RNN模型训练完成!")
        
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
                labels = batch['labels'].to(self.device)
                lengths = batch['length']
                
                logits = self.model(input_ids, lengths)
                predictions = torch.argmax(logits, dim=-1)
                
                # 收集预测和标签
                for i, length in enumerate(lengths):
                    all_predictions.extend(predictions[i][:length].cpu().numpy())
                    all_labels.extend(labels[i][:length].cpu().numpy())
        
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
            # 转换为索引
            seq_indices = [self.word_to_idx.get(char, self.word_to_idx['<UNK>']) for char in seq]
            
            # 转换为tensor
            input_tensor = torch.tensor([seq_indices], dtype=torch.long).to(self.device)
            length = len(seq_indices)
            
            with torch.no_grad():
                logits = self.model(input_tensor, [length])
                pred_indices = torch.argmax(logits, dim=-1)[0].cpu().numpy()
            
            # 转换为标签
            pred_labels = [self.idx_to_state[idx] for idx in pred_indices]
            predictions.append(pred_labels)
        
        return predictions
    
    def predict_single(self, sequence):
        """预测单个序列"""
        return self.predict([sequence])[0]
    
    def evaluate(self, true_labels, pred_labels):
        """评估模型性能"""
        print("正在评估RNN模型...")
        
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
            'word_to_idx': self.word_to_idx,
            'idx_to_word': self.idx_to_word,
            'states': self.states,
            'state_to_idx': self.state_to_idx,
            'idx_to_state': self.idx_to_state,
            'model_config': {
                'embedding_dim': self.embedding_dim,
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
                'bidirectional': self.bidirectional
            }
        }
        
        torch.save(model_data, file_path)
        print(f"模型已保存到: {file_path}")
    
    def load_model(self, file_path):
        """加载模型"""
        model_data = torch.load(file_path, map_location=self.device)
        
        # 恢复配置
        config = model_data['model_config']
        self.embedding_dim = config['embedding_dim']
        self.hidden_dim = config['hidden_dim']
        self.num_layers = config['num_layers']
        self.dropout = config['dropout']
        self.bidirectional = config['bidirectional']
        
        # 恢复词汇表和标签
        self.word_to_idx = model_data['word_to_idx']
        self.idx_to_word = model_data['idx_to_word']
        self.states = model_data['states']
        self.state_to_idx = model_data['state_to_idx']
        self.idx_to_state = model_data['idx_to_state']
        self.word_count = len(self.word_to_idx)
        
        # 重建模型
        self.model = RNNNER(
            vocab_size=self.word_count,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_classes=len(self.states),
            dropout=self.dropout,
            bidirectional=self.bidirectional
        ).to(self.device)
        
        # 加载权重
        self.model.load_state_dict(model_data['model_state_dict'])
        
        print(f"模型已从 {file_path} 加载")

def main():
    """主函数"""
    # 初始化模型
    model = RNNNERModel(
        embedding_dim=128,
        hidden_dim=256,
        num_layers=2,
        dropout=0.5,
        bidirectional=True,
        learning_rate=0.001
    )
    
    # 加载数据
    sequences, labels = model.load_data('data/ccfbdci.jsonl')
    
    # 构建词汇表
    model.build_vocabulary(sequences, min_freq=2)
    
    # 训练模型
    train_losses, test_f1_scores = model.train(sequences, labels, epochs=10)
    
    # 创建测试数据加载器进行评估
    _, test_loader, _, test_seqs, _, test_labels = model.create_data_loaders(
        sequences, labels, batch_size=32
    )
    
    # 评估模型
    f1_score = model.evaluate_model(test_loader)
    print(f"最终F1分数: {f1_score:.4f}")
    
    # 预测测试集
    pred_labels = model.predict(test_seqs)
    
    # 详细评估
    model.evaluate(test_labels, pred_labels)
    
    # 保存模型
    model.save_model('models/rnn_ner_model.pth')
    
    print("RNN NER模型训练和评估完成!")

if __name__ == "__main__":
    main() 