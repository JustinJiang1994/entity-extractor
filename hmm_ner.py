#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle
import os

class HMMNER:
    """基于隐马尔可夫模型的命名实体识别"""
    
    def __init__(self):
        self.states = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-GPE', 'I-GPE']
        self.state_to_idx = {state: idx for idx, state in enumerate(self.states)}
        self.idx_to_state = {idx: state for state, idx in self.state_to_idx.items()}
        
        # HMM参数
        self.pi = None  # 初始状态概率
        self.A = None   # 状态转移矩阵
        self.B = None   # 发射概率矩阵
        
        # 词汇表
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.word_count = 0
        
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
        
        self.word_to_idx = {word: idx for idx, word in enumerate(valid_words)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        self.word_count = len(self.word_to_idx)
        
        print(f"词汇表大小: {self.word_count}")
    
    def train(self, sequences, labels):
        """训练HMM模型"""
        print("正在训练HMM模型...")
        
        n_states = len(self.states)
        n_words = self.word_count
        
        # 初始化参数
        self.pi = np.zeros(n_states)
        self.A = np.zeros((n_states, n_states))
        self.B = np.zeros((n_states, n_words))
        
        # 统计计数
        state_count = np.zeros(n_states)
        transition_count = np.zeros((n_states, n_states))
        emission_count = np.zeros((n_states, n_words))
        
        # 统计训练数据
        for seq, label_seq in tqdm(zip(sequences, labels), desc="统计参数", total=len(sequences)):
            if len(seq) != len(label_seq):
                continue
                
            for i, (word, label) in enumerate(zip(seq, label_seq)):
                word_idx = self.word_to_idx.get(word, self.word_to_idx['<UNK>'])
                state_idx = self.state_to_idx.get(label, 0)
                
                # 发射概率
                emission_count[state_idx, word_idx] += 1
                state_count[state_idx] += 1
                
                # 转移概率
                if i > 0:
                    prev_state_idx = self.state_to_idx.get(label_seq[i-1], 0)
                    transition_count[prev_state_idx, state_idx] += 1
        
        # 计算概率
        # 初始状态概率
        total_sequences = len(sequences)
        for i in range(n_states):
            self.pi[i] = state_count[i] / np.sum(state_count)
        
        # 状态转移概率
        for i in range(n_states):
            row_sum = np.sum(transition_count[i])
            if row_sum > 0:
                self.A[i] = transition_count[i] / row_sum
            else:
                self.A[i, i] = 1.0  # 自环
        
        # 发射概率
        for i in range(n_states):
            row_sum = np.sum(emission_count[i])
            if row_sum > 0:
                self.B[i] = emission_count[i] / row_sum
            else:
                self.B[i] = np.ones(n_words) / n_words  # 均匀分布
        
        # 添加平滑
        self._add_smoothing()
        
        print("HMM模型训练完成!")
    
    def _add_smoothing(self, alpha=0.1):
        """添加平滑处理"""
        # 转移概率平滑
        self.A += alpha
        self.A /= self.A.sum(axis=1, keepdims=True)
        
        # 发射概率平滑
        self.B += alpha
        self.B /= self.B.sum(axis=1, keepdims=True)
    
    def viterbi_decode(self, sequence):
        """使用Viterbi算法进行解码"""
        n_states = len(self.states)
        n_words = len(sequence)
        
        # 动态规划表
        dp = np.zeros((n_states, n_words))
        backpointer = np.zeros((n_states, n_words), dtype=int)
        
        # 初始化
        for i in range(n_states):
            word_idx = self.word_to_idx.get(sequence[0], self.word_to_idx['<UNK>'])
            dp[i, 0] = self.pi[i] * self.B[i, word_idx]
        
        # 前向递推
        for t in range(1, n_words):
            for j in range(n_states):
                word_idx = self.word_to_idx.get(sequence[t], self.word_to_idx['<UNK>'])
                emission_prob = self.B[j, word_idx]
                
                max_prob = -np.inf
                max_state = 0
                
                for i in range(n_states):
                    prob = dp[i, t-1] * self.A[i, j] * emission_prob
                    if prob > max_prob:
                        max_prob = prob
                        max_state = i
                
                dp[j, t] = max_prob
                backpointer[j, t] = max_state
        
        # 回溯
        best_path = np.zeros(n_words, dtype=int)
        best_path[n_words-1] = np.argmax(dp[:, n_words-1])
        
        for t in range(n_words-2, -1, -1):
            best_path[t] = backpointer[best_path[t+1], t+1]
        
        return [self.idx_to_state[idx] for idx in best_path]
    
    def predict(self, sequences):
        """预测多个序列"""
        predictions = []
        for seq in tqdm(sequences, desc="预测"):
            pred = self.viterbi_decode(seq)
            predictions.append(pred)
        return predictions
    
    def evaluate(self, true_labels, pred_labels):
        """评估模型性能"""
        print("正在评估模型...")
        
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
        
        print("分类报告:")
        print(report)
        
        return report
    
    def save_model(self, file_path):
        """保存模型"""
        model_data = {
            'pi': self.pi,
            'A': self.A,
            'B': self.B,
            'word_to_idx': self.word_to_idx,
            'idx_to_word': self.idx_to_word,
            'states': self.states,
            'state_to_idx': self.state_to_idx,
            'idx_to_state': self.idx_to_state
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"模型已保存到: {file_path}")
    
    def load_model(self, file_path):
        """加载模型"""
        with open(file_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.pi = model_data['pi']
        self.A = model_data['A']
        self.B = model_data['B']
        self.word_to_idx = model_data['word_to_idx']
        self.idx_to_word = model_data['idx_to_word']
        self.states = model_data['states']
        self.state_to_idx = model_data['state_to_idx']
        self.idx_to_state = model_data['idx_to_state']
        self.word_count = len(self.word_to_idx)
        
        print(f"模型已从 {file_path} 加载")

def main():
    """主函数"""
    # 初始化模型
    model = HMMNER()
    
    # 加载数据
    sequences, labels = model.load_data('data/ccfbdci.jsonl')
    
    # 构建词汇表
    model.build_vocabulary(sequences, min_freq=2)
    
    # 划分训练集和测试集
    train_seqs, test_seqs, train_labels, test_labels = train_test_split(
        sequences, labels, test_size=0.2, random_state=42
    )
    
    print(f"训练集大小: {len(train_seqs)}")
    print(f"测试集大小: {len(test_seqs)}")
    
    # 训练模型
    model.train(train_seqs, train_labels)
    
    # 预测测试集
    pred_labels = model.predict(test_seqs)
    
    # 评估模型
    model.evaluate(test_labels, pred_labels)
    
    # 保存模型
    model.save_model('hmm_ner_model.pkl')
    
    print("HMM NER模型训练和评估完成!")

if __name__ == "__main__":
    main() 