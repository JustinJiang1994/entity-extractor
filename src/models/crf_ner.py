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
from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import flat_classification_report, flat_f1_score

class CRFNER:
    """基于条件随机场的命名实体识别"""
    
    def __init__(self, algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100):
        self.states = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-GPE', 'I-GPE']
        self.state_to_idx = {state: idx for idx, state in enumerate(self.states)}
        self.idx_to_state = {idx: state for state, idx in self.state_to_idx.items()}
        
        # CRF模型参数
        self.crf = CRF(
            algorithm=algorithm,
            c1=c1,  # L1正则化参数
            c2=c2,  # L2正则化参数
            max_iterations=max_iterations,
            all_possible_transitions=True
        )
        
        # 特征相关
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.word_count = 0
        self.feature_dict = {}
        
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
    
    def extract_features(self, sequence, position):
        """提取单个字符的特征"""
        features = {}
        
        # 当前字符
        char = sequence[position]
        features['char'] = char
        features['char_lower'] = char.lower()
        
        # 字符类型特征
        features['is_digit'] = char.isdigit()
        features['is_alpha'] = char.isalpha()
        features['is_upper'] = char.isupper()
        features['is_lower'] = char.islower()
        features['is_punct'] = not char.isalnum()
        
        # 位置特征
        features['position'] = position
        features['length'] = len(sequence)
        features['is_first'] = position == 0
        features['is_last'] = position == len(sequence) - 1
        
        # 前后字符特征
        if position > 0:
            prev_char = sequence[position - 1]
            features['prev_char'] = prev_char
            features['prev_char_lower'] = prev_char.lower()
            features['prev_is_digit'] = prev_char.isdigit()
            features['prev_is_alpha'] = prev_char.isalpha()
        else:
            features['prev_char'] = '<START>'
        
        if position < len(sequence) - 1:
            next_char = sequence[position + 1]
            features['next_char'] = next_char
            features['next_char_lower'] = next_char.lower()
            features['next_is_digit'] = next_char.isdigit()
            features['next_is_alpha'] = next_char.isalpha()
        else:
            features['next_char'] = '<END>'
        
        # 字符n-gram特征
        if position > 0 and position < len(sequence) - 1:
            features['bigram'] = sequence[position-1] + sequence[position]
            features['trigram'] = sequence[position-1] + sequence[position] + sequence[position+1]
        
        # 字符长度特征
        features['char_len'] = len(char)
        
        return features
    
    def sequence_to_features(self, sequence):
        """将序列转换为特征序列"""
        features = []
        for i in range(len(sequence)):
            features.append(self.extract_features(sequence, i))
        return features
    
    def prepare_training_data(self, sequences, labels):
        """准备训练数据"""
        print("正在准备训练数据...")
        
        X = []
        y = []
        
        for seq, label_seq in tqdm(zip(sequences, labels), desc="特征提取", total=len(sequences)):
            if len(seq) != len(label_seq):
                continue
            
            # 提取特征
            features = self.sequence_to_features(seq)
            X.append(features)
            y.append(label_seq)
        
        return X, y
    
    def train(self, sequences, labels):
        """训练CRF模型"""
        print("正在训练CRF模型...")
        
        # 准备训练数据
        X, y = self.prepare_training_data(sequences, labels)
        
        # 训练模型
        self.crf.fit(X, y)
        
        print("CRF模型训练完成!")
        
        # 打印模型信息
        print(f"模型参数: algorithm={self.crf.algorithm}, c1={self.crf.c1}, c2={self.crf.c2}")
        print(f"训练样本数: {len(X)}")
    
    def predict(self, sequences):
        """预测多个序列"""
        predictions = []
        
        for seq in tqdm(sequences, desc="预测"):
            features = self.sequence_to_features(seq)
            pred = self.crf.predict([features])[0]
            predictions.append(pred)
        
        return predictions
    
    def predict_single(self, sequence):
        """预测单个序列"""
        features = self.sequence_to_features(sequence)
        return self.crf.predict([features])[0]
    
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
        
        # 计算F1分数
        f1_score = flat_f1_score([flat_true], [flat_pred], average='weighted')
        
        print("分类报告:")
        print(report)
        print(f"加权F1分数: {f1_score:.4f}")
        
        return report, f1_score
    
    def get_feature_importance(self, top_n=20):
        """获取特征重要性"""
        if not hasattr(self.crf, 'state_features_'):
            print("模型未训练或特征重要性不可用")
            return {}
        
        # 获取状态特征
        state_features = self.crf.state_features_
        
        # 计算特征重要性
        feature_importance = defaultdict(float)
        
        for (state, feature), weight in state_features.items():
            feature_importance[feature] += abs(weight)
        
        # 排序并返回top_n
        sorted_features = sorted(feature_importance.items(), 
                               key=lambda x: x[1], reverse=True)
        
        return dict(sorted_features[:top_n])
    
    def save_model(self, file_path):
        """保存模型"""
        model_data = {
            'crf': self.crf,
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
        
        self.crf = model_data['crf']
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
    model = CRFNER(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100)
    
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
    
    # 获取特征重要性
    feature_importance = model.get_feature_importance(top_n=10)
    print("\n特征重要性 (Top 10):")
    for feature, importance in feature_importance.items():
        print(f"  {feature}: {importance:.4f}")
    
    # 保存模型
    model.save_model('crf_ner_model.pkl')
    
    print("CRF NER模型训练和评估完成!")

if __name__ == "__main__":
    main() 