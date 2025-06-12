#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.models.hmm_ner import HMMNER
from src.models.crf_ner import CRFNER
from src.visualization.visualization import HMMVisualization

class ModelComparison:
    """模型对比类"""
    
    def __init__(self):
        self.hmm_model = HMMNER()
        self.crf_model = CRFNER(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100)
        self.viz = HMMVisualization()
        
        # 结果存储
        self.results = {}
        
    def load_and_prepare_data(self, data_file):
        """加载和准备数据"""
        print("=" * 60)
        print("数据加载和准备")
        print("=" * 60)
        
        # 加载数据
        sequences, labels = self.hmm_model.load_data(data_file)
        
        # 构建词汇表
        self.hmm_model.build_vocabulary(sequences, min_freq=2)
        self.crf_model.build_vocabulary(sequences, min_freq=2)
        
        # 划分数据集
        train_seqs, test_seqs, train_labels, test_labels = train_test_split(
            sequences, labels, test_size=0.2, random_state=42
        )
        
        print(f"训练集大小: {len(train_seqs)}")
        print(f"测试集大小: {len(test_seqs)}")
        
        return train_seqs, test_seqs, train_labels, test_labels
    
    def train_hmm_model(self, train_seqs, train_labels):
        """训练HMM模型"""
        print("\n" + "=" * 60)
        print("训练HMM模型")
        print("=" * 60)
        
        start_time = time.time()
        self.hmm_model.train(train_seqs, train_labels)
        hmm_train_time = time.time() - start_time
        
        print(f"HMM训练时间: {hmm_train_time:.2f}秒")
        
        return hmm_train_time
    
    def train_crf_model(self, train_seqs, train_labels):
        """训练CRF模型"""
        print("\n" + "=" * 60)
        print("训练CRF模型")
        print("=" * 60)
        
        start_time = time.time()
        self.crf_model.train(train_seqs, train_labels)
        crf_train_time = time.time() - start_time
        
        print(f"CRF训练时间: {crf_train_time:.2f}秒")
        
        return crf_train_time
    
    def evaluate_models(self, test_seqs, test_labels):
        """评估两个模型"""
        print("\n" + "=" * 60)
        print("模型评估")
        print("=" * 60)
        
        # HMM预测和评估
        print("\n--- HMM模型评估 ---")
        start_time = time.time()
        hmm_pred_labels = self.hmm_model.predict(test_seqs)
        hmm_pred_time = time.time() - start_time
        
        hmm_report = self.hmm_model.evaluate(test_labels, hmm_pred_labels)
        
        # CRF预测和评估
        print("\n--- CRF模型评估 ---")
        start_time = time.time()
        crf_pred_labels = self.crf_model.predict(test_seqs)
        crf_pred_time = time.time() - start_time
        
        crf_report, crf_f1 = self.crf_model.evaluate(test_labels, crf_pred_labels)
        
        # 存储结果
        self.results = {
            'hmm': {
                'pred_labels': hmm_pred_labels,
                'report': hmm_report,
                'pred_time': hmm_pred_time
            },
            'crf': {
                'pred_labels': crf_pred_labels,
                'report': crf_report,
                'f1_score': crf_f1,
                'pred_time': crf_pred_time
            },
            'test_labels': test_labels,
            'test_seqs': test_seqs
        }
        
        return hmm_pred_time, crf_pred_time
    
    def compare_performance(self):
        """比较模型性能"""
        print("\n" + "=" * 60)
        print("性能对比")
        print("=" * 60)
        
        # 计算HMM的F1分数
        flat_true = []
        flat_hmm_pred = []
        
        for true_seq, pred_seq in zip(self.results['test_labels'], self.results['hmm']['pred_labels']):
            min_len = min(len(true_seq), len(pred_seq))
            flat_true.extend(true_seq[:min_len])
            flat_hmm_pred.extend(pred_seq[:min_len])
        
        hmm_f1 = f1_score(flat_true, flat_hmm_pred, average='weighted')
        
        # 获取CRF的F1分数
        crf_f1 = self.results['crf']['f1_score']
        
        # 创建对比表格
        comparison_data = {
            '指标': ['F1分数', '预测时间(秒)', '模型复杂度'],
            'HMM': [f'{hmm_f1:.4f}', f'{self.results["hmm"]["pred_time"]:.2f}', '低'],
            'CRF': [f'{crf_f1:.4f}', f'{self.results["crf"]["pred_time"]:.2f}', '高']
        }
        
        df = pd.DataFrame(comparison_data)
        print("\n模型性能对比:")
        print(df.to_string(index=False))
        
        # 判断哪个模型更好
        if hmm_f1 > crf_f1:
            print(f"\n🏆 HMM模型表现更好 (F1: {hmm_f1:.4f} vs {crf_f1:.4f})")
        elif crf_f1 > hmm_f1:
            print(f"\n🏆 CRF模型表现更好 (F1: {crf_f1:.4f} vs {hmm_f1:.4f})")
        else:
            print(f"\n🤝 两个模型表现相当 (F1: {hmm_f1:.4f})")
        
        return hmm_f1, crf_f1
    
    def create_comparison_visualizations(self):
        """创建对比可视化"""
        print("\n" + "=" * 60)
        print("创建对比可视化")
        print("=" * 60)
        
        os.makedirs('comparison_results', exist_ok=True)
        
        # 1. F1分数对比
        hmm_f1, crf_f1 = self.compare_performance()
        
        plt.figure(figsize=(10, 6))
        models = ['HMM', 'CRF']
        f1_scores = [hmm_f1, crf_f1]
        colors = ['#FF6B6B', '#4ECDC4']
        
        bars = plt.bar(models, f1_scores, color=colors, alpha=0.7)
        plt.title('HMM vs CRF F1分数对比', fontsize=16, fontweight='bold')
        plt.ylabel('F1分数', fontsize=12)
        plt.ylim(0, 1)
        
        # 添加数值标签
        for bar, score in zip(bars, f1_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.4f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('comparison_results/f1_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. 预测时间对比
        hmm_time = self.results['hmm']['pred_time']
        crf_time = self.results['crf']['pred_time']
        
        plt.figure(figsize=(10, 6))
        times = [hmm_time, crf_time]
        
        bars = plt.bar(models, times, color=colors, alpha=0.7)
        plt.title('HMM vs CRF 预测时间对比', fontsize=16, fontweight='bold')
        plt.ylabel('预测时间 (秒)', fontsize=12)
        
        # 添加数值标签
        for bar, time_val in zip(bars, times):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{time_val:.2f}s', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('comparison_results/time_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. 混淆矩阵对比
        flat_true = []
        flat_hmm_pred = []
        flat_crf_pred = []
        
        for true_seq, hmm_pred, crf_pred in zip(
            self.results['test_labels'], 
            self.results['hmm']['pred_labels'], 
            self.results['crf']['pred_labels']
        ):
            min_len = min(len(true_seq), len(hmm_pred), len(crf_pred))
            flat_true.extend(true_seq[:min_len])
            flat_hmm_pred.extend(hmm_pred[:min_len])
            flat_crf_pred.extend(crf_pred[:min_len])
        
        # HMM混淆矩阵
        self.viz.plot_confusion_matrix(
            flat_true, flat_hmm_pred, self.hmm_model.states,
            'comparison_results/hmm_confusion_matrix.png'
        )
        
        # CRF混淆矩阵
        self.viz.plot_confusion_matrix(
            flat_true, flat_crf_pred, self.crf_model.states,
            'comparison_results/crf_confusion_matrix.png'
        )
    
    def save_comparison_results(self):
        """保存对比结果"""
        print("\n" + "=" * 60)
        print("保存对比结果")
        print("=" * 60)
        
        # 保存模型
        self.hmm_model.save_model('comparison_results/hmm_model.pkl')
        self.crf_model.save_model('comparison_results/crf_model.pkl')
        
        # 保存对比报告
        hmm_f1, crf_f1 = self.compare_performance()
        
        comparison_report = f"""
HMM vs CRF 模型对比报告
======================

数据集信息:
- 训练集大小: {len(self.results['test_seqs'])}
- 测试集大小: {len(self.results['test_labels'])}
- 实体类型: {len(self.hmm_model.states)} 种

模型性能对比:
- HMM F1分数: {hmm_f1:.4f}
- CRF F1分数: {crf_f1:.4f}
- HMM预测时间: {self.results['hmm']['pred_time']:.2f}秒
- CRF预测时间: {self.results['crf']['pred_time']:.2f}秒

模型特点:
HMM:
- 优点: 训练速度快，模型简单
- 缺点: 假设观测独立性，特征表达能力有限

CRF:
- 优点: 特征表达能力强，考虑标签间依赖关系
- 缺点: 训练时间长，模型复杂

结论:
{'HMM模型表现更好' if hmm_f1 > crf_f1 else 'CRF模型表现更好' if crf_f1 > hmm_f1 else '两个模型表现相当'}
"""
        
        with open('comparison_results/comparison_report.txt', 'w', encoding='utf-8') as f:
            f.write(comparison_report)
        
        print("对比结果已保存到 comparison_results/ 目录")
    
    def run_complete_comparison(self, data_file='data/ccfbdci.jsonl'):
        """运行完整的模型对比"""
        print("🚀 开始HMM vs CRF模型对比")
        
        # 1. 数据准备
        train_seqs, test_seqs, train_labels, test_labels = self.load_and_prepare_data(data_file)
        
        # 2. 训练模型
        hmm_train_time = self.train_hmm_model(train_seqs, train_labels)
        crf_train_time = self.train_crf_model(train_seqs, train_labels)
        
        # 3. 评估模型
        hmm_pred_time, crf_pred_time = self.evaluate_models(test_seqs, test_labels)
        
        # 4. 性能对比
        self.compare_performance()
        
        # 5. 创建可视化
        self.create_comparison_visualizations()
        
        # 6. 保存结果
        self.save_comparison_results()
        
        print("\n🎉 模型对比完成!")
        print("结果文件保存在 comparison_results/ 目录")

def main():
    """主函数"""
    comparison = ModelComparison()
    comparison.run_complete_comparison()

if __name__ == "__main__":
    main() 