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

class ComprehensiveModelComparison:
    """综合模型对比类"""
    
    def __init__(self):
        self.models = {}
        self.viz = HMMVisualization()
        self.results = {}
        
        # 模型配置
        self.model_configs = {
            'hmm': HMMNER(),
            'crf': CRFNER(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100)
        }
        
    def load_and_prepare_data(self, data_file):
        """加载和准备数据"""
        print("=" * 80)
        print("数据加载和准备")
        print("=" * 80)
        
        # 使用HMM模型加载数据（所有模型使用相同的数据格式）
        sequences, labels = self.model_configs['hmm'].load_data(data_file)
        
        # 为所有模型构建词汇表
        for model_name, model in self.model_configs.items():
            if hasattr(model, 'build_vocabulary'):
                model.build_vocabulary(sequences, min_freq=2)
        
        # 划分数据集
        train_seqs, test_seqs, train_labels, test_labels = train_test_split(
            sequences, labels, test_size=0.2, random_state=42
        )
        
        print(f"训练集大小: {len(train_seqs)}")
        print(f"测试集大小: {len(test_seqs)}")
        
        return train_seqs, test_seqs, train_labels, test_labels
    
    def train_models(self, train_seqs, train_labels):
        """训练所有模型"""
        print("\n" + "=" * 80)
        print("模型训练")
        print("=" * 80)
        
        training_times = {}
        
        for model_name, model in self.model_configs.items():
            print(f"\n--- 训练 {model_name.upper()} 模型 ---")
            
            start_time = time.time()
            
            # 传统机器学习模型
            model.train(train_seqs, train_labels)
            
            training_time = time.time() - start_time
            training_times[model_name] = training_time
            
            print(f"{model_name.upper()} 训练时间: {training_time:.2f}秒")
        
        return training_times
    
    def evaluate_models(self, test_seqs, test_labels):
        """评估所有模型"""
        print("\n" + "=" * 80)
        print("模型评估")
        print("=" * 80)
        
        evaluation_results = {}
        
        for model_name, model in self.model_configs.items():
            print(f"\n--- 评估 {model_name.upper()} 模型 ---")
            
            start_time = time.time()
            
            # 预测
            pred_labels = model.predict(test_seqs)
            pred_time = time.time() - start_time
            
            # 评估
            if hasattr(model, 'evaluate'):
                report, f1_score_val = model.evaluate(test_labels, pred_labels)
            else:
                # 对于没有evaluate方法的模型，手动计算F1
                flat_true = []
                flat_pred = []
                
                for true_seq, pred_seq in zip(test_labels, pred_labels):
                    min_len = min(len(true_seq), len(pred_seq))
                    flat_true.extend(true_seq[:min_len])
                    flat_pred.extend(pred_seq[:min_len])
                
                f1_score_val = f1_score(flat_true, flat_pred, average='weighted')
                report = classification_report(flat_true, flat_pred, 
                                             target_names=model.states, 
                                             zero_division=0)
            
            evaluation_results[model_name] = {
                'pred_labels': pred_labels,
                'report': report,
                'f1_score': f1_score_val,
                'pred_time': pred_time
            }
            
            print(f"{model_name.upper()} F1分数: {f1_score_val:.4f}")
            print(f"{model_name.upper()} 预测时间: {pred_time:.2f}秒")
        
        self.results = evaluation_results
        return evaluation_results
    
    def compare_performance(self):
        """比较所有模型性能"""
        print("\n" + "=" * 80)
        print("综合性能对比")
        print("=" * 80)
        
        # 收集性能指标
        model_names = list(self.results.keys())
        f1_scores = [self.results[name]['f1_score'] for name in model_names]
        pred_times = [self.results[name]['pred_time'] for name in model_names]
        
        # 创建对比表格
        comparison_data = {
            '模型': model_names,
            'F1分数': [f'{score:.4f}' for score in f1_scores],
            '预测时间(秒)': [f'{time:.2f}' for time in pred_times]
        }
        
        df = pd.DataFrame(comparison_data)
        print("\n模型性能对比:")
        print(df.to_string(index=False))
        
        # 找出最佳模型
        best_model = model_names[np.argmax(f1_scores)]
        best_f1 = max(f1_scores)
        
        print(f"\n🏆 最佳模型: {best_model.upper()} (F1: {best_f1:.4f})")
        
        # 模型分类
        traditional_models = ['hmm', 'crf']
        
        print(f"\n📊 模型分类:")
        print(f"传统机器学习模型: {', '.join(traditional_models).upper()}")
        
        return model_names, f1_scores, pred_times
    
    def create_comprehensive_visualizations(self, test_labels):
        """创建综合可视化"""
        print("\n" + "=" * 80)
        print("创建综合可视化")
        print("=" * 80)
        
        os.makedirs('comprehensive_results', exist_ok=True)
        
        model_names, f1_scores, pred_times = self.compare_performance()
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 1. F1分数对比柱状图
        plt.figure(figsize=(14, 8))
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']
        
        bars = plt.bar(model_names, f1_scores, color=colors, alpha=0.7)
        plt.title('所有模型F1分数对比', fontsize=18, fontweight='bold')
        plt.ylabel('F1分数', fontsize=14)
        plt.xlabel('模型', fontsize=14)
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        
        # 添加数值标签
        for bar, score in zip(bars, f1_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.4f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('comprehensive_results/f1_comparison_all_models.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. 预测时间对比
        plt.figure(figsize=(14, 8))
        
        bars = plt.bar(model_names, pred_times, color=colors, alpha=0.7)
        plt.title('所有模型预测时间对比', fontsize=18, fontweight='bold')
        plt.ylabel('预测时间 (秒)', fontsize=14)
        plt.xlabel('模型', fontsize=14)
        plt.xticks(rotation=45)
        
        # 添加数值标签
        for bar, time_val in zip(bars, pred_times):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{time_val:.2f}s', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('comprehensive_results/time_comparison_all_models.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. 性能-时间散点图
        plt.figure(figsize=(12, 8))
        
        # 传统机器学习模型
        traditional_idx = [i for i, name in enumerate(model_names) if name in ['hmm', 'crf']]
        
        plt.scatter([pred_times[i] for i in traditional_idx], 
                   [f1_scores[i] for i in traditional_idx], 
                   c='red', s=100, label='传统机器学习', alpha=0.7)
        
        # 添加模型标签
        for i, name in enumerate(model_names):
            plt.annotate(name.upper(), (pred_times[i], f1_scores[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        plt.xlabel('预测时间 (秒)', fontsize=14)
        plt.ylabel('F1分数', fontsize=14)
        plt.title('模型性能-时间权衡分析', fontsize=18, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('comprehensive_results/performance_time_tradeoff.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 4. 混淆矩阵对比（选择代表性模型）
        representative_models = ['hmm', 'crf']
        
        for model_name in representative_models:
            if model_name in self.results:
                flat_true = []
                flat_pred = []
                
                for true_seq, pred_seq in zip(test_labels, 
                                             self.results[model_name]['pred_labels']):
                    min_len = min(len(true_seq), len(pred_seq))
                    flat_true.extend(true_seq[:min_len])
                    flat_pred.extend(pred_seq[:min_len])
                
                self.viz.plot_confusion_matrix(
                    flat_true, flat_pred, self.model_configs[model_name].states,
                    f'comprehensive_results/{model_name}_confusion_matrix.png'
                )
    
    def save_comprehensive_results(self, test_labels):
        """保存综合对比结果"""
        print("\n" + "=" * 80)
        print("保存综合对比结果")
        print("=" * 80)
        
        # 保存所有模型
        for model_name, model in self.model_configs.items():
            if hasattr(model, 'save_model'):
                model.save_model(f'comprehensive_results/{model_name}_model.pth')
        
        # 保存对比报告
        model_names, f1_scores, pred_times = self.compare_performance()
        
        comprehensive_report = f"""
综合模型对比报告
================

数据集信息:
- 训练集大小: {len(test_labels)}
- 测试集大小: {len(test_labels)}
- 实体类型: 4种 (PER, ORG, LOC, GPE)

模型性能排名:
"""
        
        # 按F1分数排序
        sorted_models = sorted(zip(model_names, f1_scores, pred_times), 
                              key=lambda x: x[1], reverse=True)
        
        for i, (model_name, f1, pred_time) in enumerate(sorted_models, 1):
            comprehensive_report += f"{i}. {model_name.upper()}: F1={f1:.4f}, 时间={pred_time:.2f}s\n"
        
        comprehensive_report += f"""

模型分类分析:

传统机器学习模型:
- HMM: 基于统计的序列标注，训练快速，但特征表达能力有限
- CRF: 考虑标签间依赖关系，特征工程丰富，性能较好

技术特点对比:
| 模型类型 | 训练速度 | 预测速度 | 特征表达 | 语义理解 | 计算资源 |
|---------|---------|---------|---------|---------|---------|
| 传统ML   | 快      | 快      | 有限     | 弱      | 低      |

结论:
- 最佳性能: {sorted_models[0][0].upper()} (F1: {sorted_models[0][1]:.4f})
- 最快速度: {min(zip(model_names, pred_times), key=lambda x: x[1])[0].upper()}

应用建议:
1. 资源受限场景: 选择HMM
2. 追求更好性能: 选择CRF
3. 实时应用: 选择HMM
4. 离线批处理: 选择CRF
"""
        
        with open('comprehensive_results/comprehensive_report.txt', 'w', encoding='utf-8') as f:
            f.write(comprehensive_report)
        
        # 保存详细结果到JSON
        detailed_results = {
            'model_performance': {
                name: {
                    'f1_score': float(score),
                    'prediction_time': float(time)
                } for name, score, time in zip(model_names, f1_scores, pred_times)
            },
            'best_model': sorted_models[0][0],
            'best_f1_score': float(sorted_models[0][1]),
            'fastest_model': min(zip(model_names, pred_times), key=lambda x: x[1])[0]
        }
        
        with open('comprehensive_results/detailed_results.json', 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, ensure_ascii=False, indent=2)
        
        print("综合结果已保存到 comprehensive_results/ 目录")
    
    def run_comprehensive_comparison(self, data_file='data/ccfbdci.jsonl'):
        """运行综合模型对比"""
        print("🚀 开始综合模型对比")
        print("包含模型: HMM, CRF")
        
        # 1. 数据准备
        train_seqs, test_seqs, train_labels, test_labels = self.load_and_prepare_data(data_file)
        
        # 2. 训练模型
        training_times = self.train_models(train_seqs, train_labels)
        
        # 3. 评估模型
        evaluation_results = self.evaluate_models(test_seqs, test_labels)
        
        # 4. 性能对比
        self.compare_performance()
        
        # 5. 创建可视化
        self.create_comprehensive_visualizations(test_labels)
        
        # 6. 保存结果
        self.save_comprehensive_results(test_labels)
        
        print("\n🎉 综合模型对比完成!")
        print("结果文件保存在 comprehensive_results/ 目录")

def main():
    """主函数"""
    comparison = ComprehensiveModelComparison()
    comparison.run_comprehensive_comparison()

if __name__ == "__main__":
    main() 