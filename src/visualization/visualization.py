#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import json
from collections import Counter
import matplotlib.font_manager as fm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class HMMVisualization:
    """HMM NER模型可视化类"""
    
    def __init__(self):
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F']
    
    def plot_confusion_matrix(self, y_true, y_pred, labels, save_path='confusion_matrix.png'):
        """绘制混淆矩阵"""
        plt.figure(figsize=(12, 10))
        
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        # 计算百分比
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # 创建热力图
        sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                   xticklabels=labels, yticklabels=labels,
                   cbar_kws={'label': '准确率 (%)'})
        
        plt.title('HMM NER模型混淆矩阵', fontsize=16, fontweight='bold')
        plt.xlabel('预测标签', fontsize=12)
        plt.ylabel('真实标签', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_entity_distribution(self, data_file, save_path='entity_distribution.png'):
        """绘制实体分布图"""
        entity_counts = Counter()
        entity_types = Counter()
        
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                entities = data['entities']
                
                for entity in entities:
                    entity_text = entity['entity_text']
                    entity_label = entity['entity_label']
                    
                    entity_counts[entity_text] += 1
                    entity_types[entity_label] += 1
        
        # 绘制实体类型分布
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 实体类型分布
        labels = list(entity_types.keys())
        sizes = list(entity_types.values())
        colors = self.colors[:len(labels)]
        
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        ax1.set_title('实体类型分布', fontsize=14, fontweight='bold')
        
        # 前10个最常见实体
        top_entities = entity_counts.most_common(10)
        entity_names = [item[0] for item in top_entities]
        entity_freqs = [item[1] for item in top_entities]
        
        ax2.barh(range(len(entity_names)), entity_freqs, color=self.colors[0])
        ax2.set_yticks(range(len(entity_names)))
        ax2.set_yticklabels(entity_names)
        ax2.set_xlabel('出现次数')
        ax2.set_title('最常见实体 (Top 10)', fontsize=14, fontweight='bold')
        ax2.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_training_metrics(self, train_sizes, test_sizes, save_path='training_metrics.png'):
        """绘制训练指标"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 数据集大小
        labels = ['训练集', '测试集']
        sizes = [train_sizes, test_sizes]
        colors = [self.colors[0], self.colors[1]]
        
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        ax1.set_title('数据集划分', fontsize=14, fontweight='bold')
        
        # 数据统计
        stats = ['总样本数', '有实体样本', '无实体样本']
        values = [train_sizes + test_sizes, train_sizes + test_sizes - 9943, 9943]
        
        ax2.bar(stats, values, color=self.colors[2:5])
        ax2.set_ylabel('样本数量')
        ax2.set_title('数据统计', fontsize=14, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for i, v in enumerate(values):
            ax2.text(i, v + 100, str(v), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_vocabulary_stats(self, word_to_idx, save_path='vocabulary_stats.png'):
        """绘制词汇表统计"""
        # 统计字符频率
        char_freq = Counter()
        for word in word_to_idx.keys():
            if word != '<UNK>':
                char_freq.update(word)
        
        # 绘制前20个最常见字符
        top_chars = char_freq.most_common(20)
        chars = [item[0] for item in top_chars]
        freqs = [item[1] for item in top_chars]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(chars)), freqs, color=self.colors[0])
        plt.xlabel('字符')
        plt.ylabel('出现次数')
        plt.title('最常见字符 (Top 20)', fontsize=14, fontweight='bold')
        plt.xticks(range(len(chars)), chars)
        
        # 添加数值标签
        for i, (bar, freq) in enumerate(zip(bars, freqs)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                    str(freq), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_model_performance(self, classification_report_str, save_path='model_performance.png'):
        """绘制模型性能报告"""
        # 解析分类报告
        lines = classification_report_str.strip().split('\n')
        
        # 提取数据
        data = []
        for line in lines[2:-3]:  # 跳过标题和总计行
            if line.strip():
                parts = line.split()
                if len(parts) >= 5:
                    label = parts[0]
                    precision = float(parts[1])
                    recall = float(parts[2])
                    f1 = float(parts[3])
                    support = int(parts[4])
                    data.append([label, precision, recall, f1, support])
        
        df = pd.DataFrame(data, columns=['Label', 'Precision', 'Recall', 'F1-Score', 'Support'])
        
        # 绘制性能指标
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Precision
        axes[0, 0].bar(df['Label'], df['Precision'], color=self.colors[0])
        axes[0, 0].set_title('Precision (精确率)', fontweight='bold')
        axes[0, 0].set_ylabel('Precision')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Recall
        axes[0, 1].bar(df['Label'], df['Recall'], color=self.colors[1])
        axes[0, 1].set_title('Recall (召回率)', fontweight='bold')
        axes[0, 1].set_ylabel('Recall')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # F1-Score
        axes[1, 0].bar(df['Label'], df['F1-Score'], color=self.colors[2])
        axes[1, 0].set_title('F1-Score (F1分数)', fontweight='bold')
        axes[1, 0].set_ylabel('F1-Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Support
        axes[1, 1].bar(df['Label'], df['Support'], color=self.colors[3])
        axes[1, 1].set_title('Support (支持度)', fontweight='bold')
        axes[1, 1].set_ylabel('Support')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_summary_report(self, model, data_file, save_path='summary_report.txt'):
        """创建总结报告"""
        # 统计数据
        total_samples = 0
        entity_samples = 0
        entity_counts = Counter()
        
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                total_samples += 1
                data = json.loads(line.strip())
                entities = data['entities']
                
                if entities:
                    entity_samples += 1
                
                for entity in entities:
                    entity_counts[entity['entity_label']] += 1
        
        # 生成报告
        report = f"""
HMM NER模型总结报告
==================

数据集信息:
- 总样本数: {total_samples:,}
- 有实体样本: {entity_samples:,}
- 无实体样本: {total_samples - entity_samples:,}
- 实体样本比例: {entity_samples/total_samples*100:.1f}%

实体类型分布:
"""
        
        for entity_type, count in entity_counts.most_common():
            report += f"- {entity_type}: {count:,} ({count/sum(entity_counts.values())*100:.1f}%)\n"
        
        report += f"""
模型信息:
- 词汇表大小: {model.word_count:,}
- 状态数量: {len(model.states)}
- 状态列表: {', '.join(model.states)}

模型参数:
- 初始状态概率矩阵形状: {model.pi.shape}
- 状态转移矩阵形状: {model.A.shape}
- 发射概率矩阵形状: {model.B.shape}

技术特点:
- 使用BIO标注方案
- Viterbi算法进行解码
- 拉普拉斯平滑处理
- 字符级别的序列标注
"""
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"总结报告已保存到: {save_path}")
        return report

def main():
    """主函数 - 演示可视化功能"""
    viz = HMMVisualization()
    
    # 绘制实体分布
    viz.plot_entity_distribution('data/ccfbdci.jsonl')
    
    # 绘制训练指标
    viz.plot_training_metrics(12579, 3145)  # 80%训练，20%测试
    
    print("可视化完成!")

if __name__ == "__main__":
    main() 