#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import pickle
from hmm_ner import HMMNER
from visualization import HMMVisualization
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
from tqdm import tqdm

def train_and_evaluate():
    """完整的训练和评估流程"""
    
    print("=" * 60)
    print("HMM NER模型训练和评估")
    print("=" * 60)
    
    # 初始化模型和可视化
    model = HMMNER()
    viz = HMMVisualization()
    
    # 1. 加载数据
    print("\n1. 加载数据...")
    sequences, labels = model.load_data('data/ccfbdci.jsonl')
    print(f"   加载完成: {len(sequences)} 个序列")
    
    # 2. 构建词汇表
    print("\n2. 构建词汇表...")
    model.build_vocabulary(sequences, min_freq=2)
    
    # 3. 划分数据集
    print("\n3. 划分数据集...")
    train_seqs, test_seqs, train_labels, test_labels = train_test_split(
        sequences, labels, test_size=0.2, random_state=42, stratify=None
    )
    
    print(f"   训练集: {len(train_seqs)} 个序列")
    print(f"   测试集: {len(test_seqs)} 个序列")
    
    # 4. 训练模型
    print("\n4. 训练HMM模型...")
    model.train(train_seqs, train_labels)
    
    # 5. 预测测试集
    print("\n5. 预测测试集...")
    pred_labels = model.predict(test_seqs)
    
    # 6. 评估模型
    print("\n6. 评估模型性能...")
    report = model.evaluate(test_labels, pred_labels)
    
    # 7. 保存模型
    print("\n7. 保存模型...")
    model.save_model('models/hmm_ner_model.pkl')
    
    # 8. 创建可视化
    print("\n8. 创建可视化...")
    os.makedirs('results', exist_ok=True)
    
    # 展平标签用于混淆矩阵
    flat_true = []
    flat_pred = []
    for true_seq, pred_seq in zip(test_labels, pred_labels):
        min_len = min(len(true_seq), len(pred_seq))
        flat_true.extend(true_seq[:min_len])
        flat_pred.extend(pred_seq[:min_len])
    
    # 绘制混淆矩阵
    viz.plot_confusion_matrix(flat_true, flat_pred, model.states, 
                             'results/confusion_matrix.png')
    
    # 绘制实体分布
    viz.plot_entity_distribution('data/ccfbdci.jsonl', 'results/entity_distribution.png')
    
    # 绘制训练指标
    viz.plot_training_metrics(len(train_seqs), len(test_seqs), 'results/training_metrics.png')
    
    # 绘制词汇表统计
    viz.plot_vocabulary_stats(model.word_to_idx, 'results/vocabulary_stats.png')
    
    # 绘制模型性能
    viz.plot_model_performance(report, 'results/model_performance.png')
    
    # 9. 创建总结报告
    print("\n9. 创建总结报告...")
    summary = viz.create_summary_report(model, 'data/ccfbdci.jsonl', 'results/summary_report.txt')
    
    # 10. 保存详细结果
    print("\n10. 保存详细结果...")
    save_detailed_results(model, test_seqs, test_labels, pred_labels, report)
    
    print("\n" + "=" * 60)
    print("训练和评估完成!")
    print("=" * 60)
    print(f"模型文件: models/hmm_ner_model.pkl")
    print(f"结果文件: results/")
    print(f"可视化文件: results/*.png")
    print("=" * 60)

def save_detailed_results(model, test_seqs, test_labels, pred_labels, report):
    """保存详细的评估结果"""
    
    # 保存分类报告
    with open('results/classification_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    # 保存预测结果示例
    results = []
    for i, (seq, true_label, pred_label) in enumerate(zip(test_seqs[:50], test_labels[:50], pred_labels[:50])):
        if i >= 50:  # 只保存前50个示例
            break
            
        result = {
            'id': i,
            'text': ''.join(seq),
            'true_labels': true_label,
            'pred_labels': pred_label,
            'correct': true_label == pred_label
        }
        results.append(result)
    
    with open('results/prediction_examples.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 保存模型参数统计
    model_stats = {
        'vocabulary_size': model.word_count,
        'state_count': len(model.states),
        'states': model.states,
        'pi_shape': model.pi.shape,
        'A_shape': model.A.shape,
        'B_shape': model.B.shape,
        'pi_sum': float(np.sum(model.pi)),
        'A_row_sums': [float(np.sum(row)) for row in model.A],
        'B_row_sums': [float(np.sum(row)) for row in model.B]
    }
    
    with open('results/model_statistics.json', 'w', encoding='utf-8') as f:
        json.dump(model_stats, f, ensure_ascii=False, indent=2)

def demo_prediction():
    """演示预测功能"""
    print("\n演示预测功能...")
    
    # 加载训练好的模型
    model = HMMNER()
    model.load_model('models/hmm_ner_model.pkl')
    
    # 测试文本
    test_texts = [
        "菲律宾总统埃斯特拉达宣布重要决定。",
        "北京大学的李明教授发表了新论文。",
        "中国外交部发言人回应了相关问题。",
        "这是一个普通的句子，没有实体。"
    ]
    
    print("\n预测结果:")
    print("-" * 50)
    
    for text in test_texts:
        char_seq = list(text)
        pred_labels = model.viterbi_decode(char_seq)
        
        print(f"文本: {text}")
        print(f"预测: {' '.join(pred_labels)}")
        
        # 提取实体
        entities = extract_entities_from_bio(char_seq, pred_labels)
        if entities:
            print("实体:")
            for entity in entities:
                print(f"  - {entity['text']} ({entity['type']})")
        else:
            print("  无实体")
        print("-" * 50)

def extract_entities_from_bio(char_seq, labels):
    """从BIO标签中提取实体"""
    entities = []
    current_entity = None
    
    for i, (char, label) in enumerate(zip(char_seq, labels)):
        if label.startswith('B-'):
            # 开始新实体
            if current_entity:
                entities.append(current_entity)
            current_entity = {
                'text': char,
                'type': label[2:],
                'start': i,
                'end': i
            }
        elif label.startswith('I-') and current_entity and label[2:] == current_entity['type']:
            # 继续当前实体
            current_entity['text'] += char
            current_entity['end'] = i
        else:
            # 结束当前实体
            if current_entity:
                entities.append(current_entity)
                current_entity = None
    
    # 处理最后一个实体
    if current_entity:
        entities.append(current_entity)
    
    return entities

def main():
    """主函数"""
    # 创建必要的目录
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # 训练和评估
    train_and_evaluate()
    
    # 演示预测
    demo_prediction()

if __name__ == "__main__":
    main() 