#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.models.hmm_ner import HMMNER
from src.models.crf_ner import CRFNER
from src.models.rnn_ner import RNNNERModel
from src.models.cnn_ner import CNNNERModel
from src.models.gru_ner import GRUNERModel
from src.models.lstm_ner import LSTMNERModel
from src.models.bert_ner import BERTNERModel

def load_models():
    """加载训练好的模型"""
    models = {}
    
    # 加载HMM模型
    if os.path.exists('models/hmm_ner_model.pkl'):
        hmm_model = HMMNER()
        hmm_model.load_model('models/hmm_ner_model.pkl')
        models['hmm'] = hmm_model
        print("✅ HMM模型加载成功!")
    else:
        print("❌ HMM模型文件不存在")
    
    # 加载CRF模型
    if os.path.exists('models/crf_ner_model.pkl'):
        crf_model = CRFNER()
        crf_model.load_model('models/crf_ner_model.pkl')
        models['crf'] = crf_model
        print("✅ CRF模型加载成功!")
    else:
        print("❌ CRF模型文件不存在")
    
    return models

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

def predict_with_model(text, model, model_type):
    """使用指定模型进行预测"""
    char_seq = list(text)
    
    if model_type == 'hmm':
        pred_labels = model.viterbi_decode(char_seq)
    elif model_type == 'crf':
        pred_labels = model.predict_single(char_seq)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    entities = extract_entities_from_bio(char_seq, pred_labels)
    
    return pred_labels, entities

def interactive_demo():
    """交互式演示"""
    models = load_models()
    
    if not models:
        print("❌ 没有可用的模型，请先运行训练脚本")
        return
    
    print("\n" + "="*60)
    print("增强版NER交互式演示")
    print("="*60)
    print("输入文本进行实体识别，输入 'quit' 退出")
    print("支持的实体类型: PER(人名), ORG(组织), LOC(地点), GPE(地缘政治实体)")
    print("="*60)
    
    while True:
        try:
            text = input("\n请输入文本: ").strip()
            
            if text.lower() in ['quit', 'exit', '退出']:
                print("再见!")
                break
            
            if not text:
                print("请输入有效的文本")
                continue
            
            print(f"\n原文: {text}")
            print("-" * 50)
            
            # 使用所有可用模型进行预测
            for model_type, model in models.items():
                print(f"\n{model_type.upper()}模型结果:")
                pred_labels, entities = predict_with_model(text, model, model_type)
                
                print(f"标签: {' '.join(pred_labels)}")
                
                if entities:
                    print("识别到的实体:")
                    for i, entity in enumerate(entities, 1):
                        entity_type_map = {
                            'PER': '人名',
                            'ORG': '组织',
                            'LOC': '地点', 
                            'GPE': '地缘政治实体'
                        }
                        type_name = entity_type_map.get(entity['type'], entity['type'])
                        print(f"  {i}. {entity['text']} ({type_name})")
                else:
                    print("  无实体")
            
            print("-" * 50)
                
        except KeyboardInterrupt:
            print("\n\n再见!")
            break
        except Exception as e:
            print(f"处理出错: {e}")

def batch_demo():
    """批量演示"""
    models = load_models()
    
    if not models:
        print("❌ 没有可用的模型，请先运行训练脚本")
        return
    
    test_texts = [
        "菲律宾总统埃斯特拉达宣布重要决定。",
        "北京大学的李明教授发表了新论文。",
        "中国外交部发言人回应了相关问题。",
        "上海市政府发布了新的政策文件。",
        "这是一个普通的句子，没有实体。",
        "美国总统拜登访问了日本和韩国。",
        "清华大学计算机系的张教授获得了国际奖项。",
        "中国人民银行发布了新的货币政策。"
    ]
    
    print("\n" + "="*60)
    print("增强版NER批量演示")
    print("="*60)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n{i}. 原文: {text}")
        print("-" * 50)
        
        # 使用所有可用模型进行预测
        for model_type, model in models.items():
            pred_labels, entities = predict_with_model(text, model, model_type)
            
            print(f"\n{model_type.upper()}模型:")
            print(f"   标签: {' '.join(pred_labels)}")
            
            if entities:
                print("   实体:")
                for entity in entities:
                    entity_type_map = {
                        'PER': '人名',
                        'ORG': '组织',
                        'LOC': '地点', 
                        'GPE': '地缘政治实体'
                    }
                    type_name = entity_type_map.get(entity['type'], entity['type'])
                    print(f"     - {entity['text']} ({type_name})")
            else:
                print("   无实体")
        
        print("-" * 50)
    
    print("\n" + "="*60)

def model_comparison_demo():
    """模型对比演示"""
    models = load_models()
    
    if len(models) < 2:
        print("❌ 需要至少两个模型才能进行对比")
        return
    
    test_texts = [
        "菲律宾总统埃斯特拉达宣布重要决定。",
        "北京大学的李明教授发表了新论文。",
        "中国外交部发言人回应了相关问题。"
    ]
    
    print("\n" + "="*60)
    print("HMM vs CRF 模型对比演示")
    print("="*60)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n{i}. 原文: {text}")
        print("-" * 50)
        
        results = {}
        
        # 收集所有模型的预测结果
        for model_type, model in models.items():
            pred_labels, entities = predict_with_model(text, model, model_type)
            results[model_type] = {
                'labels': pred_labels,
                'entities': entities
            }
        
        # 对比显示结果
        print("模型对比:")
        for model_type, result in results.items():
            print(f"\n{model_type.upper()}:")
            print(f"  标签: {' '.join(result['labels'])}")
            
            if result['entities']:
                print("  实体:")
                for entity in result['entities']:
                    entity_type_map = {
                        'PER': '人名',
                        'ORG': '组织',
                        'LOC': '地点', 
                        'GPE': '地缘政治实体'
                    }
                    type_name = entity_type_map.get(entity['type'], entity['type'])
                    print(f"    - {entity['text']} ({type_name})")
            else:
                print("  无实体")
        
        # 分析差异
        if len(results) == 2:
            hmm_entities = set([(e['text'], e['type']) for e in results['hmm']['entities']])
            crf_entities = set([(e['text'], e['type']) for e in results['crf']['entities']])
            
            hmm_only = hmm_entities - crf_entities
            crf_only = crf_entities - hmm_entities
            common = hmm_entities & crf_entities
            
            print(f"\n对比分析:")
            print(f"  共同识别: {len(common)} 个实体")
            print(f"  仅HMM识别: {len(hmm_only)} 个实体")
            print(f"  仅CRF识别: {len(crf_only)} 个实体")
        
        print("-" * 50)
    
    print("\n" + "="*60)

def main():
    """主函数"""
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == 'batch':
            batch_demo()
        elif mode == 'comparison':
            model_comparison_demo()
        elif mode == 'hmm':
            # 仅使用HMM模型
            models = load_models()
            if 'hmm' in models:
                print("使用HMM模型进行演示...")
                # 这里可以添加HMM特定的演示逻辑
            else:
                print("HMM模型不可用")
        elif mode == 'crf':
            # 仅使用CRF模型
            models = load_models()
            if 'crf' in models:
                print("使用CRF模型进行演示...")
                # 这里可以添加CRF特定的演示逻辑
            else:
                print("CRF模型不可用")
        else:
            print("使用方法:")
            print("  python enhanced_demo.py              # 交互式演示")
            print("  python enhanced_demo.py batch        # 批量演示")
            print("  python enhanced_demo.py comparison   # 模型对比演示")
            print("  python enhanced_demo.py hmm          # 仅HMM模型")
            print("  python enhanced_demo.py crf          # 仅CRF模型")
    else:
        interactive_demo()

if __name__ == "__main__":
    main() 