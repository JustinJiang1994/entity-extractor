#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from hmm_ner import HMMNER

def load_model():
    """加载训练好的模型"""
    model = HMMNER()
    
    if os.path.exists('models/hmm_ner_model.pkl'):
        model.load_model('models/hmm_ner_model.pkl')
        print("✅ 模型加载成功!")
        return model
    else:
        print("❌ 模型文件不存在，请先运行 train_and_evaluate.py 训练模型")
        return None

def extract_entities(text, model):
    """从文本中提取实体"""
    char_seq = list(text)
    pred_labels = model.viterbi_decode(char_seq)
    
    entities = []
    current_entity = None
    
    for i, (char, label) in enumerate(zip(char_seq, pred_labels)):
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
    
    return entities, pred_labels

def interactive_demo():
    """交互式演示"""
    model = load_model()
    if not model:
        return
    
    print("\n" + "="*60)
    print("HMM NER 交互式演示")
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
            
            # 提取实体
            entities, labels = extract_entities(text, model)
            
            # 显示结果
            print(f"\n原文: {text}")
            print(f"标签: {' '.join(labels)}")
            
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
                print("未识别到实体")
                
        except KeyboardInterrupt:
            print("\n\n再见!")
            break
        except Exception as e:
            print(f"处理出错: {e}")

def batch_demo():
    """批量演示"""
    model = load_model()
    if not model:
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
    print("HMM NER 批量演示")
    print("="*60)
    
    for i, text in enumerate(test_texts, 1):
        entities, labels = extract_entities(text, model)
        
        print(f"\n{i}. 原文: {text}")
        print(f"   标签: {' '.join(labels)}")
        
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
    
    print("\n" + "="*60)

def main():
    """主函数"""
    if len(sys.argv) > 1 and sys.argv[1] == 'batch':
        batch_demo()
    else:
        interactive_demo()

if __name__ == "__main__":
    main() 