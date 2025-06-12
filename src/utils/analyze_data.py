#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from collections import Counter
import sys

def analyze_jsonl_data(file_path):
    """分析JSONL数据文件的格式和统计信息"""
    
    print("=== 数据格式分析 ===")
    print(f"文件路径: {file_path}")
    
    # 统计变量
    total_lines = 0
    valid_lines = 0
    entity_types = Counter()
    entity_names = Counter()
    text_lengths = []
    entity_counts = []
    
    # 读取并分析数据
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            total_lines += 1
            
            try:
                # 解析JSON
                data = json.loads(line.strip())
                valid_lines += 1
                
                # 提取基本信息
                text = data.get('text', '')
                entities = data.get('entities', [])
                data_source = data.get('data_source', '')
                
                # 统计文本长度
                text_lengths.append(len(text))
                
                # 统计实体数量
                entity_counts.append(len(entities))
                
                # 统计实体类型和名称
                for entity in entities:
                    entity_label = entity.get('entity_label', '')
                    entity_names_list = entity.get('entity_names', [])
                    
                    if entity_label:
                        entity_types[entity_label] += 1
                    
                    for name in entity_names_list:
                        entity_names[name] += 1
                
                # 打印前3个样本的详细信息
                if valid_lines <= 3:
                    print(f"\n--- 样本 {valid_lines} ---")
                    print(f"文本长度: {len(text)} 字符")
                    print(f"实体数量: {len(entities)}")
                    print(f"数据源: {data_source}")
                    print(f"文本: {text[:100]}{'...' if len(text) > 100 else ''}")
                    
                    if entities:
                        print("实体列表:")
                        for i, entity in enumerate(entities[:3]):  # 只显示前3个实体
                            print(f"  {i+1}. {entity.get('entity_text', '')} ({entity.get('entity_label', '')})")
                        if len(entities) > 3:
                            print(f"  ... 还有 {len(entities) - 3} 个实体")
                    else:
                        print("  无实体")
                
            except json.JSONDecodeError as e:
                print(f"第 {line_num} 行JSON解析错误: {e}")
                continue
    
    # 输出统计结果
    print(f"\n=== 统计结果 ===")
    print(f"总行数: {total_lines}")
    print(f"有效行数: {valid_lines}")
    print(f"解析成功率: {valid_lines/total_lines*100:.2f}%")
    
    if text_lengths:
        print(f"\n文本长度统计:")
        print(f"  平均长度: {sum(text_lengths)/len(text_lengths):.1f} 字符")
        print(f"  最短: {min(text_lengths)} 字符")
        print(f"  最长: {max(text_lengths)} 字符")
    
    if entity_counts:
        print(f"\n实体数量统计:")
        print(f"  平均实体数: {sum(entity_counts)/len(entity_counts):.2f}")
        print(f"  最少实体: {min(entity_counts)}")
        print(f"  最多实体: {max(entity_counts)}")
        print(f"  无实体的样本: {entity_counts.count(0)}")
    
    if entity_types:
        print(f"\n实体类型分布 (前10):")
        for entity_type, count in entity_types.most_common(10):
            print(f"  {entity_type}: {count}")
    
    if entity_names:
        print(f"\n实体名称分布 (前10):")
        for entity_name, count in entity_names.most_common(10):
            print(f"  {entity_name}: {count}")

if __name__ == "__main__":
    file_path = "data/ccfbdci.jsonl"
    analyze_jsonl_data(file_path) 