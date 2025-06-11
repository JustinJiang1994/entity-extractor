#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json

def show_format_examples(file_path, num_examples=5):
    """展示数据格式的详细示例"""
    
    print("=== 数据格式详细示例 ===")
    print("格式说明：每行是一个JSON对象，包含以下字段：")
    print("- text: 原始文本")
    print("- entities: 实体列表，每个实体包含：")
    print("  * start_idx: 实体在文本中的起始位置")
    print("  * end_idx: 实体在文本中的结束位置")
    print("  * entity_text: 实体文本")
    print("  * entity_label: 实体类型标签")
    print("  * entity_names: 实体名称列表")
    print("- data_source: 数据来源")
    print()
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_examples:
                break
                
            data = json.loads(line.strip())
            
            print(f"--- 示例 {i+1} ---")
            print(f"原始JSON:")
            print(json.dumps(data, ensure_ascii=False, indent=2))
            print()
            
            print(f"格式化显示:")
            print(f"文本: {data['text']}")
            print(f"数据源: {data['data_source']}")
            print(f"实体数量: {len(data['entities'])}")
            
            if data['entities']:
                print("实体详情:")
                for j, entity in enumerate(data['entities']):
                    print(f"  {j+1}. 文本: '{entity['entity_text']}'")
                    print(f"     位置: {entity['start_idx']}-{entity['end_idx']}")
                    print(f"     类型: {entity['entity_label']}")
                    print(f"     名称: {', '.join(entity['entity_names'])}")
            else:
                print("  无实体")
            print("-" * 50)
            print()

if __name__ == "__main__":
    file_path = "data/ccfbdci.jsonl"
    show_format_examples(file_path, 3) 