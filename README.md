# 实体抽取数据集 (Entity Extraction Dataset)

## 📋 数据集概述

这是一个用于命名实体识别(NER)任务的中文数据集，主要包含新闻文本中的人名、地名、组织名等实体的标注信息。

## 📁 文件信息

- **文件路径**: `data/ccfbdci.jsonl`
- **文件大小**: 4.0MB
- **数据行数**: 15,724行
- **格式**: JSONL (JSON Lines) - 每行一个JSON对象
- **数据源**: CCFBDCI

## 📊 数据统计

### 基本统计
- **解析成功率**: 100%
- **平均文本长度**: 31.3字符
- **文本长度范围**: 1-409字符
- **平均实体数**: 0.80个/样本
- **无实体样本**: 9,943个 (63.2%)

### 实体类型分布
| 实体类型 | 标签 | 数量 | 占比 |
|---------|------|------|------|
| 地缘政治实体 | GPE | 4,586 | 36.8% |
| 人名 | PER | 3,995 | 32.0% |
| 组织 | ORG | 3,085 | 24.7% |
| 地点 | LOC | 915 | 7.3% |

### 实体名称分布
| 实体名称 | 数量 |
|---------|------|
| 地缘政治实体 | 4,586 |
| 政治实体 | 4,586 |
| 地理实体 | 4,586 |
| 社会实体 | 4,586 |
| 人名 | 3,995 |
| 姓名 | 3,995 |
| 组织 | 3,085 |
| 团体 | 3,085 |
| 机构 | 3,085 |
| 地址 | 915 |

## 🏗️ 数据格式

### JSON结构
每行是一个JSON对象，包含以下字段：

```json
{
  "text": "原始文本内容",
  "entities": [
    {
      "start_idx": 实体起始位置,
      "end_idx": 实体结束位置,
      "entity_text": "实体文本",
      "entity_label": "实体类型标签",
      "entity_names": ["实体名称列表"]
    }
  ],
  "data_source": "CCFBDCI"
}
```

### 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `text` | string | 中文新闻文本 |
| `entities` | array | 实体标注列表 |
| `start_idx` | integer | 实体在文本中的起始字符位置 |
| `end_idx` | integer | 实体在文本中的结束字符位置 |
| `entity_text` | string | 实体原文 |
| `entity_label` | string | 实体类型标签 (GPE/PER/ORG/LOC) |
| `entity_names` | array | 实体中文名称列表 |
| `data_source` | string | 数据来源标识 |

## 📝 数据示例

### 示例1：多实体样本
```json
{
  "text": "菲律宾总统埃斯特拉达２号透过马尼拉当地电台宣布说，在仍遭到激进的回教阿卜沙耶夫组织羁押在非国南部和落岛的１６名人质当中，军方已经营救出了１１名菲律宾人质。",
  "entities": [
    {
      "start_idx": 0,
      "end_idx": 3,
      "entity_text": "菲律宾",
      "entity_label": "GPE",
      "entity_names": ["地缘政治实体", "政治实体", "地理实体", "社会实体"]
    },
    {
      "start_idx": 5,
      "end_idx": 10,
      "entity_text": "埃斯特拉达",
      "entity_label": "PER",
      "entity_names": ["人名", "姓名"]
    }
  ],
  "data_source": "CCFBDCI"
}
```

### 示例2：单实体样本
```json
{
  "text": "获救的人质为以前电视布道家阿美达为首的基督教传教士。",
  "entities": [
    {
      "start_idx": 13,
      "end_idx": 16,
      "entity_text": "阿美达",
      "entity_label": "PER",
      "entity_names": ["人名", "姓名"]
    }
  ],
  "data_source": "CCFBDCI"
}
```

### 示例3：无实体样本
```json
{
  "text": "继续是重要的国际新闻。",
  "entities": [],
  "data_source": "CCFBDCI"
}
```

## 🔧 使用方法

### 读取数据
```python
import json

# 读取JSONL文件
with open('data/ccfbdci.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line.strip())
        text = data['text']
        entities = data['entities']
        # 处理数据...
```

### 数据预处理
```python
def extract_entities(text, entities):
    """提取文本中的实体"""
    result = []
    for entity in entities:
        start = entity['start_idx']
        end = entity['end_idx']
        entity_text = entity['entity_text']
        entity_label = entity['entity_label']
        result.append({
            'text': entity_text,
            'label': entity_label,
            'position': (start, end)
        })
    return result
```

## 📈 数据特点

1. **高质量标注**: 100%的JSON解析成功率，数据质量良好
2. **丰富的实体类型**: 涵盖人名、地名、组织名、地缘政治实体等
3. **精确的位置标注**: 提供实体的精确字符位置信息
4. **多层级标签**: 每个实体包含多个中文名称标签
5. **真实新闻数据**: 基于真实的中文新闻文本

## 🎯 适用场景

- 命名实体识别(NER)模型训练
- 中文文本实体抽取
- 新闻文本分析
- 知识图谱构建
- 信息抽取系统开发

## 📝 注意事项

1. 数据采用UTF-8编码
2. 字符位置从0开始计算
3. 部分样本可能不包含任何实体
4. 实体位置信息基于字符级别标注

## 🔍 分析工具

项目包含两个分析脚本：
- `analyze_data.py`: 数据统计和分析
- `show_format_examples.py`: 格式示例展示

运行分析：
```bash
python analyze_data.py
python show_format_examples.py
``` 