# 基于HMM的中文命名实体识别 (HMM-based Chinese NER)

## 📋 项目概述

这是一个基于隐马尔可夫模型(HMM)的中文命名实体识别项目，使用CCFBDCI数据集进行训练。项目实现了完整的NER流程，包括数据处理、模型训练、评估和可视化。

## 🏗️ 项目结构

```
entity-extractor/
├── data/
│   └── ccfbdci.jsonl          # 原始数据集
├── models/                    # 训练好的模型
├── results/                   # 评估结果和可视化
├── hmm_ner.py                 # HMM NER模型实现
├── visualization.py           # 可视化模块
├── train_and_evaluate.py      # 训练和评估脚本
├── demo.py                    # 演示脚本
├── analyze_data.py            # 数据分析脚本
├── show_format_examples.py    # 数据格式展示
├── requirements.txt           # 项目依赖
└── README.md                  # 项目文档
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 训练模型

```bash
python train_and_evaluate.py
```

### 3. 使用模型

```bash
# 交互式演示
python demo.py

# 批量演示
python demo.py batch
```

## 📊 数据集信息

### 基本统计
- **文件路径**: `data/ccfbdci.jsonl`
- **文件大小**: 4.0MB
- **数据行数**: 15,724行
- **格式**: JSONL (JSON Lines)
- **数据源**: CCFBDCI

### 实体类型分布
| 实体类型 | 标签 | 数量 | 占比 |
|---------|------|------|------|
| 地缘政治实体 | GPE | 4,586 | 36.8% |
| 人名 | PER | 3,995 | 32.0% |
| 组织 | ORG | 3,085 | 24.7% |
| 地点 | LOC | 915 | 7.3% |

## 🧠 模型架构

### HMM模型参数
- **状态空间**: 9个状态 (O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-GPE, I-GPE)
- **观测空间**: 字符级别的词汇表
- **标注方案**: BIO标注
- **解码算法**: Viterbi算法

### 核心组件
1. **初始状态概率 (π)**: 各状态的初始概率分布
2. **状态转移矩阵 (A)**: 状态间的转移概率
3. **发射概率矩阵 (B)**: 状态到观测的发射概率

## 🔧 核心功能

### 数据处理
- 自动加载JSONL格式数据
- 转换为BIO标注格式
- 构建字符级词汇表
- 数据集划分 (80%训练, 20%测试)

### 模型训练
- 统计学习HMM参数
- 拉普拉斯平滑处理
- 支持大规模数据训练

### 模型评估
- 精确率、召回率、F1分数
- 混淆矩阵可视化
- 详细的分类报告

### 可视化分析
- 实体分布饼图
- 训练指标统计
- 词汇表分析
- 模型性能对比

## 📈 使用示例

### 训练模型
```python
from hmm_ner import HMMNER

# 初始化模型
model = HMMNER()

# 加载数据
sequences, labels = model.load_data('data/ccfbdci.jsonl')

# 构建词汇表
model.build_vocabulary(sequences)

# 训练模型
model.train(sequences, labels)

# 保存模型
model.save_model('models/hmm_ner_model.pkl')
```

### 预测实体
```python
# 加载模型
model.load_model('models/hmm_ner_model.pkl')

# 预测文本
text = "菲律宾总统埃斯特拉达宣布重要决定。"
char_seq = list(text)
pred_labels = model.viterbi_decode(char_seq)

# 提取实体
entities = extract_entities_from_bio(char_seq, pred_labels)
```

## 📊 模型性能

### 评估指标
- **整体准确率**: 基于字符级别的序列标注
- **实体识别**: 支持4种实体类型的识别
- **边界检测**: 精确的实体边界定位

### 技术特点
- **字符级建模**: 适合中文文本处理
- **BIO标注**: 标准的序列标注方案
- **平滑处理**: 解决数据稀疏问题
- **高效解码**: Viterbi算法优化

## 🎯 应用场景

- **新闻文本分析**: 自动识别新闻中的人名、地名、组织名
- **知识图谱构建**: 实体抽取和关系挖掘
- **信息检索**: 提升搜索精度
- **文本挖掘**: 大规模文本的实体识别
- **自然语言处理**: 作为NER任务的基础模型

## 📝 数据格式

### 输入格式
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

### 输出格式
- **BIO标签**: 字符级别的序列标注
- **实体提取**: 结构化的实体信息
- **置信度**: 基于HMM概率的置信度

## 🔍 分析工具

### 数据分析
```bash
# 数据统计分析
python analyze_data.py

# 数据格式展示
python show_format_examples.py
```

### 可视化分析
```python
from visualization import HMMVisualization

viz = HMMVisualization()
viz.plot_entity_distribution('data/ccfbdci.jsonl')
viz.plot_confusion_matrix(y_true, y_pred, labels)
```

## 📋 文件说明

| 文件 | 功能 |
|------|------|
| `hmm_ner.py` | HMM NER模型核心实现 |
| `visualization.py` | 可视化分析模块 |
| `train_and_evaluate.py` | 完整的训练评估流程 |
| `demo.py` | 模型使用演示 |
| `analyze_data.py` | 数据集分析工具 |
| `show_format_examples.py` | 数据格式展示 |

## 🚀 部署说明

### 环境要求
- Python 3.7+
- NumPy 1.21+
- scikit-learn 1.0+
- matplotlib 3.5+
- tqdm 4.62+

### 安装步骤
1. 克隆项目
2. 安装依赖: `pip install -r requirements.txt`
3. 运行训练: `python train_and_evaluate.py`
4. 使用模型: `python demo.py`

## 📊 结果文件

训练完成后，会在以下目录生成结果：

### 模型文件
- `models/hmm_ner_model.pkl`: 训练好的HMM模型

### 评估结果
- `results/classification_report.txt`: 详细分类报告
- `results/prediction_examples.json`: 预测示例
- `results/model_statistics.json`: 模型统计信息
- `results/summary_report.txt`: 总结报告

### 可视化图表
- `results/confusion_matrix.png`: 混淆矩阵
- `results/entity_distribution.png`: 实体分布
- `results/training_metrics.png`: 训练指标
- `results/vocabulary_stats.png`: 词汇表统计
- `results/model_performance.png`: 模型性能

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进项目！

## 📄 许可证

本项目采用MIT许可证。

## 📞 联系方式

如有问题，请通过GitHub Issues联系。 