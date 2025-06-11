# 基于HMM和CRF的中文命名实体识别 (HMM & CRF-based Chinese NER)

## 📋 项目概述

这是一个基于隐马尔可夫模型(HMM)和条件随机场(CRF)的中文命名实体识别项目，使用CCFBDCI数据集进行训练。项目实现了完整的NER流程，包括数据处理、模型训练、评估、对比和可视化。

## 🏗️ 项目结构

```
entity-extractor/
├── data/
│   └── ccfbdci.jsonl          # 原始数据集
├── models/                    # 训练好的模型
├── results/                   # 评估结果和可视化
├── comparison_results/        # 模型对比结果
├── hmm_ner.py                 # HMM NER模型实现
├── crf_ner.py                 # CRF NER模型实现
├── model_comparison.py        # 模型对比脚本
├── enhanced_demo.py           # 增强版演示脚本
├── visualization.py           # 可视化模块
├── train_and_evaluate.py      # 训练和评估脚本
├── demo.py                    # 基础演示脚本
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
# 训练HMM模型
python hmm_ner.py

# 训练CRF模型
python crf_ner.py

# 运行模型对比
python model_comparison.py
```

### 3. 使用模型

```bash
# 增强版交互式演示
python enhanced_demo.py

# 批量演示
python enhanced_demo.py batch

# 模型对比演示
python enhanced_demo.py comparison

# 仅使用HMM模型
python enhanced_demo.py hmm

# 仅使用CRF模型
python enhanced_demo.py crf
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

### HMM模型
- **状态空间**: 9个状态 (O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-GPE, I-GPE)
- **观测空间**: 字符级别的词汇表
- **标注方案**: BIO标注
- **解码算法**: Viterbi算法
- **核心参数**: 初始状态概率(π)、状态转移矩阵(A)、发射概率矩阵(B)

### CRF模型
- **状态空间**: 9个状态 (O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-GPE, I-GPE)
- **特征工程**: 字符级特征、上下文特征、位置特征、n-gram特征
- **标注方案**: BIO标注
- **训练算法**: L-BFGS优化
- **正则化**: L1和L2正则化

### 特征对比
| 特征类型 | HMM | CRF |
|---------|-----|-----|
| 字符特征 | ✅ | ✅ |
| 上下文特征 | ❌ | ✅ |
| 位置特征 | ❌ | ✅ |
| n-gram特征 | ❌ | ✅ |
| 标签依赖 | 马尔可夫假设 | 全局最优 |

## 🔧 核心功能

### 数据处理
- 自动加载JSONL格式数据
- 转换为BIO标注格式
- 构建字符级词汇表
- 数据集划分 (80%训练, 20%测试)

### 模型训练
- **HMM**: 统计学习参数，拉普拉斯平滑
- **CRF**: 特征工程，L-BFGS优化，正则化

### 模型评估
- 精确率、召回率、F1分数
- 混淆矩阵可视化
- 详细的分类报告
- 模型性能对比

### 可视化分析
- 实体分布饼图
- 训练指标统计
- 词汇表分析
- 模型性能对比
- 特征重要性分析

## 📈 使用示例

### 训练HMM模型
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

### 训练CRF模型
```python
from crf_ner import CRFNER

# 初始化模型
model = CRFNER(algorithm='lbfgs', c1=0.1, c2=0.1)

# 加载数据
sequences, labels = model.load_data('data/ccfbdci.jsonl')

# 构建词汇表
model.build_vocabulary(sequences)

# 训练模型
model.train(sequences, labels)

# 获取特征重要性
feature_importance = model.get_feature_importance(top_n=10)

# 保存模型
model.save_model('models/crf_ner_model.pkl')
```

### 模型对比
```python
from model_comparison import ModelComparison

# 运行完整对比
comparison = ModelComparison()
comparison.run_complete_comparison()
```

## 📊 模型性能

### 评估指标
- **整体准确率**: 基于字符级别的序列标注
- **实体识别**: 支持4种实体类型的识别
- **边界检测**: 精确的实体边界定位
- **F1分数**: 综合精确率和召回率

### 技术特点对比
| 特点 | HMM | CRF |
|------|-----|-----|
| 训练速度 | 快 | 慢 |
| 预测速度 | 快 | 中等 |
| 特征表达能力 | 有限 | 强 |
| 标签依赖建模 | 马尔可夫假设 | 全局最优 |
| 模型复杂度 | 低 | 高 |
| 过拟合风险 | 低 | 中等 |

## 🎯 应用场景

- **新闻文本分析**: 自动识别新闻中的人名、地名、组织名
- **知识图谱构建**: 实体抽取和关系挖掘
- **信息检索**: 提升搜索精度
- **文本挖掘**: 大规模文本的实体识别
- **自然语言处理**: 作为NER任务的基础模型
- **模型对比研究**: 比较不同序列标注算法的性能

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
- **置信度**: 基于模型概率的置信度

## 🔍 分析工具

### 数据分析
```bash
# 数据统计分析
python analyze_data.py

# 数据格式展示
python show_format_examples.py
```

### 模型训练
```bash
# 训练HMM模型
python hmm_ner.py

# 训练CRF模型
python crf_ner.py

# 模型对比
python model_comparison.py
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
| `crf_ner.py` | CRF NER模型核心实现 |
| `model_comparison.py` | HMM和CRF模型对比 |
| `enhanced_demo.py` | 支持多模型的演示脚本 |
| `visualization.py` | 可视化分析模块 |
| `train_and_evaluate.py` | 完整的训练评估流程 |
| `demo.py` | 基础演示脚本 |
| `analyze_data.py` | 数据集分析工具 |
| `show_format_examples.py` | 数据格式展示 |

## 🚀 部署说明

### 环境要求
- Python 3.7+
- NumPy 1.21+
- scikit-learn 1.0+
- sklearn-crfsuite 0.3.6+
- matplotlib 3.5+
- tqdm 4.62+

### 安装步骤
1. 克隆项目
2. 安装依赖: `pip install -r requirements.txt`
3. 运行训练: 
   - `python hmm_ner.py` (HMM模型)
   - `python crf_ner.py` (CRF模型)
   - `python model_comparison.py` (模型对比)
4. 使用模型: `python enhanced_demo.py`

## 📊 结果文件

训练完成后，会在以下目录生成结果：

### 模型文件
- `models/hmm_ner_model.pkl`: 训练好的HMM模型
- `models/crf_ner_model.pkl`: 训练好的CRF模型

### 评估结果
- `results/classification_report.txt`: 详细分类报告
- `results/prediction_examples.json`: 预测示例
- `results/model_statistics.json`: 模型统计信息
- `results/summary_report.txt`: 总结报告

### 对比结果
- `comparison_results/f1_comparison.png`: F1分数对比
- `comparison_results/time_comparison.png`: 预测时间对比
- `comparison_results/hmm_confusion_matrix.png`: HMM混淆矩阵
- `comparison_results/crf_confusion_matrix.png`: CRF混淆矩阵
- `comparison_results/comparison_report.txt`: 对比报告

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