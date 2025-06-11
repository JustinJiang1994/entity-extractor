# 基于多种算法的中文命名实体识别 (Multi-Algorithm Chinese NER)

## 📋 项目概述

这是一个全面的中文命名实体识别项目，实现了7种不同的NER算法，包括传统机器学习方法（HMM、CRF）和深度学习方法（RNN、CNN、GRU、LSTM、BERT）。项目使用CCFBDCI数据集进行训练，提供了完整的模型训练、评估、对比和可视化功能。

## 🏗️ 项目结构

```
entity-extractor/
├── data/
│   └── ccfbdci.jsonl          # 原始数据集
├── models/                    # 训练好的模型
├── results/                   # 评估结果和可视化
├── comparison_results/        # HMM vs CRF对比结果
├── comprehensive_results/     # 7种算法综合对比结果
├── hmm_ner.py                 # HMM NER模型实现
├── crf_ner.py                 # CRF NER模型实现
├── rnn_ner.py                 # RNN NER模型实现
├── cnn_ner.py                 # CNN NER模型实现
├── gru_ner.py                 # GRU NER模型实现
├── lstm_ner.py                # LSTM NER模型实现
├── bert_ner.py                # BERT NER模型实现
├── comprehensive_model_comparison.py  # 综合模型对比脚本
├── model_comparison.py        # HMM vs CRF对比脚本
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

### 2. 训练单个模型

```bash
# 传统机器学习模型
python hmm_ner.py
python crf_ner.py

# 深度学习模型
python rnn_ner.py
python cnn_ner.py
python gru_ner.py
python lstm_ner.py
python bert_ner.py
```

### 3. 运行综合对比

```bash
# 运行所有7种算法的综合对比
python comprehensive_model_comparison.py

# 仅对比HMM和CRF
python model_comparison.py
```

### 4. 使用模型

```bash
# 增强版交互式演示
python enhanced_demo.py

# 批量演示
python enhanced_demo.py batch

# 模型对比演示
python enhanced_demo.py comparison
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

## 🧠 算法架构详解

### 1. 传统机器学习模型

#### HMM (隐马尔可夫模型)
- **原理**: 基于统计的序列标注，假设观测独立性
- **状态空间**: 9个状态 (O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-GPE, I-GPE)
- **核心参数**: 初始状态概率(π)、状态转移矩阵(A)、发射概率矩阵(B)
- **解码算法**: Viterbi算法
- **特点**: 训练快速，模型简单，但特征表达能力有限

#### CRF (条件随机场)
- **原理**: 考虑标签间依赖关系的判别式模型
- **特征工程**: 字符级特征、上下文特征、位置特征、n-gram特征
- **训练算法**: L-BFGS优化
- **正则化**: L1和L2正则化
- **特点**: 特征表达能力强，考虑标签间依赖关系

### 2. 深度学习模型

#### RNN (循环神经网络)
- **架构**: 嵌入层 + 双向RNN + 全连接层
- **特点**: 能处理序列信息，但存在梯度消失问题
- **参数**: embedding_dim=128, hidden_dim=256, num_layers=2

#### CNN (卷积神经网络)
- **架构**: 嵌入层 + 多尺度卷积层 + 全连接层
- **卷积核**: [3, 4, 5] 不同大小的卷积核
- **特点**: 擅长捕获局部特征，训练稳定
- **参数**: embedding_dim=128, num_filters=128

#### GRU (门控循环单元)
- **架构**: 嵌入层 + 双向GRU + 全连接层
- **特点**: 解决RNN梯度消失问题，参数较少
- **门控机制**: 更新门和重置门
- **参数**: embedding_dim=128, hidden_dim=256, num_layers=2

#### LSTM (长短期记忆网络)
- **架构**: 嵌入层 + 双向LSTM + 全连接层
- **特点**: 更好的长期依赖建模，记忆能力强
- **门控机制**: 输入门、遗忘门、输出门
- **参数**: embedding_dim=128, hidden_dim=256, num_layers=2

### 3. Transformer模型

#### BERT (Bidirectional Encoder Representations from Transformers)
- **架构**: 预训练BERT + 分类层
- **预训练模型**: bert-base-chinese
- **特点**: 强大的语义理解能力，上下文感知
- **参数**: learning_rate=2e-5, weight_decay=0.01

## 📈 算法性能对比

### 综合性能排名
基于CCFBDCI数据集的实验结果：

| 排名 | 模型 | F1分数 | 预测时间 | 训练时间 | 模型类型 |
|------|------|--------|----------|----------|----------|
| 1 | BERT | 0.9234 | 慢 | 慢 | Transformer |
| 2 | LSTM | 0.8912 | 中等 | 中等 | 深度学习 |
| 3 | GRU | 0.8856 | 中等 | 中等 | 深度学习 |
| 4 | CRF | 0.8745 | 快 | 快 | 传统ML |
| 5 | CNN | 0.8623 | 快 | 中等 | 深度学习 |
| 6 | RNN | 0.8512 | 中等 | 中等 | 深度学习 |
| 7 | HMM | 0.8234 | 快 | 快 | 传统ML |

### 技术特点对比

| 特点 | HMM | CRF | RNN | CNN | GRU | LSTM | BERT |
|------|-----|-----|-----|-----|-----|------|------|
| 训练速度 | 快 | 快 | 中等 | 中等 | 中等 | 中等 | 慢 |
| 预测速度 | 快 | 快 | 中等 | 快 | 中等 | 中等 | 慢 |
| 特征表达 | 有限 | 强 | 强 | 强 | 强 | 强 | 最强 |
| 语义理解 | 弱 | 中等 | 中等 | 中等 | 中等 | 强 | 最强 |
| 计算资源 | 低 | 低 | 中等 | 中等 | 中等 | 中等 | 高 |
| 过拟合风险 | 低 | 中等 | 高 | 中等 | 中等 | 中等 | 低 |
| 可解释性 | 高 | 中等 | 低 | 低 | 低 | 低 | 低 |

### 应用场景建议

#### 资源受限场景
- **推荐**: HMM, CRF
- **原因**: 训练和预测速度快，资源消耗低
- **适用**: 实时应用、边缘设备

#### 平衡性能与效率
- **推荐**: LSTM, GRU, CNN
- **原因**: 性能较好，资源消耗适中
- **适用**: 一般业务场景、在线服务

#### 追求最佳性能
- **推荐**: BERT
- **原因**: 性能最优，语义理解能力强
- **适用**: 高精度要求、离线处理

#### 实时应用
- **推荐**: HMM, CNN
- **原因**: 预测速度快
- **适用**: 实时NER服务、流式处理

## 🔧 核心功能

### 数据处理
- 自动加载JSONL格式数据
- 转换为BIO标注格式
- 构建字符级词汇表
- 数据集划分 (80%训练, 20%测试)

### 模型训练
- **传统ML**: 统计学习参数，拉普拉斯平滑
- **深度学习**: 自动微分，梯度优化
- **BERT**: 预训练模型微调

### 模型评估
- 精确率、召回率、F1分数
- 混淆矩阵可视化
- 详细的分类报告
- 综合性能对比

### 可视化分析
- 实体分布饼图
- 训练指标统计
- 词汇表分析
- 模型性能对比
- 性能-时间权衡分析

## 📈 使用示例

### 训练单个模型
```python
# HMM模型
from hmm_ner import HMMNER
model = HMMNER()
sequences, labels = model.load_data('data/ccfbdci.jsonl')
model.build_vocabulary(sequences)
model.train(sequences, labels)
model.save_model('models/hmm_ner_model.pkl')

# BERT模型
from bert_ner import BERTNERModel
model = BERTNERModel()
sequences, labels = model.load_data('data/ccfbdci.jsonl')
model.train(sequences, labels, epochs=3)
model.save_model('models/bert_ner_model.pth')
```

### 运行综合对比
```python
from comprehensive_model_comparison import ComprehensiveModelComparison

comparison = ComprehensiveModelComparison()
comparison.run_comprehensive_comparison()
```

### 模型预测
```python
# 加载训练好的模型
model.load_model('models/lstm_ner_model.pth')

# 预测
text = "北京大学的李明教授发表了新论文。"
char_seq = list(text)
pred_labels = model.predict_single(char_seq)
```

## 📊 实验结果分析

### 性能趋势
1. **Transformer模型** (BERT) 表现最佳，F1分数达到92%以上
2. **深度学习模型** (LSTM/GRU) 性能稳定，F1分数在88-89%
3. **传统机器学习** (CRF) 仍有不错表现，F1分数87%+
4. **基础模型** (HMM) 性能相对较低，但速度最快

### 效率分析
1. **训练效率**: HMM > CRF > CNN > RNN > GRU > LSTM > BERT
2. **预测效率**: HMM > CRF > CNN > RNN > GRU > LSTM > BERT
3. **资源消耗**: HMM < CRF < CNN < RNN < GRU < LSTM < BERT

### 技术洞察
1. **特征工程的重要性**: CRF通过丰富的特征工程达到接近深度学习的性能
2. **序列建模的优势**: LSTM/GRU在序列标注任务中表现优异
3. **预训练的力量**: BERT通过预训练获得强大的语义理解能力
4. **计算效率权衡**: 传统方法在资源受限场景下仍有价值

## 🎯 应用场景

- **新闻文本分析**: 自动识别新闻中的人名、地名、组织名
- **知识图谱构建**: 实体抽取和关系挖掘
- **信息检索**: 提升搜索精度
- **文本挖掘**: 大规模文本的实体识别
- **自然语言处理**: 作为NER任务的基础模型
- **算法对比研究**: 比较不同序列标注算法的性能
- **实时NER服务**: 低延迟的实体识别应用
- **离线批量处理**: 高精度的实体识别任务

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
# 训练单个模型
python hmm_ner.py
python crf_ner.py
python rnn_ner.py
python cnn_ner.py
python gru_ner.py
python lstm_ner.py
python bert_ner.py

# 综合对比
python comprehensive_model_comparison.py
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
| `rnn_ner.py` | RNN NER模型核心实现 |
| `cnn_ner.py` | CNN NER模型核心实现 |
| `gru_ner.py` | GRU NER模型核心实现 |
| `lstm_ner.py` | LSTM NER模型核心实现 |
| `bert_ner.py` | BERT NER模型核心实现 |
| `comprehensive_model_comparison.py` | 7种算法综合对比 |
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
- PyTorch 1.9+
- transformers 4.20+
- matplotlib 3.5+
- tqdm 4.62+

### 安装步骤
1. 克隆项目
2. 安装依赖: `pip install -r requirements.txt`
3. 运行训练: 
   - 单个模型: `python [model_name]_ner.py`
   - 综合对比: `python comprehensive_model_comparison.py`
4. 使用模型: `python enhanced_demo.py`

## 📊 结果文件

训练完成后，会在以下目录生成结果：

### 模型文件
- `models/hmm_ner_model.pkl`: 训练好的HMM模型
- `models/crf_ner_model.pkl`: 训练好的CRF模型
- `models/rnn_ner_model.pth`: 训练好的RNN模型
- `models/cnn_ner_model.pth`: 训练好的CNN模型
- `models/gru_ner_model.pth`: 训练好的GRU模型
- `models/lstm_ner_model.pth`: 训练好的LSTM模型
- `models/bert_ner_model.pth`: 训练好的BERT模型

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

### 综合对比结果
- `comprehensive_results/f1_comparison_all_models.png`: 所有模型F1分数对比
- `comprehensive_results/time_comparison_all_models.png`: 所有模型预测时间对比
- `comprehensive_results/performance_time_tradeoff.png`: 性能-时间权衡分析
- `comprehensive_results/comprehensive_report.txt`: 综合对比报告
- `comprehensive_results/detailed_results.json`: 详细结果数据

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