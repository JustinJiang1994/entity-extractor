#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
命名实体识别项目主运行脚本

使用方法:
python main.py [command] [options]

可用命令:
- analyze: 分析数据
- train: 训练模型
- demo: 交互式演示
- compare: 模型对比
- comprehensive: 综合模型对比
"""

import sys
import os
import argparse

# 添加项目根目录到路径
sys.path.append(os.path.dirname(__file__))

def main():
    parser = argparse.ArgumentParser(description='命名实体识别项目')
    parser.add_argument('command', choices=['analyze', 'train', 'demo', 'compare', 'comprehensive'],
                       help='要执行的命令')
    parser.add_argument('--model', choices=['hmm', 'crf', 'rnn', 'cnn', 'gru', 'lstm', 'bert'],
                       help='指定模型类型')
    parser.add_argument('--data', default='data/ccfbdci.jsonl',
                       help='数据文件路径')
    parser.add_argument('--epochs', type=int, default=10,
                       help='训练轮数')
    
    args = parser.parse_args()
    
    if args.command == 'analyze':
        from src.utils.analyze_data import analyze_jsonl_data
        print("🔍 开始数据分析...")
        analyze_jsonl_data(args.data)
        
    elif args.command == 'train':
        if not args.model:
            print("❌ 请指定要训练的模型类型: --model [hmm|crf|rnn|cnn|gru|lstm|bert]")
            return
            
        print(f"🚀 开始训练 {args.model.upper()} 模型...")
        
        if args.model == 'hmm':
            from scripts.train_and_evaluate import train_and_evaluate
            train_and_evaluate()
        elif args.model == 'crf':
            from src.models.crf_ner import CRFNER
            model = CRFNER()
            sequences, labels = model.load_data(args.data)
            model.build_vocabulary(sequences)
            model.train(sequences, labels)
            model.save_model(f'models/{args.model}_ner_model.pkl')
        else:
            # 深度学习模型
            model_map = {
                'rnn': 'src.models.rnn_ner.RNNNERModel',
                'cnn': 'src.models.cnn_ner.CNNNERModel', 
                'gru': 'src.models.gru_ner.GRUNERModel',
                'lstm': 'src.models.lstm_ner.LSTMNERModel',
                'bert': 'src.models.bert_ner.BERTNERModel'
            }
            
            module_name, class_name = model_map[args.model].rsplit('.', 1)
            module = __import__(module_name, fromlist=[class_name])
            ModelClass = getattr(module, class_name)
            
            model = ModelClass()
            sequences, labels = model.load_data(args.data)
            model.build_vocabulary(sequences)
            model.train(sequences, labels, epochs=args.epochs)
            model.save_model(f'models/{args.model}_ner_model.pth')
            
    elif args.command == 'demo':
        print("🎮 启动交互式演示...")
        from scripts.demo import interactive_demo
        interactive_demo()
        
    elif args.command == 'compare':
        print("⚖️ 开始模型对比...")
        from src.comparison.model_comparison import ModelComparison
        comparison = ModelComparison()
        comparison.run_complete_comparison(args.data)
        
    elif args.command == 'comprehensive':
        print("🔬 开始综合模型对比...")
        from src.comparison.comprehensive_model_comparison import ComprehensiveModelComparison
        comparison = ComprehensiveModelComparison()
        comparison.run_comprehensive_comparison(args.data)

if __name__ == "__main__":
    main() 