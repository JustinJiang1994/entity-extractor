#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
模型对比包

包含以下对比功能：
- model_comparison: HMM vs CRF对比
- comprehensive_model_comparison: 7种算法综合对比
"""

from .model_comparison import ModelComparison
from .comprehensive_model_comparison import ComprehensiveModelComparison

__all__ = [
    'ModelComparison',
    'ComprehensiveModelComparison'
] 