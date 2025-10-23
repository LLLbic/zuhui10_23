# 文章Drift-oriented Self-evolving Encrypted Traffic  Application Classification for Actual Network  Environment
## 问题来源
文章指出，加密流量分类技术在实际网络环境中的应用受到概念漂移（concept drift）的挑战。互联网应用不断更新，包括功能调整和版本变化，导致特征表示发生变化，与原始样本不一致。这种漂移使得分类器性能迅速下降，重新训练模型的成本高昂，包括获取标注样本和计算资源，使得现有方法难以适应实际网络环境。
## 文章目的
文章旨在<span style="color:red">解决加密流量分类在实际网络环境中因概念漂移导致的分类器失效问题</span>，提出一种自演化（self-evolving）方法，通过持续微调分类器，延长其有效生命周期，而<span style="color:red">无需依赖昂贵的重新训练或大量标注样本</span>。
## 解决方案
文章提出了一种漂移导向的自演化加密流量分类方法，主要包括：

概念漂移判定：基于窗口多阈值累积测量方法，监测分类置信度的下降，判断是否发生漂移。
自演化微调：利用Laida准则从无标签预测中提取高置信度“银样本”（silver samples），通过完全微调（Fully Fine-Tuning, FFT）更新分类器，适应漂移。
## 方案新颖点

## 结果

## 文章的主要贡献



## 系统图示

