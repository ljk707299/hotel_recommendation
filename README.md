
# 酒店推荐系统

## 项目简介

本项目基于Seattle酒店数据，利用文本处理和机器学习技术，实现基于酒店描述文本的相似度计算和酒店推荐功能。

项目包含两个Python脚本：

hotel_rec.py：基础版本，实现酒店描述的文本处理、特征提取（TF-IDF）、余弦相似度计算及推荐功能。

hotel_rec_nltk.py：改进版本，加入了NLTK的停用词处理，文本清洗更规范，支持更丰富的文本预处理。

## 目录结构示例

hotel_recommendation/

├── src/
|
│   ├── hotel_rec.py                # 基础版本代码
|
│   └── hotel_rec_nltk.py           # 使用NLTK的改进版代码
|
|   └── Seattle_Hotels.csv          # 酒店数据CSV文件
|
├── README.md
|
└── requirements.txt                # 依赖包列表

## 环境依赖

Python 3.7+

pandas

numpy

matplotlib

scikit-learn

nltk

### 建议使用虚拟环境安装依赖：

python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

pip install -r requirements.txt

requirements.txt 示例：

pandas

numpy

matplotlib

scikit-learn

nltk

## 使用说明

### 1. 数据准备
请确保 Seattle_Hotels.csv 文件位于 目录下。文件包含酒店的基本信息和描述文本。

### 2.基础版本（hotel_rec.py）

功能：

读取酒店数据

展示数据样例和描述

利用TF-IDF提取文本特征

计算余弦相似度

基于相似度给定酒店推荐Top10相似酒店

运行：

python src/hotel_rec.py

### 3.改进版本（hotel_rec_nltk.py）

主要改进：

集成NLTK停用词库，停用词过滤更精准

文本清洗步骤更规范（去除特殊符号、统一小写等）

依赖于NLTK资源，需要提前下载停用词库

注意事项：

需下载NLTK停用词资源。推荐在Python交互环境执行：


import nltk
nltk.download('stopwords')
如果网络无法访问，可以手动下载并设置 nltk.data.path。

运行：

python src/hotel_rec_nltk.py

## 代码结构说明
hotel_rec.py

读取数据

展示部分酒店描述

计算TF-IDF矩阵

计算余弦相似度

根据输入酒店名称推荐相似酒店


hotel_rec_nltk.py

引入NLTK下载停用词资源

定义正则表达式清洗文本

使用NLTK停用词过滤

TF-IDF特征提取与相似度计算同基础版本

推荐函数同基础版本

## 常见问题
1. NLTK下载停用词资源失败
网络环境限制，导致无法下载 stopwords。

解决方案：

手动下载 NLTK data 或相关资源，放置在指定路径。

在代码中指定路径：

import nltk
nltk.data.path.append('/path/to/your/nltk_data')
