
# PRECISE
**PRotein Enzymatic Cleavage and In-Silico Evaluation**

*A protein enzymatic cleavage and bioactive peptide prediction platform with integrated self-trained models and enzyme combination optimization.*

一个本人自行训练及集成其它方法的蛋白酶切和功能肽预测平台，额外提供酶组合优化。

---

> 📢 **This work has been submitted to *Food Chemistry* and is currently under revision. The DOI will be updated upon acceptance. Researchers and practitioners are welcome to use and extend this platform.**
>
> 📢 **本研究已投稿至 *Food Chemistry*，目前处于修订状态。DOI 将在论文接收后补充。欢迎同行和业者使用及扩展本平台。**

---

## Overview | 项目简介

**PRECISE** is a comprehensive protein analysis platform focused on protease cleavage simulation, bioactive peptide prediction, and enzyme combination optimization. The platform leverages deep learning models and heuristic algorithms for efficient bioactive peptide screening and cleavage strategy design.

**PRECISE** 是一个综合性蛋白质分析平台，专注于蛋白酶切模拟、功能肽预测和酶切组合优化。该平台通过深度学习模型和启发式算法，实现高效的功能肽筛选和酶切策略设计。

---

## Features | 核心功能

### 🔬 Virtual Protein Digestion | 蛋白虚拟酶切
Accurate protein digestion simulation based on regex-defined cleavage rules.

基于正则化方法的精确蛋白质酶切模拟。

### 🧬 Bioactive Peptide Prediction | 功能肽预测
Predicts four key bioactivities | 预测四种关键活性功能：
- Antioxidant activity | 抗氧化活性
- ACE inhibitory activity | ACE抑制活性
- Antimicrobial activity | 抗菌活性
- DPP-IV inhibitory activity | DPPIV抑制活性

提供肽序列的分子结构图下载功能。

### ⚙️ Enzyme Combination Optimization | 酶组合优化
Optimizes enzyme combinations for target bioactivities using a simulated annealing algorithm to maximize yield of desired bioactive peptides.

基于模拟退火算法，针对特定活性优化酶切组合，提高目标活性肽的产率。

---

## Technical Implementation | 技术实现

### Frontend | 前端
- [Streamlit](https://streamlit.io/) Interactive web interface | 交互式界面

### Deep Learning Models | 深度学习模型
- Multi-Layer Perceptron (MLP) | 多层感知器
- Bidirectional Long Short-Term Memory (BiLSTM) | 双向长短期记忆网络

### Optimization Methods | 优化方法
- Early Stopping | 早停机制
- Adaptive Learning Rate Scheduling | 学习率自适应调节
- Simulated Annealing for enzyme combination optimization | 模拟退火算法（酶组合优化）

---

## Usage | 使用方法

### Requirements | 环境要求
- Python 3.11+
- See `environment.yml` for full dependencies | 依赖库详见 `environment.yml`

### Quick Start | 快速开始

```bash
# 1. Clone the repository | 克隆仓库
git clone https://github.com/yourname/PRECISE.git
cd PRECISE

# 2. Create and activate environment | 创建并激活环境
conda env create -f environment.yml
conda activate webcreat

# 3. Launch the application | 启动应用
streamlit run streamlit_main_v5fixerro1.py
```

---

## Timeline | 开发时间线

| Version 版本 | Date 日期 | Changes 更新内容 |
|---|---|---|
| v1.0.0 | 2025 | Initial release 初始版本发布 |

---

## Citation | 引用

> This work has been submitted to *Food Chemistry* and is currently under revision. Citation information including DOI will be added upon acceptance.
>
> 本研究已投稿至 *Food Chemistry*，目前处于修订状态。引用格式及 DOI 将在接收后补充。

---

## Author | 作者

**Zhong Jun** — Main Developer | 主要开发者

Acknowledgements | 致谢
Special thanks to Lv Shangyu for the support and assistance provided throughout the development of this project.

特别感谢 Lv Shangyu 在本项目开发过程中提供的支持与帮助。
---

## License | 开源协议

This project is open for academic and non-commercial use. Please cite the associated publication upon acceptance.

本项目开放用于学术及非商业用途，请在论文接收后引用相关文献。
