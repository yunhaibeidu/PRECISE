# PRECISE
一个本人自行训练及集成其它方法的蛋白酶切和功能肽预测平台，额外提供酶组合优化

# PRECISE: PRotein Enzymatic Cleavage and In-Silico Evaluation  

## 项目简介  
PRECISE 是一个综合性蛋白质分析平台，专注于蛋白酶切模拟、功能肽预测和酶切组合优化。该平台通过深度学习模型和启发式算法，实现高效的功能肽筛选和酶切策略设计。  

## 核心功能  
- **蛋白虚拟酶切**：基于正则化方法的精确蛋白质酶切模拟  
- **功能肽预测**：预测四种关键活性功能：  
  - 抗氧化活性  
  - ACE抑制活性  
  - 抗菌活性  
  - DPPIV抑制活性  
- **肽分子结构可视化**：提供肽序列的分子结构图下载功能  
- **酶组合优化**：基于模拟退火算法，针对特定活性优化酶切组合，提高目标活性肽的产率  

## 技术实现  
- **前端**：Streamlit 交互式界面  
- **深度学习模型**：  
  - 多层感知器 (MLP)  
  - 双向长短期记忆网络 (BiLSTM)  
- **优化方法**：  
  - 早停机制 (Early Stopping)  
  - 学习率自适应调节  
  - 模拟退火算法 (用于酶组合优化)  

#快速开始
# 1. 克隆项目
git clone https://github.com/你的用户名/你的仓库名.git
cd 你的仓库名

# 2. 创建并激活环境
conda env create -f webcreat.yaml
conda activate webcreat

# 3. 启动应用
streamlit run streamlit_main_v5fixerro1.py
- 依赖库详见 requirements.txt  

### 快速开始  
