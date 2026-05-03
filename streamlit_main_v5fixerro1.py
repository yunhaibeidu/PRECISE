import sys
import asyncio
import os
import pandas as pd 
#from rdkit import Chem 
#from rdkit.Chem import Draw 

from pepline_main1 import PeptideAnalysisPlatform
import streamlit as st

import re
# 导入序列
import torch
from enzyme_optimizer_pytorch import optimize_enzyme_combination
from enzyme_optimizer_pytorch import export_optimization_results
from enzyme_optimizer_pytorch import generate_pdf_report
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

import signal
import types
import altair as alt


# 禁用Streamlit的文件监视器
os.environ["STREAMLIT_SERVER_WATCH_DIRS"] = "false"
os.environ["STREAMLIT_SERVER_ENABLE_STATIC_SERVING"] = "false"

# PyTorch环境变量
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
# 确保有事件循环
try:
    loop = asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# 添加这个函数来解决torch.classes.__path__的问题

def _patch_torch_classes():
    import torch
    if not hasattr(torch.classes, '__path__') or not isinstance(torch.classes.__path__, types.SimpleNamespace):
        torch.classes.__path__ = types.SimpleNamespace()
    if not hasattr(torch.classes.__path__, '_path'):
        torch.classes.__path__._path = []

# 应用补丁
_patch_torch_classes()


sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))


# 页面配置 
st.set_page_config( 
    page_title="肽段分析平台 v3.2",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)



def shutdown_app():
    """关闭应用的执行函数"""
    os.kill(os.getpid(), signal.SIGTERM)  # 发送终止信号

# ====================== 页面布局 ======================
# 在页面右上角添加关闭按钮（使用列布局定位）
cols = st.columns([0.8, 0.2])
with cols[1]:
    if st.button("⛔ 一键关闭", type="secondary", help="完全关闭当前应用"):
        shutdown_app()
# 初始化平台 
@st.cache_resource  
def init_platform():
    return PeptideAnalysisPlatform()
 
platform = init_platform()

# 页面标题 
st.title(f" 智能肽段分析系统（2025版）")
st.markdown("---") 
 
# 主功能标签页 
tab1, tab2, tab3 = st.tabs(["🔪  酶切模拟", "🔮 活性预测", "🎨 最优酶组合筛选"])
 
# ====================== 酶切模拟模块 ======================
with tab1:
    st.subheader(" 蛋白质虚拟酶切模拟器")
    col1, col2 = st.columns([3,  1])
    
    with col1:
        protein_seq = st.text_area( 
            "输入蛋白质序列（单字母缩写）",
            height=200,
            placeholder="例: MALWMRLLPLLALLALW...",
            key="protein_input"
        ).upper()
    
    with col2:
        st.markdown("###  酶切参数")
        enzyme_options = platform.available_enzymes  
        selected_enzymes = st.multiselect( 
            "选择蛋白酶",
            options=enzyme_options,
            default=["Trypsin"],
            help="支持多选酶切组合"
        )
        
        min_length = st.slider(" 最小肽段长度", 1, 20, 5)
        max_length = st.slider(" 最大肽段长度", 5, 100, 50)
        
        if st.button(" 执行酶切", use_container_width=True):
            try:
                with st.spinner(" 酶切处理中..."):
                    digest_result = platform.enzyme_digestion( 
                        protein_seq,
                        selected_enzymes 
                    )
                    
                    # 过滤肽段长度 
                    filtered_peptides = [
                        p for p in digest_result['peptides'] 
                        if min_length <= len(p) <= max_length 
                    ]
                    
                    # 保存到会话状态 
                    st.session_state.digest_result  = {
                        "enzymes": selected_enzymes,
                        "peptides": filtered_peptides,
                        "stats": digest_result['peptide_statistics']
                    }
                    
            except Exception as e:
                st.error(f" 酶切失败: {str(e)}")
 
    # 显示结果 
    if 'digest_result' in st.session_state: 
        st.success(" 酶切完成！")
        result = st.session_state.digest_result  
        
        # 统计信息 
        with st.expander("📊  统计摘要", expanded=True):
            col_stat1, col_stat2, col_stat3 = st.columns(3) 
            with col_stat1:
                st.metric(" 总肽段数", result['stats']['total_peptides'])
            with col_stat2:
                st.metric(" 平均长度", f"{result['stats']['average_peptide_length']:.1f} aa")
            with col_stat3:
                st.metric(" 最长肽段", f"{result['stats']['max_peptide_length']} aa")
        
        # 肽段列表 
        st.markdown("###  酶切产物")
        df = pd.DataFrame({
            "序号": range(1, len(result['peptides'])+1),
            "肽段序列": result['peptides'],
            "长度": [len(p) for p in result['peptides']]
        })
        st.dataframe( 
            df,
            column_config={
                "序号": st.column_config.NumberColumn(width="small"), 
                "肽段序列": st.column_config.TextColumn(width="large"), 
                "长度": st.column_config.ProgressColumn( 
                    format="%d aa",
                    min_value=min_length,
                    max_value=max_length 
                )
            },
            hide_index=True,
            use_container_width=True 
        )
 
# ====================== 活性预测模块 ====================== 
with tab2:
    st.subheader(" 肽段活性预测")
    input_mode = st.radio( 
        "输入来源",
        ["手动输入", "使用酶切结果"],
        horizontal=True,
        help="可手动输入序列或使用之前的酶切产物"
    )
    
    peptides = []
    if input_mode == "手动输入":
        seq_input = st.text_area( 
            "输入肽段序列（多序列用逗号分隔,每个肽段不应超过50个氨基酸）",
            height=100,
            placeholder="例: ACDEFG, HIJKLMN...",
            key="manual_input"
        )
        peptides = [s.strip().upper() for s in seq_input.split(",")  if s.strip()] 
    else:
        if 'digest_result' in st.session_state: 
            peptides = st.session_state.digest_result['peptides'] 
            st.info(f" 已加载酶切产物中的 {len(peptides)} 个肽段")
        else:
            st.warning(" 请先完成酶切模拟获取数据")

    # 过滤掉长度超过50的肽段
    long_peptides = [p for p in peptides if len(p) > 50]
    peptides = [p for p in peptides if len(p) <= 50]

    # 如果有被过滤的肽段，进行提示
    if long_peptides:
        st.warning(
            f"检测到 {len(long_peptides)} 个肽段超过50个氨基酸，已自动忽略。\n\n"
            f"被忽略的肽段如下：\n" + "\n".join(long_peptides)
        )
    
    # 活性类型选择 
    activity_options = platform.predictor.available_activities  
    selected_activities = st.multiselect( 
        "选择预测活性",
        options=activity_options,
        default=["ACE inhibitor", "antibacterial"],
        help="支持多选活性类型"
    )
    
    if st.button(" 开始预测", use_container_width=True) and peptides:
        try:
            with st.spinner(" 正在调用深度学习模型..."):
                predictions = platform.predictor.predict( 
                    peptides,
                    selected_activities 
                )
                #st.write(f"预测结果: {predictions}")

                # 构建结果表格 
                results = []
                for idx, (peptide, pred) in enumerate(zip(peptides, predictions)):
                    row = {"序号": idx+1, "肽段序列": peptide, "长度": len(peptide)}
                
    
                    if isinstance(pred, str):
                        for activity, prob in predictions.items():
                            row[activity] = f"{prob * 100:.1f}"
                    elif isinstance(pred, dict):
                        for activity, prob in pred.items():
                            row[activity] = f"{prob * 100:.1f}"
                #else:
                    #st.warning(f"未识别的预测结果格式：{pred}")
    
                    results.append(row)

                
                df_pred = pd.DataFrame(results)
                st.session_state.prediction_results  = df_pred 
        
        except Exception as e:
            st.error(f" 预测失败: {str(e)}")
    
    # 显示预测结果 
    if 'prediction_results' in st.session_state: 
        st.success(" 预测完成！")
        df = st.session_state.prediction_results  
        
        # 交互式表格 
        edited_df = st.data_editor( 
            df,
            column_config={
                "序号": st.column_config.NumberColumn(width="small"), 
                "肽段序列": st.column_config.TextColumn(width="large"), 
                "长度": st.column_config.NumberColumn(width="small"), 
                **{act: st.column_config.ProgressColumn( 
                    label=act,
                    format="%.1f%%",
                    min_value=0.0,
                    max_value=100.0 
                ) for act in selected_activities}
            },
            hide_index=True,
            use_container_width=True 
        )
        
        # 结果下载 
        csv = edited_df.to_csv(index=False).encode('utf-8') 
        st.download_button( 
            "导出预测结果",
            data=csv,
            file_name=f"activity_pred_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv", 
            mime="text/csv",
            use_container_width=True 
        )
 
# ====================== 分子可视化模块 ======================
with tab3: 

 
    # 初始化页面配置  
    st.title("🧬  虚拟酶切酶组合优化系统")   
    
 
# ====================== 参数配置 ======================  
    with st.container(border=True):   
        st.markdown(" ⚙️ 参数配置") 

        # 使用列布局组织参数
        col_param1, col_param2 = st.columns(2)

        with col_param1:
    # 酶选择器  
            available_enzymes = platform.available_enzymes  
            selected_enzymes = st.multiselect(   
                "选择蛋白酶:",  
                options=available_enzymes,  
                default=['Trypsin', 'Pepsin_1'],  
                help="支持多选，至少选择一种酶"  
            )  
 
    # 活性类型选择  
            activity_options = platform.predictor.available_activities
            target_activity = st.selectbox(   
                "目标活性类型:",  
                options=activity_options,  
                index=0  
            )  

        activity_thresholds = {
            "ACE inhibitor": 0.76,
            "antibacterial": 0.09,
            "antioxidative": 0.66,
            "dipeptidyl peptidase IV inhibitor": 0.71
        }
 
        with col_param2:
            # 算法参数
            with st.expander("高级算法参数", expanded=True):
                max_iter = st.slider(" 最大迭代次数:", 100, 2000, 500, step=100)  
                initial_temp = st.number_input(" 初始温度:", 50.0, 500.0, 100.0, step=10.0)  
                cooling_rate = st.slider(" 冷却速率:", 0.8, 0.99, 0.95, step=0.01) 
      
            # 硬件选择  
            device = st.radio(   
                "计算设备:",  
                ["cpu", "cuda"] if torch.cuda.is_available()  else ["cpu"],  
                index=0 if not torch.cuda.is_available()  else 1  
            )  
 
# ====================== 主内容区 ======================  
     
 
# ---------------------- 蛋白质序列输入 ----------------------  
    protein_seq = st.text_area(   
        "输入蛋白质序列 (单字母氨基酸代码):",  
        value="MKVLWAALLVTFLAGCQAKVEQAVETEPEPELRQQTEWQSGQRWELALGRFWDYLRWVQTLSEQVQEELLSSQVTQELRALMDETMKELKAYKSELEEQLTPVAEETRARLSKELQAAQARLGADVLASHGRLVQYRGEVQAMLGQSTEELRVRLASHLRKLRKRLLRDADDLQKRLAVYQAGAREGAERGLSAIRERLGPLVEQGRVRAATVGSLAGQPLQERAQAWGERLRARMEEMGSRTRDRLDEVKEQVAEVRAKLEEQAQQRLEEQLGMDTQKEIMDLQARKASIRAQDVHEPSEWRNRLLLLETQAGEGN",  
        height=200,  
        help="仅支持标准20种氨基酸的大写字母（ACDEFGHIKLMNPQRSTVWY）"  
    )  
    protein_name = st.text_area("输入蛋白质名称:")
 
# ---------------------- 验证与启动 ----------------------  
    if st.button("🚀  启动优化", type="primary", use_container_width=True):  
    # 输入校验  
        if not selected_enzymes:  
            st.error(" 请至少选择一种蛋白酶！")  
            st.stop()   
 
        if not re.match("^[ACDEFGHIKLMNPQRSTVWY]+$",  protein_seq):  
            st.error(" 蛋白质序列包含非法字符！")  
            st.stop() 

        if not protein_name.strip():
            protein_name = "unknown_protein"  
 
    # 执行优化  
        with st.spinner("⚡  正在执行模拟退火优化..."):  
            progress_bar = st.progress(0)   
            result = optimize_enzyme_combination(  
                protein_sequence=protein_seq,  
                target_activity=target_activity,  
                available_enzymes=selected_enzymes,  
                max_iterations=max_iter,  
                initial_temp=initial_temp,  
                cooling_rate=cooling_rate,  
                device=device, 
                activity_threshold = activity_thresholds.get(target_activity, 1.0)
                #progress_callback=lambda x: progress_bar.progress(x)   
            )  
            # Export results to Excel and PDF  
            excel_name = f"{protein_name}_{target_activity}.xlsx"
            pdf_name = f"{protein_name}_{target_activity}.pdf"
            excel_path, pdf_path = export_optimization_results(  
                result=result,  
                protein_sequence=protein_seq,
                output_excel=excel_name,
                output_pdf=pdf_name 
            ) 

            # 将结果存入 session_state
            st.session_state['optimization_result'] = result
            st.session_state['protein_seq'] = protein_seq
 
        # 结果可视化
    if 'optimization_result' in st.session_state:
        result = st.session_state['optimization_result']
        protein_seq = st.session_state['protein_seq']
        analysis = result["analysis"]

        threshold = activity_thresholds[target_activity]
        summary_text = (
            f"• 蛋白质序列长度: {len(protein_seq)} 个氨基酸\n"
            f"• 最佳适应度: {result['best_fitness']:.4f}\n"
            f"• 最优酶组合: {', '.join(result['best_enzymes'])}\n"
            f"• 总生成肽段数: {analysis['peptide_count']}\n"
            f"• 有效肽段数 (长度≥2): {analysis['valid_peptide_count']}\n"
            f"• 高活性肽段数 (活性≥{threshold}): {analysis['high_activity_count']} "
            f"({analysis['high_activity_percentage']:.1f}%)\n"
            f"• 平均预测活性值: {analysis['activity_stats']['mean']:.4f}\n"
            f"• 优化耗时: {result['optimization_time']:.2f} 秒"
        )

        st.markdown("### 📄 优化摘要统计")
        st.code(summary_text, language="markdown")
        
        st.subheader("📊  优化结果分析")  
        col1, col2 = st.columns([0.3,  0.7])  
 
        with col1:  
            st.metric(" 最佳适应度", f"{result['best_fitness']:.4f}")  
            st.metric(" 生成肽段总数", result['analysis']['peptide_count'])  
            st.metric(" 高活性肽占比",  
                     f"{result['analysis']['high_activity_percentage']:.1f}%")  
 
        with col2:  
            st.line_chart(   
                pd.DataFrame(result['fitness_history'], columns=["适应度"]),  
                use_container_width=True  
            )  
 
        # 展示高活性肽段  
        st.subheader("🔬  高活性肽段列表")  
        high_activity_df = pd.DataFrame(  
            result["analysis"]["high_activity_peptides"],  
            columns=["肽段序列", "预测活性值"]  
        )  
        high_activity_df["是否超过阈值"] = high_activity_df.apply(
            lambda row: "✅ 是" if row["预测活性值"] >= activity_thresholds.get(target_activity, 1.0) else "❌ 否",
            axis=1
        )
        st.dataframe(   
            high_activity_df,  
            column_config={  
                "预测活性值": st.column_config.ProgressColumn(   
                    format="%.3f",  
                    min_value=0,  
                    max_value=1.0  
                )  
            },  
            hide_index=True  
        )  

        threshold = activity_thresholds[target_activity]
        labels = ['有活性', '无活性']
        values = [
            analysis['high_activity_count'],
            analysis['valid_peptide_count'] - analysis['high_activity_count']
        ]
        
        pie_data = pd.DataFrame({
            'Category': labels,
            'Count': values
        })

        pie_chart = alt.Chart(pie_data).mark_arc().encode(
            theta='Count:Q',
            color=alt.Color('Category:N', 
                          scale=alt.Scale(range=['#006BA2', '#FF800E']),
                          legend=alt.Legend(title="活性状态")),
            tooltip=['Category', 'Count']
        ).properties(
            width=200,
            height=200,
            title=f'活性阈值: {threshold}'
        )
        
        st.altair_chart(pie_chart, use_container_width=True)

        if "score_distribution" in analysis:
            st.subheader("📈 活性得分分布")
            
            # 获取当前阈值
            current_threshold = activity_thresholds.get(target_activity, 1.0)
            
            # 准备数据（保持不变）
            score_ranges = list(analysis["score_distribution"].keys())
            counts = list(analysis["score_distribution"].values())
            
            df = pd.DataFrame({
                "score_range": score_ranges,
                "count": counts,
                "min_score": [float(x.split("-")[0]) for x in score_ranges],
                "max_score": [float(x.split("-")[1]) for x in score_ranges]
            })
            
            # 颜色计算（保持不变）
            df["color"] = df.apply(lambda row: 
                "#2ca02c" if row["min_score"] >= current_threshold else
                "#ff7f0e" if row["min_score"] <= current_threshold <= row["max_score"] else
                "#1f77b4", 
                axis=1
            )
            
            # 柱状图（保持不变）
            base = alt.Chart(df).encode(
                x=alt.X('score_range:N', 
                    sort=alt.EncodingSortField(field='min_score', order='ascending'),
                    title="活性区间")
            )
            
            bars = base.mark_bar().encode(
                y=alt.Y('count:Q', title="肽段数"),
                color=alt.Color('color:N', scale=None),
                tooltip=[
                    alt.Tooltip('score_range:N', title="活性区间"),
                    alt.Tooltip('count:Q', title="肽段数"),
                    alt.Tooltip('min_score:Q', format=".2f", title="区间最小值"),
                    alt.Tooltip('max_score:Q', format=".2f", title="区间最大值")
                ]
            )
            
            # 修改阈值线部分：添加x轴范围限制
            threshold_line = alt.Chart(pd.DataFrame({'threshold': [current_threshold]}))\
                .mark_rule(color='red', strokeWidth=2, strokeDash=[5,5])\
                .encode(
                    x=alt.X('threshold:Q', 
                            scale=alt.Scale(domain=[0, 1]),  # 固定x轴范围为0-1
                            title="活性阈值")
                )
            
            # 组合图表（保持不变）
            chart = alt.layer(
                bars,
                threshold_line
            ).resolve_scale(
                x='independent'
            ).properties(
                width=700,
                height=400
            )
            
            # 图例（保持不变）
            legend_df = pd.DataFrame({
                "type": ["超过阈值区间", "阈值交界区间", "低活性区间"],
                "color": ["#2ca02c", "#ff7f0e", "#1f77b4"]
            })
            legend = alt.Chart(legend_df).mark_rect(size=50).encode(
                y=alt.Y('type:N', axis=alt.Axis(orient='right', title=None)),
                color=alt.Color('color:N', scale=None, legend=None)
            )
            
            st.altair_chart(alt.hconcat(chart, legend))

        if "length_distribution" in analysis:
            st.subheader("肽段长度分布")
            length_df = pd.DataFrame({
                "肽段长度": list(analysis["length_distribution"].keys()),
                "数量": list(analysis["length_distribution"].values())
            })
            st.bar_chart(length_df.set_index("肽段长度"))

        if "aa_composition" in analysis and analysis["aa_composition"]:
            st.subheader("氨基酸组成比例")
            aa_df = pd.DataFrame({
                "氨基酸": list(analysis["aa_composition"].keys()),
                "占比": list(analysis["aa_composition"].values())
            }).sort_values(by="占比", ascending=False)
            st.bar_chart(aa_df.set_index("氨基酸"))

        if "best_enzymes" in result:
            st.subheader("最优酶组合")
            st.markdown("、".join(result["best_enzymes"]))
 

# ====================== 安全验证 ======================
# 在关闭前添加确认对话框
if "confirm_shutdown" not in st.session_state:
    st.session_state.confirm_shutdown = False

if st.session_state.confirm_shutdown:
    st.warning("确认要关闭应用吗？此操作不可逆！")
    if st.button("✅ 确认关闭"):
        shutdown_app()
    if st.button("❌ 取消关闭"):
        st.session_state.confirm_shutdown = False

st.markdown("---") 
st.caption(f" 系统时间：2025年5月24日 10:32 | 平台版本：v3.2.2025")
