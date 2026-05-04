
import sys
import asyncio
import os
import pandas as pd 

from pepline_main1 import PeptideAnalysisPlatform
import streamlit as st

import re
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

def _patch_torch_classes():
    import torch
    if not hasattr(torch.classes, '__path__') or not isinstance(torch.classes.__path__, types.SimpleNamespace):
        torch.classes.__path__ = types.SimpleNamespace()
    if not hasattr(torch.classes.__path__, '_path'):
        torch.classes.__path__._path = []

_patch_torch_classes()

# ✅ 修复1：将硬编码绝对路径改为相对路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# ✅ 修复2：指定输出目录，避免文件输出到不确定位置
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 页面配置 
st.set_page_config( 
    page_title="Peptide analysis platform - PRECISE",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ✅ 修复3：Windows兼容的关闭函数
def shutdown_app():
    """关闭应用的执行函数"""
    if sys.platform == "win32":
        os.kill(os.getpid(), signal.SIGINT)
    else:
        os.kill(os.getpid(), signal.SIGTERM)

# ====================== 页面布局 ======================
cols = st.columns([0.8, 0.2])
with cols[1]:
    if st.button("⛔ One-click close", type="secondary", help="Completely close the current application"):
        shutdown_app()

# 初始化平台 
@st.cache_resource  
def init_platform():
    return PeptideAnalysisPlatform()
 
platform = init_platform()

# 页面标题 
st.title(" Intelligent Peptide Analysis System (2025 Edition)")
st.markdown("---") 
 
# 主功能标签页 
tab1, tab2, tab3 = st.tabs(["🔪  Enzyme digestion simulation", "🔮 Activity prediction", "🎨 Optimal enzyme combination screening"])
 
# ====================== 酶切模拟模块 ======================
with tab1:
    st.subheader(" Protein virtual enzyme digestion simulator")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        protein_seq = st.text_area( 
            "Input the protein sequence (single-letter abbreviation)",
            height=200,
            placeholder="Eg: MALWMRLLPLLALLALW...",
            key="protein_input"
        ).upper()
    
    with col2:
        st.markdown("### Enzyme digestion parameters")
        enzyme_options = platform.available_enzymes  
        selected_enzymes = st.multiselect( 
            "Selective proteases",
            options=enzyme_options,
            default=["Trypsin"],
            help="Support multiple enzymatic digestion combinations"
        )
        
        min_length = st.slider("Minimum peptide length", 1, 20, 5)
        max_length = st.slider("Maximum peptide length", 5, 100, 50)
        
        if st.button("Perform enzymatic digestion", use_container_width=True):
            try:
                with st.spinner("In the process of enzymatic digestion..."):
                    digest_result = platform.enzyme_digestion( 
                        protein_seq,
                        selected_enzymes 
                    )
                    
                    filtered_peptides = [
                        p for p in digest_result['peptides'] 
                        if min_length <= len(p) <= max_length 
                    ]
                    
                    st.session_state.digest_result = {
                        "enzymes": selected_enzymes,
                        "peptides": filtered_peptides,
                        "stats": digest_result['peptide_statistics']
                    }
                    
            except Exception as e:
                st.error(f"Enzyme digestion failure: {str(e)}")
 
    if 'digest_result' in st.session_state: 
        st.success("Enzymatic digestion completed!")
        result = st.session_state.digest_result  
        
        with st.expander("📊 Statistical summary", expanded=True):
            col_stat1, col_stat2, col_stat3 = st.columns(3) 
            with col_stat1:
                st.metric("Total number of peptide segments", result['stats']['total_peptides'])
            with col_stat2:
                st.metric("Average length", f"{result['stats']['average_peptide_length']:.1f} aa")
            with col_stat3:
                st.metric("Longest peptide segment", f"{result['stats']['max_peptide_length']} aa")
        
        st.markdown("### Enzyme digestion products")
        df = pd.DataFrame({
            "Serial number": range(1, len(result['peptides'])+1),
            "Peptide sequence": result['peptides'],
            "Length": [len(p) for p in result['peptides']]
        })
        st.dataframe( 
            df,
            column_config={
                "Serial number": st.column_config.NumberColumn(width="small"), 
                "Peptide sequence": st.column_config.TextColumn(width="large"), 
                "Length": st.column_config.ProgressColumn( 
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
    st.subheader("Peptide activity prediction")
    input_mode = st.radio( 
        "Input source",
        ["Manual input", "Use the enzymatic digestion results"],
        horizontal=True,
        help="The sequence can be manually entered or the previous enzymatic digestion products can be used"
    )
    
    peptides = []
    if input_mode == "Manual input":
        seq_input = st.text_area( 
            "Input the peptide sequence (separate multiple sequences with commas, and each peptide should not exceed 50 amino acids)",
            height=100,
            placeholder="Eg: ACDEFG, HIJKLMN...",
            key="manual_input"
        )
        peptides = [s.strip().upper() for s in seq_input.split(",") if s.strip()] 
    else:
        if 'digest_result' in st.session_state: 
            peptides = st.session_state.digest_result['peptides'] 
            st.info(f"The enzyme digestion products have been loaded: {len(peptides)} peptide segments")
        else:
            st.warning("Please complete the enzymatic digestion simulation to obtain the data first")

    long_peptides = [p for p in peptides if len(p) > 50]
    peptides = [p for p in peptides if len(p) <= 50]

    if long_peptides:
        st.warning(
            f"{len(long_peptides)} peptide segments exceeding 50 amino acids have been detected and have been automatically ignored.\n\n"
            f"The overlooked peptides are as follows:\n" + "\n".join(long_peptides)
        )
    
    activity_options = platform.predictor.available_activities  
    selected_activities = st.multiselect( 
        "Select predictive activity types",
        options=activity_options,
        default=["ACE inhibitor", "antibacterial"],
        help="Support multiple active types"
    )
    
    if st.button("Start the prediction", use_container_width=True) and peptides:
        try:
            with st.spinner("The deep learning model is being invoked..."):
                predictions = platform.predictor.predict( 
                    peptides,
                    selected_activities 
                )

                results = []
                for idx, (peptide, pred) in enumerate(zip(peptides, predictions)):
                    row = {"Serial number": idx+1, "Peptide sequence": peptide, "Length": len(peptide)}
    
                    if isinstance(pred, str):
                        for activity, prob in predictions.items():
                            row[activity] = f"{prob * 100:.1f}"
                    elif isinstance(pred, dict):
                        for activity, prob in pred.items():
                            row[activity] = f"{prob * 100:.1f}"
    
                    results.append(row)

                df_pred = pd.DataFrame(results)
                st.session_state.prediction_results = df_pred 
        
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
    
    if 'prediction_results' in st.session_state: 
        st.success("The prediction is complete!")
        df = st.session_state.prediction_results  
        
        edited_df = st.data_editor( 
            df,
            column_config={
                # ✅ 修复4：column_config key改为正确的列名"Serial number"
                "Serial number": st.column_config.NumberColumn(width="small"), 
                "Peptide sequence": st.column_config.TextColumn(width="large"), 
                "Length": st.column_config.NumberColumn(width="small"), 
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
        
        csv = edited_df.to_csv(index=False).encode('utf-8') 
        st.download_button( 
            "Export the prediction results",
            data=csv,
            file_name=f"activity_pred_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv", 
            mime="text/csv",
            use_container_width=True 
        )
 
# ====================== 酶组合优化模块 ======================
with tab3: 
    st.title("🧬 Virtual enzyme digestion enzyme combination optimization system")   
    
    with st.container(border=True):   
        st.markdown("⚙️ Parameter configuration") 

        col_param1, col_param2 = st.columns(2)

        with col_param1:
            available_enzymes = platform.available_enzymes  
            selected_enzymes = st.multiselect(   
                "Selective protease:",  
                options=available_enzymes,  
                default=['Trypsin', 'Pepsin_1'],  
                help="Support multiple selections. At least one enzyme should be chosen"  
            )  
 
            activity_options = platform.predictor.available_activities
            target_activity = st.selectbox(   
                "Target activity type:",  
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
            with st.expander("Advanced algorithm parameters", expanded=True):
                max_iter = st.slider("Maximum number of iterations:", 100, 2000, 500, step=100)  
                # ✅ 修复5：label重复问题，改为正确描述
                initial_temp = st.number_input("Initial temperature:", 50.0, 500.0, 100.0, step=10.0)  
                cooling_rate = st.slider("Cooling rate:", 0.8, 0.99, 0.95, step=0.01) 
      
            device = st.radio(   
                "Computing equipment:",  
                ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"],  
                index=0 if not torch.cuda.is_available() else 1  
            )  
 
    protein_seq = st.text_area(   
        "Enter the protein sequence (single-letter amino acid code):",  
        value="MKVLWAALLVTFLAGCQAKVEQAVETEPEPELRQQTEWQSGQRWELALGRFWDYLRWVQTLSEQVQEELLSSQVTQELRALMDETMKELKAYKSELEEQLTPVAEETRARLSKELQAAQARLGADVLASHGRLVQYRGEVQAMLGQSTEELRVRLASHLRKLRKRLLRDADDLQKRLAVYQAGAREGAERGLSAIRERLGPLVEQGRVRAATVGSLAGQPLQERAQAWGERLRARMEEMGSRTRDRLDEVKEQVAEVRAKLEEQAQQRLEEQLGMDTQKEIMDLQARKASIRAQDVHEPSEWRNRLLLLETQAGEGN",  
        height=200,  
        help="Only capital letters of the standard 20 amino acids (ACDEFGHIKLMNPQRSTVWY) are supported"  
    )  
    protein_name = st.text_input("Enter the name of the protein:")
 
    if st.button("🚀 Start optimization", type="primary", use_container_width=True):  
        if not selected_enzymes:  
            st.error("Please choose at least one protease!")  
            st.stop()   
 
        if not re.match("^[ACDEFGHIKLMNPQRSTVWY]+$", protein_seq):  
            st.error("The protein sequence contains illegal characters!")  
            st.stop() 

        if not protein_name.strip():
            protein_name = "unknown_protein"  
 
        with st.spinner("⚡ Simulated annealing optimization is currently underway..."):  
            progress_bar = st.progress(0)   
            result = optimize_enzyme_combination(  
                protein_sequence=protein_seq,  
                target_activity=target_activity,  
                available_enzymes=selected_enzymes,  
                max_iterations=max_iter,  
                initial_temp=initial_temp,  
                cooling_rate=cooling_rate,  
                device=device, 
                activity_threshold=activity_thresholds.get(target_activity, 1.0)
            )  
            # ✅ 修复2：输出文件写入指定OUTPUT_DIR
            excel_name = os.path.join(OUTPUT_DIR, f"{protein_name}_{target_activity}.xlsx")
            pdf_name = os.path.join(OUTPUT_DIR, f"{protein_name}_{target_activity}.pdf")
            excel_path, pdf_path = export_optimization_results(  
                result=result,  
                protein_sequence=protein_seq,
                output_excel=excel_name,
                output_pdf=pdf_name 
            ) 

            st.session_state['optimization_result'] = result
            st.session_state['protein_seq'] = protein_seq
 
    if 'optimization_result' in st.session_state:
        result = st.session_state['optimization_result']
        protein_seq = st.session_state['protein_seq']
        analysis = result["analysis"]
        threshold = activity_thresholds[target_activity]

        # ✅ 修复6：summary_text括号和描述错误全部修正
        summary_text = (
            f"• Protein sequence length: {len(protein_seq)} amino acids\n"
            f"• Optimal fitness: {result['best_fitness']:.4f}\n"
            f"• Optimal enzyme combination: {', '.join(result['best_enzymes'])}\n"
            f"• Total number of peptide segments generated: {analysis['peptide_count']}\n"
            f"• Number of effective peptide segments (length ≥ 2): {analysis['valid_peptide_count']}\n"
            f"• Number of highly active peptide segments (activity ≥ {threshold}): {analysis['high_activity_count']} "
            f"({analysis['high_activity_percentage']:.1f}%)\n"
            f"• Mean activity score: {analysis['activity_stats']['mean']:.4f}\n"
            f"• Optimization time: {result['optimization_time']:.2f} s"
        )

        st.markdown("### 📄 Optimization summary statistics")
        st.code(summary_text, language="markdown")
        
        st.subheader("📊 Optimization results analysis")  
        col1, col2 = st.columns([0.3, 0.7])  
 
        with col1:  
            st.metric("Optimal fitness", f"{result['best_fitness']:.4f}")  
            # ✅ 修复7：两个metric标签相同问题修正
            st.metric("Total peptide segments", result['analysis']['peptide_count'])  
            st.metric("Proportion of highly active peptides",  
                     f"{result['analysis']['high_activity_percentage']:.1f}%")  
 
        with col2:  
            st.line_chart(   
                pd.DataFrame(result['fitness_history'], columns=["Fitness Value"]),  
                use_container_width=True  
            )  
 
        st.subheader("🔬 List of highly active peptides")  
        high_activity_df = pd.DataFrame(  
            result["analysis"]["high_activity_peptides"],  
            columns=["Peptide sequence", "Predicted activity value"]  
        )  
        high_activity_df["Exceeds threshold"] = high_activity_df.apply(
            lambda row: "✅ Yes" if row["Predicted activity value"] >= activity_thresholds.get(target_activity, 1.0) else "❌ No",
            axis=1
        )
        st.dataframe(   
            high_activity_df,  
            column_config={  
                "Predicted activity value": st.column_config.ProgressColumn(   
                    format="%.3f",  
                    min_value=0,  
                    max_value=1.0  
                )  
            },  
            hide_index=True  
        )  

        labels = ['Active', 'Inactive']
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
                          legend=alt.Legend(title="Active state")),
            tooltip=['Category', 'Count']
        ).properties(
            width=200,
            height=200,
            title=f'Activity threshold: {threshold}'
        )
        
        st.altair_chart(pie_chart, use_container_width=True)

        if "score_distribution" in analysis:
            st.subheader("📈 Distribution of activity scores")
            
            current_threshold = activity_thresholds.get(target_activity, 1.0)
            
            score_ranges = list(analysis["score_distribution"].keys())
            counts = list(analysis["score_distribution"].values())
            
            df = pd.DataFrame({
                "score_range": score_ranges,
                "count": counts,
                "min_score": [float(x.split("-")[0]) for x in score_ranges],
                "max_score": [float(x.split("-")[1]) for x in score_ranges]
            })
            
            df["color"] = df.apply(lambda row: 
                "#2ca02c" if row["min_score"] >= current_threshold else
                "#ff7f0e" if row["min_score"] <= current_threshold <= row["max_score"] else
                "#1f77b4", 
                axis=1
            )
            
            base = alt.Chart(df).encode(
                x=alt.X('score_range:N', 
                    sort=alt.EncodingSortField(field='min_score', order='ascending'),
                    title="Active range")
            )
            
            bars = base.mark_bar().encode(
                y=alt.Y('count:Q', title="Peptide number"),
                color=alt.Color('color:N', scale=None),
                tooltip=[
                    alt.Tooltip('score_range:N', title="Active range"),
                    alt.Tooltip('count:Q', title="Peptide number"),
                    alt.Tooltip('min_score:Q', format=".2f", title="Interval minimum value"),
                    alt.Tooltip('max_score:Q', format=".2f", title="Maximum value of the interval")
                ]
            )
            
            threshold_line = alt.Chart(pd.DataFrame({'threshold': [current_threshold]}))\
                .mark_rule(color='red', strokeWidth=2, strokeDash=[5,5])\
                .encode(
                    x=alt.X('threshold:Q', 
                            scale=alt.Scale(domain=[0, 1]),
                            title="Activity threshold")
                )
            
            chart = alt.layer(
                bars,
                threshold_line
            ).resolve_scale(
                x='independent'
            ).properties(
                width=700,
                height=400
            )
            
            legend_df = pd.DataFrame({
                "type": ["Above threshold", "Threshold boundary interval", "Below threshold"],
                "color": ["#2ca02c", "#ff7f0e", "#1f77b4"]
            })
            legend = alt.Chart(legend_df).mark_rect(size=50).encode(
                y=alt.Y('type:N', axis=alt.Axis(orient='right', title=None)),
                color=alt.Color('color:N', scale=None, legend=None)
            )
            
            st.altair_chart(alt.hconcat(chart, legend))

        if "length_distribution" in analysis:
            st.subheader("Peptide length distribution")
            length_df = pd.DataFrame({
                "Peptide length": list(analysis["length_distribution"].keys()),
                "Quantity": list(analysis["length_distribution"].values())
            })
            st.bar_chart(length_df.set_index("Peptide length"))

        if "aa_composition" in analysis and analysis["aa_composition"]:
            st.subheader("Amino acid composition ratio")
            aa_df = pd.DataFrame({
                "Amino acid": list(analysis["aa_composition"].keys()),
                "Proportion": list(analysis["aa_composition"].values())
            }).sort_values(by="Proportion", ascending=False)
            st.bar_chart(aa_df.set_index("Amino acid"))

        if "best_enzymes" in result:
            st.subheader("Optimal enzyme combination")
            st.markdown(", ".join(result["best_enzymes"]))
 
# ====================== 安全验证 ======================
if "confirm_shutdown" not in st.session_state:
    st.session_state.confirm_shutdown = False

if st.session_state.confirm_shutdown:
    st.warning("Are you sure you want to close the application? This operation is irreversible!")
    if st.button("✅ Confirm closure"):
        shutdown_app()
    if st.button("❌ Cancel"):
        st.session_state.confirm_shutdown = False

st.markdown("---") 
st.caption("Platform version: v3.2.2025")
