import streamlit as st  
import io  
from rdkit import Chem  
from rdkit.Chem import Draw  
from typing import List, Optional, Union  
import os  

class PeptideStructureVisualizer:  
    def __init__(self):  
        """  
        初始化肽序列结构可视化工具  
        """  
        self.supported_formats = ['png', 'pdf', 'svg']  
        # 标准氨基酸字母  
        self.valid_amino_acids = set('ACDEFGHIKLMNPQRSTVWY')  
    
    def is_valid_sequence(self, sequence: str) -> bool:  
        """  
        验证序列是否只包含标准氨基酸  
        """  
        return all(aa.upper() in self.valid_amino_acids for aa in sequence)  
    
    def visualize_peptide(  
        self,   
        sequence: str,   
        img_size: tuple = (400, 400)  
    ) -> Union[dict, None]:  
        """  
        可视化单个肽段分子结构  
        
        Returns:  
            dict: 包含序列、图像二进制数据的字典  
            None: 如果创建分子对象失败  
        """  
        # 转换为大写并去除空白  
        sequence = sequence.upper().replace(' ', '')  
        
        # 验证序列  
        if not self.is_valid_sequence(sequence):  
            st.error(f"序列 {sequence} 包含非标准氨基酸")  
            return None  
        
        try:  
            # 从序列创建分子对象  
            mol = Chem.MolFromSequence(sequence)  
            
            if mol is None:  
                st.error(f"无法从序列 {sequence} 创建分子对象")  
                return None  
            
            # 生成图像  
            img = Draw.MolToImage(mol, size=img_size)  
            
            # 将图像转换为可以在Streamlit中显示的格式  
            buffered = io.BytesIO()  
            img.save(buffered, format="PNG")  
            
            return {  
                "sequence": sequence,  
                "image_bytes": buffered.getvalue()  
            }  
        
        except Exception as e:  
            st.error(f"分子结构可视化出错: {e}")  
            return None  

def main():  
    st.set_page_config(page_title="肽段分子结构可视化", layout="wide")  
    st.title("🧬 肽段分子结构可视化")  
    
    # 创建可视化工具实例  
    visualizer = PeptideStructureVisualizer()  
    
    # 多肽序列输入  
    st.subheader("多肽序列输入")  
    sequences_input = st.text_area(  
        "请输入多个肽段序列（每行一个）",   
        "ACDEFG\nPQRSTVW\nHIJKLMN"  
    )  
    
    # 处理输入的序列  
    sequences = [seq.strip() for seq in sequences_input.split('\n') if seq.strip()]  
    
    # 图像尺寸调整  
    col1, col2 = st.columns(2)  
    with col1:  
        width = st.number_input("图像宽度", min_value=100, max_value=1000, value=400)  
    with col2:  
        height = st.number_input("图像高度", min_value=100, max_value=1000, value=400)  
    
    # 展示按钮  
    if st.button("展示分子结构"):  
        # 创建列  
        cols = st.columns(3)  
        
        # 展示图像  
        for i, sequence in enumerate(sequences):  
            col = cols[i % 3]  
            with col:  
                img_data = visualizer.visualize_peptide(sequence, img_size=(width, height))  
                if img_data:  
                    st.image(img_data['image_bytes'], caption=f"序列: {sequence}")  

    # 保存选项  
    st.subheader("图像保存选项")  
    save_sequence = st.selectbox("选择要保存的序列", sequences)  
    
    # 选择保存格式  
    save_format = st.selectbox("选择保存格式", ['PNG', 'SVG', 'PDF'])  
    
    # 保存按钮  
    if st.button("保存分子结构"):  
        img_data = visualizer.visualize_peptide(save_sequence, img_size=(width, height))  
        if img_data:  
            # 根据不同格式处理  
            if save_format == 'PNG':  
                file_ext = 'png'  
                mime_type = 'image/png'  
            elif save_format == 'SVG':  
                file_ext = 'svg'  
                mime_type = 'image/svg+xml'  
            else:  
                file_ext = 'pdf'  
                mime_type = 'application/pdf'  
            
            # 创建下载按钮  
            st.download_button(  
                label=f"保存 {save_sequence} 的分子结构",  
                data=img_data['image_bytes'],  
                file_name=f"{save_sequence}_molecule.{file_ext}",  
                mime=mime_type  
            )  

    # 侧边栏  
    st.sidebar.title("系统操作")  
    if st.sidebar.button("😴 关闭应用"):  
        st.sidebar.warning("应用即将关闭...")  
        # 在Streamlit中，直接使用st.stop()来停止执行  
        st.stop()  
        
if __name__ == "__main__":  
    main()  