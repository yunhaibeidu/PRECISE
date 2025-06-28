# -*- coding: utf-8 -*-
"""
酶切处理模块
========================
本模块用于处理蛋白质序列的酶切操作，支持多种酶的选择和结果导出。
"""


import os  
import sys  
import pandas as pd  
from typing import List, Dict  

# 动态添加项目根目录到 Python 路径  
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  
sys.path.insert(0, project_root)  

# 修改导入方式  
from src.protease_rules import ENZYME_FUNCTIONS, apply_enzymes, digest_protein  


def process_protein_digestion(  
    protein_sequence: str,   
    selected_enzymes: List[str]  
) -> Dict:  
    """  
    处理蛋白质酶切的核心后台函数  
    
    Args:  
        protein_sequence (str): 输入的蛋白质序列  
        selected_enzymes (List[str]): 选择的酶种名称列表  
    
    Returns:  
        Dict: 酶切处理结果  
    """  
    # 验证序列  
    if not is_valid_sequence(protein_sequence):  
        return {  
            "error": "无效的蛋白质序列！请确保只包含标准氨基酸字母"  
        }  
    
    # 转换酶种名称为函数  
    enzyme_functions = [  
        ENZYME_FUNCTIONS[enzyme_name]   
        for enzyme_name in selected_enzymes  
    ]  
    
    # 执行酶切  
    peptides = digest_protein(protein_sequence, enzyme_functions)  
    
    # 准备结果  
    results = {  
        "original_sequence": protein_sequence,  
        "original_sequence_length": len(protein_sequence),  
        "enzymes_used": selected_enzymes,  
        "peptides": peptides,  
        "peptide_statistics": {  
            "total_peptides": len(peptides),  
            "min_peptide_length": min(len(p) for p in peptides) if peptides else 0,  
            "max_peptide_length": max(len(p) for p in peptides) if peptides else 0,  
            "average_peptide_length": sum(len(p) for p in peptides) / len(peptides) if peptides else 0  
        }  
    }  
    
    return results  

def is_valid_sequence(sequence: str) -> bool:  
    """  
    验证蛋白质序列是否有效  
    
    Args:  
        sequence (str): 待验证的序列  
    
    Returns:  
        bool: 序列是否有效  
    """  
    valid_aa = set('ACDEFGHIKLMNPQRSTVWY')  
    return all(aa in valid_aa for aa in sequence.upper())  

def export_digestion_results(  
    results: Dict,   
    output_file: str = 'digestion_results1.xlsx'  
) -> str:  
    """  
    导出酶切结果到Excel  
    
    Args:  
        results (Dict): 酶切结果  
        output_file (str): 输出文件名  
    
    Returns:  
        str: 保存的文件路径  
    """  
    # 创建肽段DataFrame  
    peptides_df = pd.DataFrame({  
        'Peptide_Sequence': results['peptides'],  
        'Peptide_Length': [len(p) for p in results['peptides']]  
    })  
    
    # 创建元数据DataFrame  
    metadata_df = pd.DataFrame({  
        'Original_Sequence': [results['original_sequence']],  
        'Sequence_Length': [results['original_sequence_length']],  
        'Enzymes_Used': [', '.join(results['enzymes_used'])],  
        'Total_Peptides': [results['peptide_statistics']['total_peptides']],  
        'Min_Peptide_Length': [results['peptide_statistics']['min_peptide_length']],  
        'Max_Peptide_Length': [results['peptide_statistics']['max_peptide_length']],  
        'Average_Peptide_Length': [results['peptide_statistics']['average_peptide_length']]  
    })  
    
    # 使用ExcelWriter保存多个工作表  
    with pd.ExcelWriter(output_file) as writer:  
        peptides_df.to_excel(writer, sheet_name='Peptides', index=False)  
        metadata_df.to_excel(writer, sheet_name='Metadata', index=False)  
    
    return output_file  

# 可用的全部酶种列表  
AVAILABLE_ENZYMES = list(ENZYME_FUNCTIONS.keys())  

# 使用示例  
if __name__ == "__main__":  
    # 示例使用  
    test_sequence = "MKVLWAALLVTFLAGCQAKVEQAVETEPEPELRQQTEWQSGQRWELALGRFWDYLRWVQTLSEQVQEELLSSQVTQELRALMDETMKELKAYKSELEEQLTPVAEETRARLSKELQAAQARLGADVLASHGRLVQYRGEVQAMLGQSTEELRVRLASHLRKLRKRLLRDADDLQKRLAVYQAGAREGAERGLSAIRERLGPLVEQGRVRAATVGSLAGQPLQERAQAWGERLRARMEEMGSRTRDRLDEVKEQVAEVRAKLEEQAQQRLEEQLGMDTQKEIMDLQARKASIRAQDVHEPSEWRNRLLLLETQAGEGN"  
    
    result = process_protein_digestion(  
        test_sequence,   
        ['Trypsin', 'Proteinase_K']  
    )  
    print(result)  
    
    # 导出结果  
    export_file = export_digestion_results(result)  
    print(f"结果已导出到: {export_file}")  
