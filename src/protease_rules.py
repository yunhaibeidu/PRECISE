# -*- coding: utf-8 -*-
"""
酶切位点识别与肽段生成
========================================

"""
from typing import List, Callable

def Trypsin(seq: str, seq_len: int) -> List[int]:
    """
    胰蛋白酶切割位点识别
    特殊规则：
    1. 在K、R后切割，但P除外
    2. 特殊情况下阻止切割
    """
    cleavage = []
    
    # 基本切割规则
    for i in range(seq_len):
        if i < seq_len - 1:
            # 基本切割：K、R后（除P外）
            if seq[i] in ['K', 'R'] and seq[i+1] != 'P':
                cleavage.append(i)
            
            # 特殊情况：WKP
            if i < seq_len - 2 and seq[i] == 'W':
                if seq[i+1] == 'K' and seq[i+2] == 'P':
                    cleavage.append(i+1)
            
            # 特殊情况：MRP
            if i < seq_len - 2 and seq[i] == 'M':
                if seq[i+1] == 'R' and seq[i+2] == 'P':
                    cleavage.append(i+1)
    
    # 阻止切割的特殊情况
    block_conditions = [
        # DC条件
        (lambda i: i < seq_len - 2 and 
         seq[i] in ['D', 'C'] and 
         seq[i+1] == 'K' and 
         seq[i+2] == 'D'),
        
        # CK特定条件
        (lambda i: i < seq_len - 2 and 
         seq[i] == 'C' and 
         seq[i+1] == 'K' and 
         seq[i+2] in ['H', 'Y']),
        
        # CRK条件
        (lambda i: i < seq_len - 2 and 
         seq[i] == 'C' and 
         seq[i+1] == 'R' and 
         seq[i+2] == 'K'),
        
        # RRH条件
        (lambda i: i < seq_len - 2 and 
         seq[i] == 'R' and 
         seq[i+1] == 'R' and 
         seq[i+2] in ['H', 'R'])
    ]
    
    # 移除被阻止的切割位点
    for condition in block_conditions:
        for i in range(seq_len):
            if condition(i) and i+1 in cleavage:
                cleavage.remove(i+1)
    
    return cleavage

def Pepsin_1(seq: str, seq_len: int) -> List[int]:
    """
    Pepsin-1酶切位点识别
    """
    cleavage = []
    
    for i in range(seq_len):
        # 在L、F后切割
        if i < seq_len - 1:
            # 在L、F后切割
            peptide_end_conditions = [
                seq[i+1] in ['L', 'F'],
                seq[i] in ['L', 'F']
            ]
            
            for condition in peptide_end_conditions:
                if condition:
                    # 先添加位点
                    if i not in cleavage:
                        cleavage.append(i)
                    
                    # 特殊阻止条件
                    block_conditions = [
                        # 前后有P
                        (i >= 2 and (seq[i-1] == 'P' or seq[i-2] == 'P')),
                        # 特定氨基酸影响
                        (i >= 2 and seq[i-2] in ['H', 'K', 'R']),
                        # 特定位置限制
                        (i == 0 and seq[i+2] == 'P'),
                        (i == seq_len - 1 and seq[i-1] == 'P'),
                        (seq[i] == 'R')
                    ]
                    
                    # 如果满足阻止条件，移除该位点
                    if any(block_conditions):
                        cleavage.remove(i)
    
    return cleavage

def Pepsin_2(seq: str, seq_len: int) -> List[int]:
    """
    Pepsin-2酶切位点识别（pH>2）
    """
    cleavage = []
    
    for i in range(seq_len):
        # 扩展切割位点
        valid_amino_acids = ['L', 'F', 'Y', 'W']
        
        if i < seq_len - 1:
            # 在特定氨基酸后切割
            peptide_end_conditions = [
                seq[i+1] in valid_amino_acids,
                seq[i] in valid_amino_acids
            ]
            
            for condition in peptide_end_conditions:
                if condition:
                    # 先添加位点
                    if i not in cleavage:
                        cleavage.append(i)
                    
                    # 特殊阻止条件
                    block_conditions = [
                        (i >= 2 and (seq[i-1] == 'P' or seq[i-2] == 'P')),
                        (i >= 2 and seq[i-2] in ['H', 'K', 'R']),
                        (i == 0 and seq[i+2] == 'P'),
                        (i == seq_len - 1 and seq[i-1] == 'P'),
                        (seq[i] == 'R')
                    ]
                    
                    # 如果满足阻止条件，移除该位点
                    if any(block_conditions):
                        cleavage.remove(i)
    
    return cleavage

def chymo_high(seq: str, seq_len: int) -> List[int]:
    """
    高特异性胰凝乳蛋白酶切位点识别
    """
    cleavage = []
    
    for i in range(seq_len):
        # 高特异性：仅限F、Y、W，且不在P前
        if i < seq_len - 1:
            amino_acids = ['F', 'Y']
            
            for aa in amino_acids:
                if seq[i] == aa and seq[i+1] != 'P':
                    cleavage.append(i)
            
            # W特殊处理
            if seq[i] == 'W' and seq[i+1] != 'P' and seq[i] != 'M':
                cleavage.append(i)
    
    return cleavage

def chymo_low(seq: str, seq_len: int) -> List[int]:
    """
    低特异性胰凝乳蛋白酶切位点识别
    """
    cleavage = []
    
    for i in range(seq_len - 1):  # 直接限制范围，避免重复判断
        # 低特异性：扩展切割位点
        specific_conditions = [
            (seq[i] in ['F', 'L', 'Y'] and seq[i+1] != 'P'),
            (seq[i] == 'W' and seq[i+1] != 'P' and seq[i] != 'M'),
            (seq[i] == 'M' and seq[i+1] != 'P' and seq[i] != 'Y'),
            (seq[i] == 'H' and seq[i+1] != 'D' and seq[i] != 'M' and 
             seq[i] != 'P' and seq[i] != 'W')
        ]
        
        # 检查是否满足任一条件
        if any(specific_conditions):
            cleavage.append(i)
    
    return cleavage

def GluC(seq: str, seq_len: int) -> List[int]:
    """
    谷氨酸内切酶切位点识别
    切割规则：
    1. 在谷氨酸(E)后切割
    2. 排除特定情况
    """
    cleavage = []
    
    for i in range(seq_len):
        if i < seq_len - 1:
            # 基本切割：E后，但不在P前
            if seq[i] == 'E' and seq[i+1] != 'P':
                cleavage.append(i)
    
    return cleavage

def AspN(seq: str, seq_len: int) -> List[int]:
    """
    天冬氨酸内切酶(AspN)切位点识别
    切割规则：
    1. 在天冬氨酸(D)前切割
    """
    cleavage = []
    
    for i in range(seq_len):
        if i < seq_len - 1:
            # 在D前切割
            if seq[i+1] == 'D':
                cleavage.append(i)
    
    return cleavage

def LysC(seq: str, seq_len: int) -> List[int]:
    """
    LysC酶切位点识别
    切割规则：
    1. 仅在赖氨酸(K)后切割
    2. 排除特定情况
    """
    cleavage = []
    
    for i in range(seq_len):
        if i < seq_len - 1:
            # 基本切割：K后，排除P和特定情况
            if seq[i] == 'K' and seq[i+1] != 'P':
                cleavage.append(i)
    
    return cleavage

def Proteinase_K(seq: str, seq_len: int) -> List[int]:  
    """  
    Proteinase K酶切位点识别  
    切割规则：根据PeptideCutter数据  
    1. 在A,E,F,I,L,T,V,W,Y后切割  
    """  
    cleavage = []  
    
    # PeptideCutter数据的切割位点  
    target_residues = ['A', 'E', 'F', 'I', 'L', 'T', 'V', 'W', 'Y']  
    
    for i in range(seq_len):  
        if i < seq_len - 1:  
            # 在特定氨基酸后切割  
            if seq[i] in target_residues:  
                cleavage.append(i)  
    
    return cleavage  

#def Papain(seq: str, seq_len: int) -> List[int]:  
  #  """  
  #  木瓜蛋白酶切割位点识别  
    
   # 基于MEROPS数据库验证的切割规则：  
   # 1. 主要在R、F、L、G后切割  
   # 2. 实验证实酶切位点可靠性（Melo et al., 2001）  
 #   """  
   # cleavage = []  
    
  #  # 确认的切割规则：在R、F、L、G后切割  
  #  target_residues = ['R', 'F', 'L', 'G']  
    
  #  for i in range(seq_len):  
  #      if i < seq_len - 1:  
   #         # 在R、F、L、G后切割  
       #     if seq[i] in target_residues:  
   #             cleavage.append(i)  
    
  #  return cleavage  

def Papain(seq: str, seq_len: int) -> List[int]:
    """
    木瓜蛋白酶切割位点识别

    基于MEROPS数据库（C01.001）验证的切割规则（已修订）：
    1. 主要P1切割位点：R（Arg）、K（Lys）、G（Gly）
       - 来源：MEROPS C01.001实验记录的P1位点高频残基
         (Rawlings et al., 2018, Nucleic Acids Res, 46, D624-D632)
       - 数据库地址：https://www.ebi.ac.uk/merops/
    2. 修订说明：
       - 原规则中F（Phe）和L（Leu）的P1位点证据不充分，
         F在MEROPS P1实验记录中频率较低，
         L主要出现在P1'位点而非P1位点，
         故本次修订将二者从规则中移除。
    """
    cleavage = []

    # 修订后的P1切割规则：R、K、G（基于MEROPS C01.001实验记录）
    target_residues = ['R', 'K', 'G']

    for i in range(seq_len):
        if i < seq_len - 1:
            # 在R、K、G后切割（P1位点）
            if seq[i] in target_residues:
                cleavage.append(i)

    return cleavage

def apply_enzymes(seq: str, enzymes: List[Callable]) -> List[int]:
    """
    应用多种酶切酶
    
    Args:
        seq (str): 蛋白质序列
        enzymes (List[Callable]): 酶切函数列表
    
    Returns:
        List[int]: 去重排序的切割位点
    """
    seq_len = len(seq)
    clv = []
    
    for enzyme in enzymes:
        cleavage = enzyme(seq, seq_len)
        clv.extend(cleavage)
    
    # 去重并排序
    return sorted(list(set(clv)))

def digest_protein(seq: str, enzymes: List[Callable]) -> List[str]:
    """
    蛋白质序列酶切
    
    Args:
        seq (str): 蛋白质序列
        enzymes (List[Callable]): 酶切函数列表
    
    Returns:
        List[str]: 酶切后的肽段列表
    """
    seq_len = len(seq)
    cleavage_sites = [0] + apply_enzymes(seq, enzymes) + [seq_len-1]
    
    # 生成肽段
    peptides = [
        seq[cleavage_sites[i]:cleavage_sites[i+1]+1] if i == 0 
        else seq[cleavage_sites[i]+1:cleavage_sites[i+1]+1] 
        for i in range(len(cleavage_sites)-1)
    ]
    
    return peptides

def Alcalase(seq: str, seq_len: int) -> List[int]:  
    """  
    S8.001碱性蛋白酶(Subtilisin Carlsberg/Alcalase)切割位点识别  
    
    基于MEROPS热图的高分值位点:  
    1. 主要切割位点: L(8分), A(6分)  
    2. 次要切割位点: S, Y(5分)  
    3. 实验确认位点: K(组合肽数据)  
    
    来源: Bacillus licheniformis (MEROPS S8.001)  
    """  
    cleavage = []  
    
    # 仅限最高分值和实验确认的氨基酸  
    target_residues = ['L', 'A', 'S', 'Y', 'K']  
    
    for i in range(seq_len):  
        if i < seq_len - 1:  
            # 在目标氨基酸后切割  
            if seq[i] in target_residues:  
                cleavage.append(i)  
    
    return cleavage  


# 可用的酶切函数
ENZYME_FUNCTIONS = {
    'Trypsin': Trypsin,
    'Pepsin_1': Pepsin_1,
    'Pepsin_2': Pepsin_2,
    'Chymotrypsin_High': chymo_high,
    'Chymotrypsin_Low': chymo_low,
  #  'GluC': GluC,
 #   'AspN': AspN,
 #   'LysC': LysC
    'Proteinase_K': Proteinase_K,
    'Papain': Papain,
    'Alcalase': Alcalase
    
}
