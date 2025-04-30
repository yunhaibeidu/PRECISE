#特征提取应用版
# -*- coding: utf-8 -*-
"""
基于预训练的标准器特征提取器类
"""
import os  
import sys  
import numpy as np  
import joblib  
from rdkit import Chem  
from rdkit.Chem import Descriptors  
from Bio.SeqUtils.ProtParam import ProteinAnalysis  

class SequenceFeatureExtractor:  
    def __init__(  
        self,   
        chemical_scaler_path=r'F:\硕士阶段任务\毕业论文2\peptide_prediction\src\chemical_scaler.joblib',  
        sequence_scaler_path=r'F:\硕士阶段任务\毕业论文2\peptide_prediction\src\sequence_scaler.joblib',  
        max_length=50  
    ):  
        """  
        初始化特征工程处理器  
        
        Args:  
            chemical_scaler_path (str): 化学特征标准化器路径  
            sequence_scaler_path (str): 序列特征标准化器路径  
            max_length (int): 最大序列长度  
        """  
        # 加载标准化器  
        self.chemical_scaler = joblib.load(chemical_scaler_path)  
        self.sequence_scaler = joblib.load(sequence_scaler_path)  
        
        self.max_length = max_length  
        
        # 氨基酸编码字典  
        self.aa_encoding_dict = {  
            aa: idx+1 for idx, aa in enumerate(sorted('ACDEFGHIKLMNPQRSTVWY'))  
        }  
    
    def _extract_molecular_descriptors(self, sequence):  
        """  
        提取分子描述符  
        
        Args:  
            sequence (str): 氨基酸序列  
        
        Returns:  
            list: 分子描述符列表  
        """  
        try:  
            mol = Chem.MolFromSequence(sequence)  
            protein_analysis = ProteinAnalysis(sequence)  
            
            descriptors = [  
                Descriptors.ExactMolWt(mol),  
                Descriptors.MolLogP(mol),  
                Descriptors.MolMR(mol),  
                Descriptors.HeavyAtomCount(mol),  
                mol.GetNumAtoms(),  
                mol.GetNumBonds(),  
                Descriptors.RingCount(mol),  
                Descriptors.NumAromaticRings(mol),  
                Descriptors.NumRotatableBonds(mol),  
                Descriptors.FractionCSP3(mol),  
                Descriptors.TPSA(mol),  
                Descriptors.NumHDonors(mol),  
                Descriptors.NumHAcceptors(mol),  
                Descriptors.NumHeteroatoms(mol),  
                Descriptors.NumRadicalElectrons(mol),  
                Descriptors.MaxPartialCharge(mol),  
                Descriptors.MinPartialCharge(mol),  
                Descriptors.qed(mol),  
                Descriptors.NumSaturatedRings(mol),  
                Descriptors.NumSpiroAtoms(mol),  
                Descriptors.NumBridgeheadAtoms(mol),  
                Descriptors.MaxEStateIndex(mol),  
                Descriptors.MinEStateIndex(mol),  
                protein_analysis.molecular_weight(),  
                protein_analysis.isoelectric_point(),  
                protein_analysis.gravy(),  
                protein_analysis.instability_index(),  
                len(sequence)  
            ]  
            
            return descriptors  
        
        except Exception as e:  
            print(f"特征提取错误: {e}")  
            return [0] * 28  # 返回全零数组  
    
    def _extract_sequence_features(self, sequence):  
        """  
        提取序列特征  
        
        Args:  
            sequence (str): 氨基酸序列  
        
        Returns:  
            numpy.ndarray: 序列特征矩阵  
        """  
        feature_matrix = np.zeros((self.max_length, 2))  
        
        # 截断或填充序列  
        sequence = sequence[:self.max_length]  
        
        for i, aa in enumerate(sequence):  
            if aa in self.aa_encoding_dict:  
                # 位置信息（归一化）  
                feature_matrix[i, 0] = (i + 1) / self.max_length  
                
                # 氨基酸数值编码  
                feature_matrix[i, 1] = self.aa_encoding_dict.get(aa, 0)  
        
        return feature_matrix  
    
    def extract_features(self, sequence):  
        """  
        提取并标准化特征  
        
        Args:  
            sequence (str): 输入的氨基酸序列  
        
        Returns:  
            numpy.ndarray: 融合的特征向量  
        """  
        # 提取分子描述符  
        chemical_features = self._extract_molecular_descriptors(sequence)  
        
        # 标准化分子描述符  
        chemical_features_scaled = self.chemical_scaler.transform([chemical_features])  
        
        # 提取序列特征  
        sequence_features = self._extract_sequence_features(sequence)  
        
        # 标准化序列特征  
        sequence_features_flat = sequence_features.flatten()  
        sequence_features_scaled = self.sequence_scaler.transform([sequence_features_flat])  
        
        # 特征拼接  
        fused_features = np.hstack([chemical_features_scaled, sequence_features_scaled])  
        
        return fused_features  
    
    def batch_extract_features(self, sequences):  
        """  
        批量提取多个序列的特征  
        
        Args:  
            sequences (list): 氨基酸序列列表  
        
        Returns:  
            list: 特征向量列表  
        """  
        return [self.extract_features(seq) for seq in sequences]  

def main():  
    # 初始化特征提取器  
    feature_processor = SequenceFeatureExtractor()  
    
    # 测试序列  
    test_sequences = [  
        "ACDEFGHIKLM",  # 示例序列1  
        "NPQRSTVWY",    # 示例序列2  
        "MKLFGHJKL"     # 示例序列3  
    ]  
    
    # 单个序列特征提取  
    print("单个序列特征提取:")  
    for seq in test_sequences:  
        features = feature_processor.extract_features(seq)  
        print(f"\n序列 {seq}:")  
        print("特征形状:", features.shape)  
        print("特征预览:", features)  
    
    # 批量特征提取  
    print("\n批量特征提取:")  
    batch_features = feature_processor.batch_extract_features(test_sequences)  
    for seq, features in zip(test_sequences, batch_features):  
        print(f"\n序列 {seq}:")  
        print("特征形状:", features.shape)  
        print("特征预览:", features)  

if __name__ == "__main__":  
    main()  