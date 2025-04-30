#返回特征给模型
# -*- coding: utf-8 -*-
"""  
特征准备器类  
"""  
import sys  
import numpy as np  
import logging  

# 添加源代码目录到 Python 路径  
sys.path.append(r'F:\硕士阶段任务\毕业论文2\peptide_prediction\src')  

from chem_sequence_feature_extractor import SequenceFeatureExtractor  

class PeptideFeaturePreparer:  
    def __init__(  
        self,   
        chemical_scaler_path=r'F:\硕士阶段任务\毕业论文2\peptide_prediction\src\chemical_scaler.joblib',  
        sequence_scaler_path=r'F:\硕士阶段任务\毕业论文2\peptide_prediction\src\sequence_scaler.joblib'  
    ):  
        """  
        初始化特征准备器  
        
        Args:  
            chemical_scaler_path (str): 化学特征标准化器路径  
            sequence_scaler_path (str): 序列特征标准化器路径  
        """  
        self.feature_extractor = SequenceFeatureExtractor(  
            chemical_scaler_path=chemical_scaler_path,  
            sequence_scaler_path=sequence_scaler_path  
        )  
    
    def validate_sequence(self, sequence):  
        """  
        验证肽序列有效性  
        
        Args:  
            sequence (str): 输入的氨基酸序列  
        
        Returns:  
            bool: 序列是否有效  
        """  
        valid_amino_acids = set('ACDEFGHIKLMNPQRSTVWY')  
        return (  
            isinstance(sequence, str) and   
            len(sequence) > 0 and   
            all(aa in valid_amino_acids for aa in sequence.upper())  
        )  
    
    def prepare_features(self, sequences):  
        """  
        准备特征  
        
        Args:  
            sequences (str or list): 单个或多个肽序列  
        
        Returns:  
            numpy.ndarray: 特征矩阵  
        """  
        # 处理单个序列和多个序列的情况  
        if isinstance(sequences, str):  
            sequences = [sequences]  
        
        # 验证并过滤序列  
        valid_sequences = [  
            seq.upper() for seq in sequences   
            if self.validate_sequence(seq)  
        ]  
        
        # 增加日志输出  
        logging.info(f"输入序列: {sequences}")  
        logging.info(f"有效序列: {valid_sequences}")  
        
        if not valid_sequences:  
            raise ValueError("无效的肽序列输入")  
        
        # 提取特征  
        features = self.feature_extractor.batch_extract_features(valid_sequences)  
        
        # 转换为numpy数组  
        features_array = np.vstack(features)  
        
        # 增加日志输出  
        logging.info(f"特征数组形状: {features_array.shape}")  
        
        return features_array  

def main():  
    # 配置日志  
    logging.basicConfig(level=logging.INFO,   
                        format='%(asctime)s - %(levelname)s - %(message)s')  
    
    # 示例使用  
    preparer = PeptideFeaturePreparer()  
    
    # 单个序列示例  
    single_sequence = "ACDEFG"  
    single_features = preparer.prepare_features(single_sequence)  
    print("单个序列特征:")  
    print(single_features)  
    print("特征形状:", single_features.shape)  
    
    # 多个序列示例  
    multiple_sequences = ["ACDEFG", "HIJKLMN", "OPQRST"]  
    multi_features = preparer.prepare_features(multiple_sequences)  
    print("\n多个序列特征:")  
    print(multi_features)  
    print("特征形状:", multi_features.shape)  

if __name__ == "__main__":  
    main()  
