import sys  
import os  
import torch  
import torch.nn as nn  
import numpy as np  
import logging  
from typing import List, Dict, Optional, Union  

# 添加项目根目录到系统路径  
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)  

# 导入酶切相关模块  
from src.protease_rules import ENZYME_FUNCTIONS  
from src.protein_digestion import process_protein_digestion  
from src.prepare_features import PeptideFeaturePreparer  

# 日志配置  
logging.basicConfig(  
    level=logging.INFO,  
    format='%(asctime)s - %(levelname)s - %(message)s'  
)  

class BiLSTMModel(nn.Module):  
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout_rate=0.3):  
        super(BiLSTMModel, self).__init__()  
        
        # 特征预处理层  
        self.feature_preprocessing = nn.Sequential(  
            nn.Linear(input_size, hidden_size),  
            nn.BatchNorm1d(hidden_size),  
            nn.ReLU(),  
            nn.Dropout(dropout_rate)  
        )  
        
        # BiLSTM层  
        self.bilstm = nn.LSTM(  
            input_size=hidden_size,  
            hidden_size=hidden_size//2,  
            num_layers=num_layers,  
            batch_first=True,  
            dropout=dropout_rate,  
            bidirectional=True  
        )  
        
        # 全连接层  
        self.fc_layers = nn.Sequential(  
            nn.Linear(hidden_size, hidden_size),  
            nn.BatchNorm1d(hidden_size),  
            nn.ReLU(),  
            nn.Dropout(dropout_rate),  
            nn.Linear(hidden_size, hidden_size//2),  
            nn.BatchNorm1d(hidden_size//2),  
            nn.ReLU(),  
            nn.Dropout(dropout_rate),  
            nn.Linear(hidden_size//2, 1),  
            nn.Sigmoid()  
        )  
    
    def forward(self, x):  
        # 保证输入是2D张量  
        if x.dim() == 1:  
            x = x.unsqueeze(0)  
        
        # 特征预处理  
        x = self.feature_preprocessing(x)  
        
        # BiLSTM处理 (需要增加一个维度)  
        x_lstm, _ = self.bilstm(x.unsqueeze(1))  
        x_lstm = x_lstm.squeeze(1)  
        
        # 全连接层  
        output = self.fc_layers(x_lstm)  
        
        return output.squeeze()  

class PeptideActivityPredictor:  
    def __init__(self, model_paths=None, base_dir: str = None):  
        if base_dir is None:
            base_dir = os.path.join(project_root, 'model')  
                
        # 默认模型文件路径  
        if model_paths is None:  
            model_paths = {  
                'ACE inhibitor': os.path.join(base_dir, 'ACE inhibitor_best_model_fold_3_auc_0.8269.pth'),  
                'antibacterial': os.path.join(base_dir, 'antibacterial_best_model_fold_5_auc_0.9892.pth'),  
                'antioxidative': os.path.join(base_dir, 'antioxidative_best_model_fold_5_auc_0.8626.pth'),  
                'dipeptidyl peptidase IV inhibitor': os.path.join(base_dir, 'dipeptidyl peptidase IV inhibitor_best_model_fold_7_auc_0.9298.pth')  
            }  
        
        # 设备配置  
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
        logging.info(f"使用设备: {self.device}")  
        
        # 特征准备器  
        self.feature_preparer = PeptideFeaturePreparer(  
            chemical_scaler_path=os.path.join(project_root, 'feature', 'chemical_scaler.joblib'),  
            sequence_scaler_path=os.path.join(project_root, 'feature', 'sequence_scaler.joblib')  
        )  
        
        # 模型加载  
        self.models = self._load_models(model_paths)  
        
        # 可用活性类型  
        self.available_activities = list(model_paths.keys())  
    
    def _load_models(self, model_paths: Dict[str, str]) -> Dict[str, torch.nn.Module]:  
        """  
        加载模型  
        """  
        models = {}  
        
        for activity, model_path in model_paths.items():  
            try:  
                logging.info(f"正在加载模型: {activity}")  
                
                # 检查模型文件是否存在  
                if not os.path.exists(model_path):  
                    logging.error(f"模型文件不存在: {model_path}")  
                    continue  

                # 加载检查点  
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)  
                
                # 创建模型实例  
                model = BiLSTMModel(input_size=128)  
                
                # 加载状态  
                model.load_state_dict(checkpoint['model_state_dict'])  
                model.to(self.device)  
                model.eval()  
                
                models[activity] = model  
                logging.info(f"成功加载 {activity} 模型")  
            
            except Exception as e:  
                logging.error(f"加载 {activity} 模型失败: {e}")  
                import traceback  
                traceback.print_exc()  
        
        return models  
    
    def predict(  
        self,   
        sequences: Union[str, List[str]],   
        selected_activities: Optional[List[str]] = None  
    ) -> Union[Dict[str, float], List[Dict[str, float]]]:  
        """  
        预测肽序列的活性概率  
        
        Args:  
            sequences (str or List[str]): 输入的肽序列  
            selected_activities (List[str], optional): 选择预测的活性类型  
        
        Returns:  
            Dict[str, float] or List[Dict[str, float]]: 活性预测结果  
        """  
        try:  
            # 如果是单个序列，转换为列表  
            if isinstance(sequences, str):  
                sequences = [sequences]  
            
            # 如果未指定活性，使用全部可用活性  
            if selected_activities is None:  
                selected_activities = self.available_activities  
            
            # 提取特征  
            features = self.feature_preparer.prepare_features(sequences)  
            
            # 转换为张量  
            feature_tensor = torch.tensor(features, dtype=torch.float32).to(self.device)  
            
            # 预测  
            predictions = []  
            with torch.no_grad():  
                # 如果特征是一维的，增加一个维度  
                if feature_tensor.dim() == 1:  
                    feature_tensor = feature_tensor.unsqueeze(0)  
                
                # 遍历每个序列的特征进行预测  
                for i in range(feature_tensor.shape[0]):  
                    prediction = {}  
                    for activity in selected_activities:  
                        if activity not in self.models:  
                            logging.warning(f"跳过未找到模型的活性: {activity}")  
                            continue  
                        
                        prob = self.models[activity](feature_tensor[i].unsqueeze(0)).item()  
                        prediction[activity] = prob  
                    predictions.append(prediction)  
            
            # 返回结果  
            return predictions if len(sequences) > 1 else predictions[0]  
        
        except Exception as e:  
            logging.error(f"预测过程出错: {e}")  
            import traceback  
            traceback.print_exc()  
            return None  

class PeptideAnalysisPlatform:  
    def __init__(  
        self,   
        base_path=project_root,  
        predictor=None,  
        model_paths: Optional[Dict[str, str]] = None  
    ):  
        """  
        初始化肽分析平台  
        
        Args:  
            base_path (str): 项目根目录  
            predictor (PeptideActivityPredictor, optional): 活性预测器  
            model_paths (dict, optional): 模型路径配置  
        """  
        self.base_path = base_path  
        
        # 可用酶种列表  
        self.available_enzymes = list(ENZYME_FUNCTIONS.keys())  
        
        # 活性预测器  
        self.predictor = predictor or PeptideActivityPredictor(model_paths)  
    
    def enzyme_digestion(  
        self,   
        protein_sequence: str,   
        selected_enzymes: Optional[List[str]] = None  
    ) -> dict:  
        """  
        执行酶切处理  
        
        Args:  
            protein_sequence (str): 输入序列  
            selected_enzymes (list, optional): 选择的酶种，默认为所有可用酶  
        
        Returns:  
            dict: 酶切结果  
        """  
        # 如果未指定酶，使用所有可用酶  
        if selected_enzymes is None:  
            selected_enzymes = self.available_enzymes  
        
        return process_protein_digestion(  
            protein_sequence.upper(),   
            selected_enzymes  
        )  
    
    def predict_peptide_activities(  
        self,   
        peptides: List[str],  
        selected_activities: Optional[List[str]] = None  
    ) -> List[Dict[str, float]]:  
        """  
        预测肽段活性  
        
        Args:  
            peptides (List[str]): 肽段列表  
            selected_activities (List[str], optional): 选择预测的活性类型  
        
        Returns:  
            List[Dict[str, float]]: 活性预测结果  
        """  
        return self.predictor.predict(  
            peptides,   
            selected_activities  
        )  
    
    def comprehensive_analysis(  
        self,   
        protein_sequence: str,   
        selected_enzymes: Optional[List[str]] = None,  
        selected_activities: Optional[List[str]] = None  
    ) -> dict:  
        """  
        对蛋白质序列进行全面分析  
        
        Args:  
            protein_sequence (str): 输入序列  
            selected_enzymes (list, optional): 选择的酶种  
            selected_activities (list, optional): 选择预测的活性类型  
        
        Returns:  
            dict: 全面分析结果  
        """  
        # 执行酶切  
        digestion_result = self.enzyme_digestion(  
            protein_sequence,   
            selected_enzymes  
        )  
        
        # 预测肽段活性  
        peptide_activities = self.predict_peptide_activities(  
            digestion_result['peptides'],  
            selected_activities  
        )  
        digestion_result['peptide_activities'] = peptide_activities  
        
        return digestion_result  

def main():  
    # 创建平台实例  
    platform = PeptideAnalysisPlatform()  
    
    # 示例1: 查看可用酶种和活性类型  
    print("可用酶种:")  
    print(platform.available_enzymes)  
    print("\n可用活性类型:")  
    print(platform.predictor.available_activities)  
    
    # 示例2: 单一序列活性预测  
    print("\n单一序列活性预测:")  
    sequence = "ACDEFG"  
    activities = platform.predictor.predict(  
        sequence,   
        selected_activities=['ACE inhibitor', 'antioxidative']  
    )  
    print(f"序列 {sequence} 的活性:")  
    for activity, prob in activities.items():  
        print(f"- {activity}: {prob:.4f}")  
    
    # 示例3: 多序列活性预测  
    print("\n多序列活性预测:")  
    sequences = ["ACDEFG", "HIJKLMN"]  
    activities_multi = platform.predictor.predict(  
        sequences,   
        selected_activities=['antibacterial', 'dipeptidyl peptidase IV inhibitor']  
    )  
    for seq, seq_activities in zip(sequences, activities_multi):  
        print(f"\n序列 {seq} 的活性:")  
        for activity, prob in seq_activities.items():  
            print(f"- {activity}: {prob:.4f}")  
    
    # 示例4: 酶切分析  
    print("\n酶切分析:")  
    test_sequence = "MKWVTFISLLLLFSSAYSRGVFRRDTHKSEIAHRFKDLGEEHFKGLVLIAFSQYLQQCPFDEHVKLVNELTEFAKTCVADESHAGCEKSLHTLFGDELCKVASLRETYGDMADCCEKQEPERNECFLSHKDDSPDLPKLKPDPNTLCDEFKADEKKFWGKYLYEIARRHPYFYAPELLYYANKYNGVFQECCQAEDKGACLLPKIETMREKVLASSARQRLRCASIQKFGERALKAWSVARLSQKFPKAEFVEVTKLVTDLTKVHKECCHGDLLECADDRADLAKYICDNQDTISSKLKECCDKPLLEKSHCIAEVEKDAIPENLPPLTADFAEDKDVCKNYQEAKDAFLGSFLYEYSRRHPEYAVSVLLRLAKEYEATLEECCAKDDPHACYSTVFDKLKHLVDEPQNLIKQNCDQFEKLGEYGFQNALIVRYTRKVPQVSTPTLVEVSRSLGKVGTRCCTKPESERMPCTEDYLSLILNRLCVLHEKTPVSEKVTKCCTESLVNRRPCFSALTPDETYVPKAFDEKLFTFHADICTLPDTEKQIKKQTALVELLKHKPKATEEQLKTVMENFVAFVDKCCAADDKEACFAVEGPKLVVSTQTALA"  
    
    # 使用部分酶种  
    digestion_result = platform.enzyme_digestion(  
        test_sequence,  
        selected_enzymes=['Trypsin']  
    )  
    
    print("使用的酶:", ', '.join(digestion_result['enzymes_used']))  
    print("肽段数量:", digestion_result['peptide_statistics']['total_peptides'])  
    
    # 示例5: 全面分析  
    print("\n全面分析:")  
    comprehensive_result = platform.comprehensive_analysis(  
        test_sequence,  
        selected_enzymes=['Trypsin'],  
        selected_activities=['ACE inhibitor', 'antioxidative']  
    )  
    
    print("使用的酶:", ', '.join(comprehensive_result['enzymes_used']))  
    print("肽段数量:", comprehensive_result['peptide_statistics']['total_peptides'])  
    
    print("\n肽段活性预测:")  
    for peptide, activities in zip(  
        comprehensive_result['peptides'],   
        comprehensive_result['peptide_activities']  
    ):  
        print(f"\n肽段: {peptide}")  
        for activity, prob in activities.items():  
            print(f"- {activity}: {prob:.4f}")  

if __name__ == "__main__":  
    main()  
