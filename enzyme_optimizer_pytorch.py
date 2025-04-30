"""  
Simulated Annealing-based Enzyme Combination Optimizer - Single Activity Version  
For selecting optimal enzyme combinations to generate peptides with specific bioactivity  
"""  

import os  
import sys  
import torch  
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
import logging  
import time  
import traceback  
import random  
from typing import List, Dict, Tuple, Optional, Union  
from matplotlib.backends.backend_pdf import PdfPages  

# 关键修复：禁用matplotlib字体调试输出  
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)  

# 关键修复：使用简单字体配置  
plt.rcParams.update({  
    'font.family': 'sans-serif',  
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica', 'sans-serif'],  
    'pdf.fonttype': 42,  # 使用TrueType字体  
    'font.size': 10  
})  

# 添加项目根目录到系统路径  
project_root = r"F:\硕士阶段任务\毕业论文2\peptide_prediction"  
sys.path.insert(0, project_root)  

# 添加安全的全局变量  
torch.serialization.add_safe_globals([np.float64, np.ndarray, np.generic])  

# 导入项目模块  
from src.protease_rules import ENZYME_FUNCTIONS, digest_protein  
from src.protein_digestion import process_protein_digestion  
from src.prepare_features import PeptideFeaturePreparer  

# 导入肽活性预测模型  
import torch.nn as nn  

# 定义输出文件夹  
OUTPUT_DIR = r"F:\硕士阶段任务\毕业论文2\result_enzy"  
# 确保输出文件夹存在  
os.makedirs(OUTPUT_DIR, exist_ok=True)  

# 日志配置  
logging.basicConfig(  
    level=logging.INFO,  
    format='%(asctime)s - %(levelname)s - %(message)s'  
)  

# 导入BiLSTM模型定义  
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

# Peptide Activity Predictor Class  
class PeptideActivityPredictor:  
    def __init__(  
        self,   
        model_paths: Optional[Dict[str, str]] = None,  
        base_dir: str = r'F:\硕士阶段任务\毕业论文2\result_comparasion\all'  
    ):  
        # Default model file paths  
        if model_paths is None:  
            model_paths = {  
                'ACE inhibitor': os.path.join(base_dir, 'ACE inhibitor_best_model_fold_3_auc_0.8269.pth'),  
                'antibacterial': os.path.join(base_dir, 'antibacterial_best_model_fold_5_auc_0.9892.pth'),  
                'antioxidative': os.path.join(base_dir, 'antioxidative_best_model_fold_5_auc_0.8626.pth'),  
                'dipeptidyl peptidase IV inhibitor': os.path.join(base_dir, 'dipeptidyl peptidase IV inhibitor_best_model_fold_7_auc_0.9298.pth')  
            }  
        
        # Device configuration  
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
        logging.info(f"Using device: {self.device}")  
        
        # Feature preparer  
        self.feature_preparer = PeptideFeaturePreparer(  
            chemical_scaler_path=os.path.join(project_root, 'src', 'chemical_scaler.joblib'),  
            sequence_scaler_path=os.path.join(project_root, 'src', 'sequence_scaler.joblib')  
        )  
        
        # Load models  
        self.models = self._load_models(model_paths)  
    
    def _load_models(self, model_paths: Dict[str, str]) -> Dict[str, nn.Module]:  
        """  
        Load models  
        """  
        models = {}  
        
        for activity, model_path in model_paths.items():  
            try:  
                logging.info(f"Loading model: {activity}")  
                
                # Check if model file exists  
                if not os.path.exists(model_path):  
                    logging.error(f"Model file does not exist: {model_path}")  
                    continue  

                # Load checkpoint  
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)  
                
                # Create model instance  
                model = BiLSTMModel(input_size=128)  
                
                # Load state  
                model.load_state_dict(checkpoint['model_state_dict'])  
                model.to(self.device)  
                model.eval()  
                
                models[activity] = model  
                logging.info(f"Successfully loaded {activity} model")  
            
            except Exception as e:  
                logging.error(f"Failed to load {activity} model: {e}")  
                traceback.print_exc()  
        
        return models  
    
    def predict(self, sequences: Union[str, List[str]]) -> Union[Dict[str, float], List[Dict[str, float]]]:  
        """  
        Predict activity probability for peptide sequences  
        
        Args:  
            sequences (str or List[str]): Input peptide sequences  
        
        Returns:  
            Dict[str, float] or List[Dict[str, float]]: Activity prediction results  
        """  
        try:  
            # If a single sequence, convert to list  
            if isinstance(sequences, str):  
                sequences = [sequences]  
            
            # Extract features  
            try:  
                features = self.feature_preparer.prepare_features(sequences)  
            except Exception as feat_error:  
                logging.error(f"Feature extraction failed: {feat_error}")  
                traceback.print_exc()  
                return None  
        
            # Convert to tensor  
            feature_tensor = torch.tensor(features, dtype=torch.float32).to(self.device)  
        
            # Predict  
            predictions = []  
            with torch.no_grad():  
                # If features are 1D, add a dimension  
                if feature_tensor.dim() == 1:  
                    feature_tensor = feature_tensor.unsqueeze(0)  
            
                # Process each sequence  
                for i in range(feature_tensor.shape[0]):  
                    prediction = {}  
                    for activity, model in self.models.items():  
                        # Use features for specific sequence  
                        prob = model(feature_tensor[i].unsqueeze(0)).item()  
                        prediction[activity] = prob  
                    predictions.append(prediction)  
        
            # Return results  
            return predictions if len(sequences) > 1 else predictions[0]  
    
        except Exception as e:  
            logging.error(f"Prediction error: {e}")  
            traceback.print_exc()  
            return None  


class SimulatedAnnealingOptimizer:  
    """  
    Simulated annealing optimizer for enzyme combinations to generate peptides with specific bioactivity  
    """  
    def __init__(  
        self,   
        protein_sequence: str,  
        available_enzymes: List[str] = None,  
        peptide_predictor = None,  
        initial_temp: float = 100.0,  
        cooling_rate: float = 0.95,  
        max_iterations: int = 1000,  
        min_peptide_length: int = 2,  
        max_peptide_length: int = 50,  
        activity_threshold: float = 0.7,  
        device: str = None  
    ):  
        """  
        Initialize the simulated annealing optimizer  
        
        Args:  
            protein_sequence: Protein sequence to be digested  
            available_enzymes: List of available enzymes, default is all enzymes  
            peptide_predictor: Peptide activity predictor  
            initial_temp: Initial temperature  
            cooling_rate: Cooling rate  
            max_iterations: Maximum number of iterations  
            min_peptide_length: Minimum peptide length  
            max_peptide_length: Maximum peptide length  
            activity_threshold: Activity threshold  
            device: Computing device  
        """  
        # Device configuration  
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')  
        logging.info(f"Using device: {self.device}")  
        
        self.protein_sequence = protein_sequence  
        self.available_enzymes = available_enzymes or list(ENZYME_FUNCTIONS.keys())  
        logging.info(f"Available enzymes: {len(self.available_enzymes)}")  
        
        self.peptide_predictor = peptide_predictor  
        self.initial_temp = initial_temp  
        self.cooling_rate = cooling_rate  
        self.max_iterations = max_iterations  
        
        # Peptide filtering parameters  
        self.min_peptide_length = min_peptide_length  
        self.max_peptide_length = max_peptide_length  
        self.activity_threshold = activity_threshold  
        
        # Result storage  
        self.best_solution = None  
        self.best_fitness = 0  
        self.fitness_history = []  
        
    def evaluate_solution(self, solution: np.ndarray, target_activity: str) -> float:  
        """  
        Evaluate the fitness of an enzyme combination solution  
        
        Args:  
            solution: Binary-encoded enzyme combination  
            target_activity: Target activity  
            
        Returns:  
            float: Fitness value  
        """  
        # Decode solution  
        selected_enzymes = [self.available_enzymes[i] for i, val in enumerate(solution) if val == 1]  
        
        # If no enzymes selected, return zero fitness  
        if not selected_enzymes:  
            return 0  
        
        # Get enzyme functions  
        enzyme_functions = [ENZYME_FUNCTIONS[enzyme] for enzyme in selected_enzymes]  
        
        # Digest protein sequence  
        peptides = digest_protein(self.protein_sequence, enzyme_functions)  
        
        # If no peptides generated, return zero fitness  
        if not peptides:  
            return 0  
        
        # Filter peptides: keep those within length range  
        valid_peptides = [p for p in peptides if self.min_peptide_length <= len(p) <= self.max_peptide_length]  
        
        if not valid_peptides:  
            return 0  
        
        total_score = 0  
        total_predicted = 0  
        high_activity_count = 0  # Count of peptides above threshold  
        
        try:  
            # Use deep learning model to predict peptide activity  
            if self.peptide_predictor:  
                # Get batch predictions  
                predictions = self.peptide_predictor.predict(valid_peptides)  
                
                # If single peptide, convert result to list  
                if not isinstance(predictions, list):  
                    predictions = [predictions]  
                
                # Calculate target function score  
                for peptide_pred in predictions:  
                    if peptide_pred and target_activity in peptide_pred:  
                        activity_score = peptide_pred[target_activity]  
                        total_score += activity_score  
                        total_predicted += 1  
                        
                        # Count high activity peptides  
                        if activity_score >= self.activity_threshold:  
                            high_activity_count += 1  
        
        except Exception as e:  
            logging.error(f"Peptide evaluation error: {e}")  
            traceback.print_exc()  
            return 0  
        
        # If no valid predictions, return 0  
        if total_predicted == 0:  
            return 0  
        
        # Calculate fitness  
        avg_score = total_score / total_predicted  
        
        # Fitness definition:  
        # 1. High activity peptide ratio (≥0.7) - increases weight of high-quality peptides  
        # 2. Average activity score - reflects overall peptide activity level  
        # 3. Minus complexity penalty - encourages using fewer enzymes  
        
        # Calculate high activity peptide ratio  
        high_activity_ratio = high_activity_count / total_predicted if total_predicted > 0 else 0  
        
        # Complexity penalty (fewer enzymes have lower penalty)  
        complexity_penalty = np.log(1 + len(selected_enzymes)) / 10  
        
        # Final fitness = weighted average score + weighted high activity ratio - complexity penalty  
        # Higher weight on high activity ratio because we prioritize generating high activity peptides  
        fitness = (avg_score * 0.4 + high_activity_ratio * 0.6) - complexity_penalty  
        
        return fitness  
    
    def get_neighbor(self, solution: np.ndarray) -> np.ndarray:  
        """  
        Generate a neighboring solution (by randomly flipping 1-3 genes)  
        
        Args:  
            solution: Current solution  
            
        Returns:  
            np.ndarray: Neighboring solution  
        """  
        neighbor = solution.copy()  
        
        # Randomly select 1-3 positions to flip  
        num_flips = min(3, len(solution))  
        flip_positions = np.random.choice(len(solution), size=num_flips, replace=False)  
        
        for pos in flip_positions:  
            neighbor[pos] = 1 - neighbor[pos]  
        
        return neighbor  
    
    def optimize(self, target_activity: str) -> Dict:  
        """  
        Run the simulated annealing algorithm main loop  
        
        Args:  
            target_activity: Target activity  
            
        Returns:  
            Dict: Optimization results  
        """  
        start_time = time.time()  
        
        # Initialize random solution  
        current_solution = np.random.randint(0, 2, size=len(self.available_enzymes))  
        
        # Evaluate initial solution  
        current_fitness = self.evaluate_solution(current_solution, target_activity)  
        
        # Initialize best solution  
        self.best_solution = current_solution.copy()  
        self.best_fitness = current_fitness  
        
        # Record initial fitness  
        self.fitness_history = [current_fitness]  
        
        # Initialize temperature  
        temp = self.initial_temp  
        
        # Track additional metrics  
        acceptance_rate_history = []  
        improvement_count = 0  
        iterations_since_improvement = 0  
        max_iterations_without_improvement = 100  # Early stopping parameter  
        
        # Simulated annealing main loop  
        for iteration in range(self.max_iterations):  
            iter_start_time = time.time()  
            
            # Generate neighboring solution  
            neighbor = self.get_neighbor(current_solution)  
            
            # Evaluate neighboring solution  
            neighbor_fitness = self.evaluate_solution(neighbor, target_activity)  
            
            # Calculate fitness delta  
            fitness_delta = neighbor_fitness - current_fitness  
            
            # Acceptance rule:  
            # 1. If better, always accept  
            # 2. If worse, accept with probability based on temperature  
            accepted = False  
            if fitness_delta > 0:  
                # Better solution, accept directly  
                current_solution = neighbor  
                current_fitness = neighbor_fitness  
                accepted = True  
                # Count improvements  
                if neighbor_fitness > self.best_fitness:  
                    improvement_count += 1  
                    iterations_since_improvement = 0  
                else:  
                    iterations_since_improvement += 1  
            else:  
                # Worse solution, probabilistic acceptance  
                # Higher temperature = more likely to accept worse solutions  
                acceptance_probability = np.exp(fitness_delta / temp)  
                
                if np.random.random() < acceptance_probability:  
                    current_solution = neighbor  
                    current_fitness = neighbor_fitness  
                    accepted = True  
                    iterations_since_improvement += 1  
                else:  
                    iterations_since_improvement += 1  
            
            # Track acceptance rate for this iteration  
            acceptance_rate_history.append(1 if accepted else 0)  
            
            # Update best solution  
            if current_fitness > self.best_fitness:  
                self.best_solution = current_solution.copy()  
                self.best_fitness = current_fitness  
                logging.info(f"Iteration {iteration+1}: Found better solution, fitness: {self.best_fitness:.6f}")  
            
            # Record history  
            self.fitness_history.append(self.best_fitness)  
            
            # Cool down temperature  
            temp *= self.cooling_rate  
            
            # Output progress periodically  
            if (iteration + 1) % 50 == 0 or iteration == 0:  
                iter_time = time.time() - iter_start_time  
                recent_acceptance_rate = np.mean(acceptance_rate_history[-50:]) if len(acceptance_rate_history) >= 50 else np.mean(acceptance_rate_history)  
                logging.info(f"Iteration {iteration+1}/{self.max_iterations}, "  
                             f"Temp: {temp:.6f}, "  
                             f"Current fitness: {current_fitness:.6f}, "  
                             f"Best fitness: {self.best_fitness:.6f}, "  
                             f"Accept rate: {recent_acceptance_rate:.2f}, "  
                             f"Time: {iter_time:.2f}s")  
            
            # Early stopping if no improvement for a while  
            if iterations_since_improvement >= max_iterations_without_improvement:  
                logging.info(f"Early stopping after {iteration+1} iterations with no improvement")  
                break  
        
        # Decode best solution  
        best_enzymes = [self.available_enzymes[i] for i, val in enumerate(self.best_solution) if val == 1]  
        
        # Get peptides from best solution  
        if best_enzymes:  
            enzyme_functions = [ENZYME_FUNCTIONS[enzyme] for enzyme in best_enzymes]  
            best_peptides = digest_protein(self.protein_sequence, enzyme_functions)  
        else:  
            best_peptides = []  
        
        total_time = time.time() - start_time  
        logging.info(f"Optimization completed, total time: {total_time:.2f}s")  
        
        # Calculate additional performance metrics  
        avg_acceptance_rate = np.mean(acceptance_rate_history)  
        improvement_rate = improvement_count / len(self.fitness_history) if self.fitness_history else 0  
        
        # Return results  
        return {  
            "best_enzymes": best_enzymes,  
            "best_fitness": self.best_fitness,  
            "best_peptides": best_peptides,  
            "fitness_history": self.fitness_history,  
            "optimization_time": total_time,  
            "target_activity": target_activity,  
            "metrics": {  
                "avg_acceptance_rate": avg_acceptance_rate,  
                "improvement_rate": improvement_rate,  
                "total_improvements": improvement_count,  
                "early_stopped": iterations_since_improvement >= max_iterations_without_improvement,  
                "iterations_completed": len(self.fitness_history) - 1  
            }  
        }  
    
    def plot_fitness_history(self, target_activity: str, save_path: str = None):  
        """Plot fitness history curve with basic styling"""  
        # 使用简单的字体设置  
        plt.figure(figsize=(10, 6))  
        
        plt.plot(range(1, len(self.fitness_history) + 1), self.fitness_history,   
                 linewidth=2.0, color='blue')  
        
        plt.title(f'Optimization Process (Target: {target_activity})')  
        plt.xlabel('Iteration')  
        plt.ylabel('Best Fitness')  
        plt.grid(True)  
        
        # If no save path specified, use default  
        if not save_path:  
            save_path = os.path.join(OUTPUT_DIR, f"enzyme_optimization_history_{target_activity}.png")  
            
        plt.savefig(save_path, dpi=300, bbox_inches='tight')  
        logging.info(f"Optimization history saved to: {save_path}")  
            
        plt.close()  
        
        return save_path  
        
    def analyze_results(self, result: Dict) -> Dict:  
        """  
        Analyze peptides generated by the best enzyme combination  
        
        Args:  
            result: Optimization result  
            
        Returns:  
            Dict: Analysis result  
        """  
        peptides = result["best_peptides"]  
        target_activity = result["target_activity"]  
        
        # Basic statistics  
        peptide_lengths = [len(p) for p in peptides]  
        
        # Filter peptides: keep those within length range  
        valid_peptides = [p for p in peptides if self.min_peptide_length <= len(p) <= self.max_peptide_length]  
        
        # Peptide composition analysis  
        aa_counts = {}  
        total_aas = 0  
        
        for peptide in valid_peptides:  
            for aa in peptide:  
                aa_counts[aa] = aa_counts.get(aa, 0) + 1  
                total_aas += 1  
        
        # Calculate amino acid composition percentages  
        aa_composition = {}  
        if total_aas > 0:  
            for aa, count in aa_counts.items():  
                aa_composition[aa] = (count / total_aas) * 100  
        
        # Peptide activity prediction  
        peptide_predictions = {}  
        high_activity_peptides = []  
        
        # Count high activity peptides  
        high_activity_count = 0  
        
        # Additional evaluation metrics  
        activity_scores = []  
        
        try:  
            if valid_peptides and self.peptide_predictor:  
                # Batch predict all peptides  
                predictions = self.peptide_predictor.predict(valid_peptides)  
                
                # If single peptide, convert to list  
                if not isinstance(predictions, list):  
                    predictions = [predictions]  
                
                # Save prediction results  
                for i, peptide in enumerate(valid_peptides):  
                    if i < len(predictions) and predictions[i]:  
                        peptide_pred = predictions[i]  
                        peptide_predictions[peptide] = peptide_pred  
                        
                        # If contains target activity, add to high activity peptides  
                        if target_activity in peptide_pred:  
                            score = peptide_pred[target_activity]  
                            activity_scores.append(score)  
                            high_activity_peptides.append((peptide, score))  
                            
                            # Count high activity peptides  
                            if score >= self.activity_threshold:  
                                high_activity_count += 1  
                        
            # Sort by activity score  
            high_activity_peptides.sort(key=lambda x: x[1], reverse=True)  
            
        except Exception as e:  
            logging.error(f"Peptide analysis error: {e}")  
            traceback.print_exc()  
        
        # Calculate additional metrics  
        avg_activity_score = np.mean(activity_scores) if activity_scores else 0  
        median_activity_score = np.median(activity_scores) if activity_scores else 0  
        activity_std = np.std(activity_scores) if len(activity_scores) > 1 else 0  
        
        # Score distribution  
        score_distribution = {  
            "0.0-0.1": len([s for s in activity_scores if 0.0 <= s < 0.1]),  
            "0.1-0.2": len([s for s in activity_scores if 0.1 <= s < 0.2]),  
            "0.2-0.3": len([s for s in activity_scores if 0.2 <= s < 0.3]),  
            "0.3-0.4": len([s for s in activity_scores if 0.3 <= s < 0.4]),  
            "0.4-0.5": len([s for s in activity_scores if 0.4 <= s < 0.5]),  
            "0.5-0.6": len([s for s in activity_scores if 0.5 <= s < 0.6]),  
            "0.6-0.7": len([s for s in activity_scores if 0.6 <= s < 0.7]),  
            "0.7-0.8": len([s for s in activity_scores if 0.7 <= s < 0.8]),  
            "0.8-0.9": len([s for s in activity_scores if 0.8 <= s < 0.9]),  
            "0.9-1.0": len([s for s in activity_scores if 0.9 <= s <= 1.0])  
        }  
        
        # Length distribution  
        length_distribution = {}  
        for length in range(self.min_peptide_length, self.max_peptide_length + 1):  
            count = len([p for p in valid_peptides if len(p) == length])  
            if count > 0:  
                length_distribution[length] = count  
        
        # Return comprehensive analysis  
        return {  
            "peptide_count": len(peptides),  
            "valid_peptide_count": len(valid_peptides),  
            "peptide_length_stats": {  
                "min": min(peptide_lengths) if peptide_lengths else 0,  
                "max": max(peptide_lengths) if peptide_lengths else 0,  
                "mean": np.mean(peptide_lengths) if peptide_lengths else 0,  
                "median": np.median(peptide_lengths) if peptide_lengths else 0,  
                "std": np.std(peptide_lengths) if len(peptide_lengths) > 1 else 0  
            },  
            "aa_composition": aa_composition,  
            "target_activity": target_activity,  
            "activity_stats": {  
                "mean": avg_activity_score,  
                "median": median_activity_score,  
                "std": activity_std,  
                "min": min(activity_scores) if activity_scores else 0,  
                "max": max(activity_scores) if activity_scores else 0  
            },  
            "score_distribution": score_distribution,  
            "length_distribution": length_distribution,  
            "high_activity_peptides": high_activity_peptides[:10],  # Top 10 high activity peptides  
            "high_activity_count": high_activity_count,  # High activity peptide count  
            "high_activity_percentage": 100 * high_activity_count / len(valid_peptides) if valid_peptides else 0  # Percentage  
        }  


def get_available_enzymes() -> List[str]:  
    """  
    Get all available enzyme types  
    
    Returns:  
        List[str]: List of available enzymes  
    """  
    return list(ENZYME_FUNCTIONS.keys())  


def get_available_activities() -> List[str]:  
    """  
    Get all predictable activity types  
    
    Returns:  
        List[str]: List of available activities  
    """  
    # Try to initialize predictor and get activity list  
    try:  
        predictor = PeptideActivityPredictor()  
        return list(predictor.models.keys())  
    except Exception as e:  
        logging.error(f"Failed to get activity list: {e}")  
        # Default activity list  
        return [  
            'ACE inhibitor',   
            'antibacterial',   
            'antioxidative',   
            'dipeptidyl peptidase IV inhibitor'  
        ]  


def optimize_enzyme_combination(  
    protein_sequence: str,  
    target_activity: str,  
    available_enzymes: List[str] = None,  
    max_iterations: int = 1000,  
    initial_temp: float = 100.0,  
    cooling_rate: float = 0.95,  
    device: str = None,  
    min_peptide_length: int = 2,  
    max_peptide_length: int = 50,  
    activity_threshold: float = 0.7,  
    export_figures: bool = False  # 默认禁用单独图表导出，避免错误  
) -> Dict:  
    """  
    使用模拟退火算法优化酶组合以生成特定功能的肽段  
    
    Args:  
        protein_sequence: 蛋白质序列  
        target_activity: 目标功能  
        available_enzymes: 可用酶列表，默认为全部  
        max_iterations: 最大迭代次数  
        initial_temp: 初始温度  
        cooling_rate: 冷却速率  
        device: 使用的设备 ('cuda' 或 'cpu')  
        min_peptide_length: 最小肽段长度  
        max_peptide_length: 最大肽段长度  
        activity_threshold: 活性阈值  
        export_figures: 是否导出单独图表  
        
    Returns:  
        Dict: 优化结果  
    """  
    # 确保输出目录存在  
    os.makedirs(OUTPUT_DIR, exist_ok=True)  
    
    # 初始化肽活性预测器  
    try:  
        peptide_predictor = PeptideActivityPredictor()  
        logging.info("Successfully initialized peptide activity predictor")  
    except Exception as e:  
        logging.error(f"Failed to initialize peptide activity predictor: {e}")  
        traceback.print_exc()  
        return {"error": "Predictor initialization failed, cannot continue optimization"}  
    
    # 创建优化器  
    optimizer = SimulatedAnnealingOptimizer(  
        protein_sequence=protein_sequence,  
        available_enzymes=available_enzymes,  
        peptide_predictor=peptide_predictor,  
        initial_temp=initial_temp,  
        cooling_rate=cooling_rate,  
        max_iterations=max_iterations,  
        min_peptide_length=min_peptide_length,  
        max_peptide_length=max_peptide_length,  
        activity_threshold=activity_threshold,  
        device=device  
    )  
    
    # 运行优化  
    print(f"Starting enzyme combination optimization using simulated annealing, target: {target_activity}, threshold: {activity_threshold}")  
    result = optimizer.optimize(target_activity)  
    
    # 分析结果  
    analysis = optimizer.analyze_results(result)  
    result["analysis"] = analysis  
    
    # 绘制优化历史  
    history_path = optimizer.plot_fitness_history(  
        target_activity,  
        os.path.join(OUTPUT_DIR, f"enzyme_optimization_history_{target_activity}.png")  
    )  
    result["history_plot_path"] = history_path  
    
    return result  


def generate_pdf_report(result: Dict, protein_sequence: str, output_file: str):  
    """  
    Generate a comprehensive PDF report for optimization results  
    
    Args:  
        result: Optimization result dictionary  
        protein_sequence: The original protein sequence  
        output_file: Output PDF file path  
    """  
    # 关键修复：禁用字体调试输出  
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)  
    
    # 关键修复：简化字体配置  
    plt.rcParams.update({  
        'font.family': 'sans-serif',  
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],  
        'pdf.fonttype': 42,  # 使用TrueType字体  
        'font.size': 12  
    })  
    
    target_activity = result["target_activity"]  
    analysis = result["analysis"]  
    
    try:  
        # 创建PdfPages对象  
        with PdfPages(output_file) as pdf:  
            # 标题页  
            plt.figure(figsize=(11.7, 8.3))  # A4尺寸  
            plt.axis('off')  
            
            plt.text(0.5, 0.9, "Enzyme Combination Optimization Report",   
                     fontsize=24, ha='center')  
            plt.text(0.5, 0.8, f"Target Activity: {target_activity}",   
                     fontsize=18, ha='center')  
            plt.text(0.5, 0.7, f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}",   
                     fontsize=14, ha='center')  
            plt.text(0.5, 0.6, "Optimized using Simulated Annealing Algorithm",   
                     fontsize=14, ha='center')  
            plt.text(0.5, 0.3, "Summary:", fontsize=16, ha='center')  
            
            # 添加摘要统计  
            summary_text = (  
                f"• Protein length: {len(protein_sequence)} amino acids\n"  
                f"• Best fitness: {result['best_fitness']:.4f}\n"  
                f"• Optimal enzyme combination: {', '.join(result['best_enzymes'])}\n"  
                f"• Total peptides generated: {analysis['peptide_count']}\n"  
                f"• Valid peptides (length ≥2): {analysis['valid_peptide_count']}\n"  
                f"• High activity peptides (≥0.7): {analysis['high_activity_count']} ({analysis['high_activity_percentage']:.1f}%)\n"  
                f"• Mean activity score: {analysis['activity_stats']['mean']:.4f}\n"  
                f"• Optimization time: {result['optimization_time']:.2f} seconds"  
            )  
            plt.text(0.5, 0.2, summary_text, fontsize=12, ha='center', va='center', linespacing=1.5)  
            
            pdf.savefig()  
            plt.close()  
            
            # 优化历史曲线  
            plt.figure(figsize=(11.7, 8.3))  
            plt.subplot(111)  
            plt.plot(range(1, len(result["fitness_history"]) + 1), result["fitness_history"],  
                    linewidth=2.0, color='blue')  
            plt.title(f'Optimization Process (Target: {target_activity})', fontsize=16)  
            plt.xlabel('Iteration', fontsize=12)  
            plt.ylabel('Best Fitness', fontsize=12)  
            plt.grid(True)  
            
            plt.tight_layout()  
            pdf.savefig()  
            plt.close()  
            
            # 活性得分分布图  
            if "score_distribution" in analysis:  
                plt.figure(figsize=(11.7, 8.3))  
                
                score_ranges = list(analysis["score_distribution"].keys())  
                counts = list(analysis["score_distribution"].values())  
                
                # 使用简单的颜色列表  
                colors = []  
                for score_range in score_ranges:  
                    if score_range in ["0.7-0.8", "0.8-0.9", "0.9-1.0"]:  
                        colors.append('green')  # 高活性区域  
                    else:  
                        colors.append('blue')   # 其他区域  
                
                bars = plt.bar(score_ranges, counts, color=colors, width=0.7, edgecolor='black')  
                
                # 添加数值标签  
                for bar in bars:  
                    height = bar.get_height()  
                    if height > 0:  
                        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,  
                                f'{height}', ha='center', va='bottom', fontsize=10)  
                
                plt.title(f'Activity Score Distribution for {target_activity}', fontsize=16)  
                plt.xlabel('Activity Score Range', fontsize=12)  
                plt.ylabel('Number of Peptides', fontsize=12)  
                plt.xticks(rotation=45)  
                plt.grid(axis='y', linestyle='--', alpha=0.7)  
                
                # 添加高活性区域标注  
                plt.axvspan(6.5, 9.5, alpha=0.1, color='green')  
                plt.text(8, max(counts)/2, 'High Activity Zone (≥0.7)',   
                         ha='center', va='center', fontsize=10, color='darkgreen')  
                
                plt.tight_layout()  
                pdf.savefig()  
                plt.close()  
            
            # 肽段长度分布图  
            if "length_distribution" in analysis:  
                plt.figure(figsize=(11.7, 8.3))  
                
                lengths = list(analysis["length_distribution"].keys())  
                counts = list(analysis["length_distribution"].values())  
                
                plt.bar(lengths, counts, color='royalblue', width=0.7, edgecolor='black')  
                
                plt.title(f'Peptide Length Distribution', fontsize=16)  
                plt.xlabel('Peptide Length (amino acids)', fontsize=12)  
                plt.ylabel('Count', fontsize=12)  
                plt.grid(axis='y', linestyle='--', alpha=0.7)  
                
                plt.tight_layout()  
                pdf.savefig()  
                plt.close()  
            
            # 氨基酸组成图  
            if "aa_composition" in analysis and analysis["aa_composition"]:  
                plt.figure(figsize=(11.7, 8.3))  
                
                aa_letters = list(analysis["aa_composition"].keys())  
                aa_percentages = list(analysis["aa_composition"].values())  
                
                # 按百分比排序  
                sorted_pairs = sorted(zip(aa_letters, aa_percentages), key=lambda x: x[1], reverse=True)  
                aa_letters, aa_percentages = zip(*sorted_pairs) if sorted_pairs else ([], [])  
                
                # 使用简单的颜色  
                bars = plt.bar(aa_letters, aa_percentages, color='cornflowerblue', width=0.7, edgecolor='black')  
                
                # 在柱状图上添加百分比标签  
                for bar in bars:  
                    height = bar.get_height()  
                    if height > 1.0:  # 只为较大百分比添加标签  
                        plt.text(bar.get_x() + bar.get_width()/2., height + 0.3,  
                                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)  
                
                plt.title(f'Amino Acid Composition of Generated Peptides', fontsize=16)  
                plt.xlabel('Amino Acid', fontsize=12)  
                plt.ylabel('Percentage (%)', fontsize=12)  
                plt.grid(axis='y', linestyle='--', alpha=0.7)  
                
                plt.tight_layout()  
                pdf.savefig()  
                plt.close()  
            
            # 高活性肽段表格  
            if "high_activity_peptides" in analysis and analysis["high_activity_peptides"]:  
                plt.figure(figsize=(11.7, 8.3))  
                plt.axis('off')  
                plt.text(0.5, 0.95, f"Top High Activity Peptides for {target_activity}",   
                         fontsize=16, ha='center')  
                
                # 创建高活性肽段表格  
                high_activity_data = []  
                for i, (peptide, score) in enumerate(analysis["high_activity_peptides"][:15], 1):  
                    high_activity_data.append([i, peptide, f"{score:.4f}", len(peptide), "Yes" if score >= 0.7 else "No"])  
                
                columns = ["#", "Peptide Sequence", "Activity Score", "Length", "High Activity (≥0.7)"]  
                table = plt.table(  
                    cellText=high_activity_data,  
                    colLabels=columns,  
                    loc='center',  
                    cellLoc='center',  
                    colWidths=[0.05, 0.4, 0.15, 0.1, 0.2]  
                )  
                table.auto_set_font_size(False)  
                table.set_fontsize(10)  
                table.scale(1, 1.5)  
                
                pdf.savefig()  
                plt.close()  
            
            # 酶组合详情  
            plt.figure(figsize=(11.7, 8.3))  
            plt.axis('off')  
            plt.text(0.5, 0.95, "Optimal Enzyme Combination Details",   
                     fontsize=16, ha='center')  
            
            enzyme_text = "\n".join([f"• {enzyme}" for enzyme in result["best_enzymes"]])  
            plt.text(0.5, 0.7, enzyme_text, fontsize=12, ha='center', va='center', linespacing=1.5)  
            
            # 算法性能指标  
            if "metrics" in result:  
                plt.text(0.5, 0.4, "Algorithm Performance Metrics:",   
                         fontsize=14, ha='center')  
                
                metrics_text = (  
                    f"• Average acceptance rate: {result['metrics']['avg_acceptance_rate']:.4f}\n"  
                    f"• Improvement rate: {result['metrics']['improvement_rate']:.4f}\n"  
                    f"• Total improvements: {result['metrics']['total_improvements']}\n"  
                    f"• Early stopped: {'Yes' if result['metrics']['early_stopped'] else 'No'}\n"  
                    f"• Completed iterations: {result['metrics']['iterations_completed']}"  
                )  
                plt.text(0.5, 0.25, metrics_text, fontsize=12, ha='center', va='center', linespacing=1.5)  
            
            pdf.savefig()  
            plt.close()  
            
        logging.info(f"PDF report successfully generated: {output_file}")  
        return output_file  
    
    except Exception as e:  
        logging.error(f"Error generating PDF report: {e}")  
        traceback.print_exc()  
        # 尝试生成简化版PDF  
        try:  
            return generate_simple_pdf_report(result, protein_sequence, output_file)  
        except Exception as e2:  
            logging.error(f"Error generating simple PDF report: {e2}")  
            return None  


def generate_simple_pdf_report(result: Dict, protein_sequence: str, output_file: str):  
    """生成简化版PDF报告，减少字体问题"""  
    # 关键修复：使用最基本的字体设置  
    plt.rcParams.update({  
        'font.family': 'monospace',  # 使用等宽字体，通常更可靠  
        'pdf.fonttype': 42,  
        'font.size': 10  
    })  
    
    target_activity = result["target_activity"]  
    analysis = result["analysis"]  
    
    with PdfPages(output_file) as pdf:  
        # 简化的标题页  
        plt.figure(figsize=(8.5, 11))  
        plt.axis('off')  
        plt.text(0.5, 0.95, "Enzyme Optimization Report", fontsize=16, ha='center')  
        plt.text(0.5, 0.9, f"Target: {target_activity}", fontsize=14, ha='center')  
        
        # 简要摘要  
        summary = (  
            f"Protein length: {len(protein_sequence)}\n"  
            f"Best fitness: {result['best_fitness']:.4f}\n"  
            f"Enzymes: {', '.join(result['best_enzymes'])}\n"  
            f"Total peptides: {analysis['peptide_count']}\n"  
            f"High activity peptides: {analysis['high_activity_count']}"  
        )  
        plt.text(0.5, 0.8, summary, fontsize=12, ha='center', va='center', linespacing=1.5)  
        
        pdf.savefig()  
        plt.close()  
        
        # 简化的图表页 - 只包含优化历史  
        plt.figure(figsize=(8.5, 11))  
        plt.plot(range(1, len(result["fitness_history"]) + 1), result["fitness_history"])  
        plt.title('Optimization History')  
        plt.xlabel('Iteration')  
        plt.ylabel('Fitness')  
        plt.grid(True)  
        
        pdf.savefig()  
        plt.close()  
    
    return output_file  


def export_optimization_results(  
    result: Dict,   
    protein_sequence: str,  
    output_excel: str = None,  
    output_pdf: str = None  
):  
    """  
    Export optimization results to Excel and PDF files  
    
    Args:  
        result: Optimization result  
        protein_sequence: Original protein sequence  
        output_excel: Excel output filename, auto-generated if not specified  
        output_pdf: PDF output filename, auto-generated if not specified  
    
    Returns:  
        Tuple[str, str]: Paths to the output Excel and PDF files  
    """  
    # Ensure output directory exists  
    os.makedirs(OUTPUT_DIR, exist_ok=True)  
    
    # Get target activity  
    target_activity = result["target_activity"]  
    
    # Generate timestamp for filenames  
    timestamp = time.strftime("%Y%m%d_%H%M%S")  
    
    # If Excel output filename not specified, generate one  
    if not output_excel:  
        output_excel = os.path.join(OUTPUT_DIR, f"enzyme_optimization_results_{target_activity}_{timestamp}.xlsx")  
    else:  
        output_excel = os.path.join(OUTPUT_DIR, output_excel)  
    
    # Export to Excel  
    with pd.ExcelWriter(output_excel) as writer:  
        # Summary  
        summary_data = {  
            "Property": [  
                "Protein Sequence Length",   
                "Target Activity",   
                "Activity Threshold",  
                "Best Fitness",   
                "Optimal Enzyme Count",   
                "Total Peptides Generated",   
                "Valid Peptides (length ≥2)",  
                "Mean Activity Score",  
                f"High Activity Peptides (≥0.7)",  
                f"High Activity Percentage (%)",  
                "Optimization Time (s)",  
                "Algorithm"  
            ],  
            "Value": [  
                len(protein_sequence),  
                target_activity,  
                0.7,  # Fixed threshold  
                result["best_fitness"],  
                len(result["best_enzymes"]),  
                result["analysis"]["peptide_count"],  
                result["analysis"]["valid_peptide_count"],  
                result["analysis"]["activity_stats"]["mean"],  
                result["analysis"]["high_activity_count"],  
                f"{result['analysis']['high_activity_percentage']:.1f}",  
                result["optimization_time"],  
                "Simulated Annealing"  
            ]  
        }  
        
        pd.DataFrame(summary_data).to_excel(writer, sheet_name="Summary", index=False)  
        
        # Algorithm metrics  
        if "metrics" in result:  
            metrics_data = {  
                "Metric": [  
                    "Average Acceptance Rate",  
                    "Improvement Rate",  
                    "Total Improvements",  
                    "Early Stopped",  
                    "Completed Iterations"  
                ],  
                "Value": [  
                    f"{result['metrics']['avg_acceptance_rate']:.4f}",  
                    f"{result['metrics']['improvement_rate']:.4f}",  
                    result['metrics']['total_improvements'],  
                    "Yes" if result['metrics']['early_stopped'] else "No",  
                    result['metrics']['iterations_completed']  
                ]  
            }  
            pd.DataFrame(metrics_data).to_excel(writer, sheet_name="Algorithm Metrics", index=False)  
        
        # Best enzyme combination  
        pd.DataFrame({  
            "Enzyme": result["best_enzymes"],  
            "Selected": [1] * len(result["best_enzymes"])  
        }).to_excel(writer, sheet_name="Best Enzyme Combination", index=False)  
        
        # Generated peptides  
        peptides_df = pd.DataFrame({  
            "Peptide Sequence": result["best_peptides"],  
            "Length": [len(p) for p in result["best_peptides"]]  
        })  
        peptides_df.to_excel(writer, sheet_name="Generated Peptides", index=False)  
        
        # Score distribution  
        if "score_distribution" in result["analysis"]:  
            score_dist = result["analysis"]["score_distribution"]  
            score_data = {  
                "Score Range": list(score_dist.keys()),  
                "Peptide Count": list(score_dist.values())  
            }  
            pd.DataFrame(score_data).to_excel(writer, sheet_name="Score Distribution", index=False)  
        
        # Length distribution  
        if "length_distribution" in result["analysis"]:  
            length_dist = result["analysis"]["length_distribution"]  
            length_data = {  
                "Peptide Length": list(length_dist.keys()),  
                "Count": list(length_dist.values())  
            }  
            pd.DataFrame(length_data).to_excel(writer, sheet_name="Length Distribution", index=False)  
        
        # Amino acid composition  
        if "aa_composition" in result["analysis"]:  
            aa_comp = result["analysis"]["aa_composition"]  
            aa_data = {  
                "Amino Acid": list(aa_comp.keys()),  
                "Percentage (%)": [f"{val:.2f}" for val in aa_comp.values()]  
            }  
            pd.DataFrame(aa_data).to_excel(writer, sheet_name="AA Composition", index=False)  
        
        # High activity peptides  
        if "analysis" in result and "high_activity_peptides" in result["analysis"]:  
            high_activity_data = []  
            for peptide, score in result["analysis"]["high_activity_peptides"]:  
                high_activity_data.append({  
                    "Peptide Sequence": peptide,  
                    f"{target_activity} Score": score,  
                    "Length": len(peptide),  
                    "High Activity (≥0.7)": "Yes" if score >= 0.7 else "No"  
                })  
            
            if high_activity_data:  
                pd.DataFrame(high_activity_data).to_excel(writer, sheet_name="High Activity Peptides", index=False)  
        
        # Optimization history  
        history_df = pd.DataFrame({  
            "Iteration": list(range(1, len(result["fitness_history"]) + 1)),  
            "Fitness": result["fitness_history"]  
        })  
        history_df.to_excel(writer, sheet_name="Optimization History", index=False)  
    
    print(f"Optimization results exported to Excel: {output_excel}")  
    
    # Generate PDF report if requested  
    if output_pdf is None:  
        output_pdf = os.path.join(OUTPUT_DIR, f"enzyme_optimization_report_{target_activity}_{timestamp}.pdf")  
    else:  
        output_pdf = os.path.join(OUTPUT_DIR, output_pdf)  
    
    try:  
        # 使用完整PDF报告  
        generate_pdf_report(result, protein_sequence, output_pdf)  
        print(f"Comprehensive PDF report generated: {output_pdf}")  
    except Exception as e:  
        logging.error(f"Failed to generate comprehensive PDF report: {e}")  
        try:  
            # 回退到简单报告  
            generate_simple_pdf_report(result, protein_sequence, output_pdf)  
            print(f"Simple PDF report generated as fallback: {output_pdf}")  
        except Exception as e2:  
            logging.error(f"Failed to generate simple PDF report: {e2}")  
            output_pdf = None  
    
    return output_excel, output_pdf  


# 主函数: 用于VSCode调试  
if __name__ == "__main__":  
    print("==== Food Bioactive Peptide Simulated Annealing Optimizer ====")  
    print(f"Output directory: {OUTPUT_DIR}")  
    
    # Test protein sequence - Apolipoprotein A1 (ApoA1)  
    test_sequence = "MKVLWAALLVTFLAGCQAKVEQAVETEPEPELRQQTEWQSGQRWELALGRFWDYLRWVQTLSEQVQEELLSSQVTQELRALMDETMKELKAYKSELEEQLTPVAEETRARLSKELQAAQARLGADVLASHGRLVQYRGEVQAMLGQSTEELRVRLASHLRKLRKRLLRDADDLQKRLAVYQAGAREGAERGLSAIRERLGPLVEQGRVRAATVGSLAGQPLQERAQAWGERLRARMEEMGSRTRDRLDEVKEQVAEVRAKLEEQAQQRLEEQLGMDTQKEIMDLQARKASIRAQDVHEPSEWRNRLLLLETQAGEGN"  
    
    # Print all available enzyme types  
    print("\nAvailable enzyme types:")  
    for i, enzyme in enumerate(get_available_enzymes(), 1):  
        print(f"{i}. {enzyme}")  
    
    # Print all predictable activity types  
    available_activities = get_available_activities()  
    print("\nPredictable activity types:")  
    for i, activity in enumerate(available_activities, 1):  
        print(f"{i}. {activity}")  
    
    # Select enzymes to use  
    # For actual use, let the user select or input  
    selected_enzymes = ['Trypsin', 'Pepsin_1', 'GluC', 'LysC']  
    print(f"\nSelected enzymes: {selected_enzymes}")  
    
    # Select target activity  
    # For actual use, let the user select  
    target_activity = 'antioxidative'  # Can be 'ACE inhibitor', 'antibacterial', 'antioxidative', etc.  
    print(f"Target activity: {target_activity}")  
    
    # Run optimization  
    print("\nStarting optimization...")  
    
    # Set breakpoints here for debugging  
    result = optimize_enzyme_combination(  
        protein_sequence=test_sequence,  
        target_activity=target_activity,  
        available_enzymes=selected_enzymes,  
        max_iterations=500,  # Set number of iterations  
        initial_temp=100.0,  # Initial temperature  
        cooling_rate=0.95,   # Cooling rate  
        device='cuda' if torch.cuda.is_available() else 'cpu',  
        min_peptide_length=2,  
        max_peptide_length=50,  
        activity_threshold=0.7,  
        export_figures=False  # 禁用单独图表导出，避免错误  
    )  
    
    # Export results to Excel and PDF  
    excel_path, pdf_path = export_optimization_results(  
        result=result,  
        protein_sequence=test_sequence  
    )  
    
    # Print optimal enzyme combination  
    print("\nOptimal enzyme combination:")  
    for enzyme in result["best_enzymes"]:  
        print(f"- {enzyme}")  
        
    print(f"\nFitness score: {result['best_fitness']:.4f}")  
    print(f"Generated peptides: {result['analysis']['peptide_count']}")  
    print(f"Valid peptides: {result['analysis']['valid_peptide_count']}")  
    
    # Print high activity peptides  
    print(f"\nHigh {target_activity} activity peptides:")  
    for i, (peptide, score) in enumerate(result["analysis"]["high_activity_peptides"][:5]):  
        print(f"{i+1}. {peptide} (score: {score:.4f}, length: {len(peptide)})")  
    
    # Print high activity peptide statistics  
    print(f"\nHigh activity peptide statistics (activity ≥ 0.7):")  
    print(f"Count: {result['analysis']['high_activity_count']} peptides")  
    print(f"Percentage: {result['analysis']['high_activity_percentage']:.1f}%")  
    
    print("\nOptimization complete! All results exported to output directory.")  
    print(f"Excel file: {excel_path}")  
    if pdf_path:  
        print(f"PDF report: {pdf_path}")  