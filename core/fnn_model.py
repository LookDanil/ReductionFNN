"""
Модуль оптимизированной нечеткой нейронной сети
Единая логика: MIN-композиция, CF = Σmin/Nk, без нормализации
"""

import numpy as np
from typing import List, Tuple
from sklearn.metrics import accuracy_score


class OptimizedReducedFuzzyNeuralNetwork:
    """FNN с MIN-композицией и единой логикой классификации"""
    
    def __init__(self, n_features: int, n_classes: int,
                 gradations: List[int], membership_funcs: List[List],
                 active_rules: List[tuple], active_cfs: List[np.ndarray]):
        
        self.n_features = n_features
        self.n_classes = n_classes
        self.gradations = gradations
        self.membership_funcs = membership_funcs
        self.active_rules = active_rules
        self.active_cfs = active_cfs
        
        self._precompute_rule_indices()
    
    def _precompute_rule_indices(self):
        self.rule_term_indices = np.array(self.active_rules, dtype=np.int32)
        self.rule_cfs_array = np.array(self.active_cfs)
    
    def _vectorized_activation(self, X: np.ndarray) -> np.ndarray:
        """MIN-композиция для активации правил"""
        n_samples = X.shape[0]
        n_rules = len(self.active_rules)
        
        # Инициализируем единицами (для MIN)
        activations = np.ones((n_samples, n_rules))
        
        for f_idx in range(self.n_features):
            feature_values = X[:, f_idx]
            feature_mfs = self.membership_funcs[f_idx]
            n_terms = len(feature_mfs)
            
            term_activations = np.zeros((n_samples, n_terms))
            
            for t_idx, mf in enumerate(feature_mfs):
                a, b, c, d = mf.get_params()
                act = np.zeros(n_samples)
                
                mask_left = (feature_values > a) & (feature_values < b)
                if b > a:
                    act[mask_left] = (feature_values[mask_left] - a) / (b - a)
                
                mask_plateau = (feature_values >= b) & (feature_values <= c)
                act[mask_plateau] = 1.0
                
                mask_right = (feature_values > c) & (feature_values < d)
                if d > c:
                    act[mask_right] = (d - feature_values[mask_right]) / (d - c)
                
                term_activations[:, t_idx] = act
            
            rule_terms = self.rule_term_indices[:, f_idx]
            
            for r_idx in range(n_rules):
                t_idx = rule_terms[r_idx]
                # MIN-композиция
                activations[:, r_idx] = np.minimum(activations[:, r_idx], term_activations[:, t_idx])
        
        return activations
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Предсказание: сумма act×CF → argmax (без нормализации)"""
        activations = self._vectorized_activation(X)
        
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples, dtype=int)
        
        for i in range(n_samples):
            acts = activations[i]
            
            # Суммируем act × CF для всех правил (без фильтрации)
            confidences = np.sum(acts[:, np.newaxis] * self.rule_cfs_array, axis=0)
            
            if np.max(confidences) > 0:
                predictions[i] = np.argmax(confidences)
            else:
                predictions[i] = np.random.randint(self.n_classes)
        
        return predictions
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """Оценка точности"""
        pred = self.predict(X)
        return accuracy_score(y, pred)
    
    def get_membership_functions(self) -> List[List]:
        return self.membership_funcs