"""
Модуль загрузки и подготовки данных
"""

import numpy as np
from sklearn.datasets import load_iris, load_wine


class GroupBootstrapSampler:
    """Групповой бутстрэп с замещением для разбиения данных"""
    
    @staticmethod
    def split_with_replacement(X: np.ndarray, y: np.ndarray, random_state: int = 42):
        np.random.seed(random_state)
        unique_classes = np.unique(y)
        
        X_train_parts, X_test_parts = [], []
        y_train_parts, y_test_parts = [], []
        
        for k in unique_classes:
            class_mask = (y == k)
            X_class = X[class_mask]
            y_class = y[class_mask]
            n_class = len(X_class)
            
            indices = np.arange(n_class)
            selected_indices = np.random.choice(indices, size=n_class, replace=True)
            selection_counts = np.bincount(selected_indices, minlength=n_class)
            
            X_train_parts.append(X_class[selected_indices])
            y_train_parts.append(y_class[selected_indices])
            
            not_selected_mask = selection_counts == 0
            X_test_parts.append(X_class[not_selected_mask])
            y_test_parts.append(y_class[not_selected_mask])
        
        X_train = np.vstack(X_train_parts)
        y_train = np.hstack(y_train_parts)
        X_test = np.vstack(X_test_parts)
        y_test = np.hstack(y_test_parts)
        
        train_shuffle = np.random.permutation(len(X_train))
        X_train = X_train[train_shuffle]
        y_train = y_train[train_shuffle]
        
        if len(X_test) > 0:
            test_shuffle = np.random.permutation(len(X_test))
            X_test = X_test[test_shuffle]
            y_test = y_test[test_shuffle]
        
        return X_train, X_test, y_train, y_test


def load_dataset(name: str):
    """Загрузка предустановленных датасетов"""
    if name.lower() == 'iris':
        data = load_iris()
    elif name.lower() == 'wine':
        data = load_wine()
    else:
        raise ValueError(f"Неизвестный датасет: {name}")
    
    return data.data, data.target, list(data.feature_names), list(data.target_names)


def split_data_by_bootstrap(X, y, random_state=42):
    """
    Разбиение бутстрэпом.
    Возвращает X_train, X_test, y_train, y_test, X_all, y_all
    где X_all — ИСХОДНЫЙ датасет (копия)
    """
    X_train, X_test, y_train, y_test = GroupBootstrapSampler.split_with_replacement(
        X, y, random_state
    )
    
    # ALL — это исходный датасет (копия)
    X_all = X.copy()
    y_all = y.copy()
    
    return X_train, X_test, y_train, y_test, X_all, y_all