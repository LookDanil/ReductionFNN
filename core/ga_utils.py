"""Вспомогательные функции для ГА: сериализация, параллельное вычисление"""
from typing import List, Tuple
from core.fuzzy_system import TrapezoidalMF
from core.fnn_model import OptimizedReducedFuzzyNeuralNetwork


def serialize_mfs(mfs: List[List]) -> List[List[Tuple]]:
    data = []
    for f_mfs in mfs:
        f_data = []
        for mf in f_mfs:
            f_data.append(mf.get_params())
        data.append(f_data)
    return data


def deserialize_mfs(data: List[List[Tuple]]) -> List[List]:
    mfs = []
    for f_data in data:
        f_mfs = []
        for params in f_data:
            f_mfs.append(TrapezoidalMF(*params))
        mfs.append(f_mfs)
    return mfs


def evaluate_single_value_process(args):
    (value, feature_idx, grad_idx, param_name,
     mfs_data, X_all, y_all,
     active_rules, active_cfs, n_classes) = args
    mfs = deserialize_mfs(mfs_data)
    a, b, c, d = mfs[feature_idx][grad_idx].get_params()
    if param_name == 'a': a = value
    elif param_name == 'b': b = value
    elif param_name == 'c': c = value
    elif param_name == 'd': d = value
    mfs[feature_idx][grad_idx] = TrapezoidalMF(a, b, c, d)
    gradations = [len(f_mfs) for f_mfs in mfs]
    temp_fnn = OptimizedReducedFuzzyNeuralNetwork(
        n_features=len(mfs), n_classes=n_classes,
        gradations=gradations, membership_funcs=mfs,
        active_rules=active_rules, active_cfs=active_cfs
    )
    return temp_fnn.evaluate(X_all, y_all)