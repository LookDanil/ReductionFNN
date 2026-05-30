"""
Модуль нечеткой системы: функции принадлежности и визуализация
"""

import numpy as np
from typing import List, Tuple
import os

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class TrapezoidalMF:
    """Трапециевидная функция принадлежности"""
    __slots__ = ('a', 'b', 'c', 'd')
    
    def __init__(self, a: float, b: float, c: float, d: float):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
    
    def __call__(self, x: float) -> float:
        if x <= self.a or x >= self.d:
            return 0.0
        elif self.a < x < self.b:
            return (x - self.a) / (self.b - self.a) if self.b > self.a else 0.0
        elif self.b <= x <= self.c:
            return 1.0
        else:
            return (self.d - x) / (self.d - self.c) if self.d > self.c else 0.0
    
    def get_params(self) -> Tuple[float, float, float, float]:
        return (self.a, self.b, self.c, self.d)
    
    def __repr__(self):
        return f"TrapezoidalMF(a={self.a:.3f}, b={self.b:.3f}, c={self.c:.3f}, d={self.d:.3f})"


class MembershipFunctionVisualizer:
    """Визуализация функций принадлежности"""
    
    @staticmethod
    def plot_membership_functions(
        mfs: List[List],
        feature_idx: int,
        feature_name: str = None,
        title: str = "Функции принадлежности",
        save_path: str = None
    ):
        if not MATPLOTLIB_AVAILABLE:
            return None, None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = list(mcolors.TABLEAU_COLORS.values())
        feature_mfs = mfs[feature_idx]
        
        x_range = np.linspace(-0.1, 1.1, 1000)
        
        for term_idx, mf in enumerate(feature_mfs):
            y_values = [mf(x) for x in x_range]
            color = colors[term_idx % len(colors)]
            ax.plot(x_range, y_values, color=color, linewidth=2,
                    label=f'Терм {term_idx + 1}')
            a, b, c, d = mf.get_params()
            ax.plot([a, b, c, d], [0, 1, 1, 0], 'o', color=color, markersize=6)
        
        ax.set_xlabel('Нормализованное значение признака', fontsize=12)
        ax.set_ylabel('Степень принадлежности', fontsize=12)
        title_text = f"{title} - {feature_name}" if feature_name else title
        ax.set_title(title_text, fontsize=14)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlim(-0.05, 1.05)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig, ax
    
    @staticmethod
    def plot_all_features(
        mfs: List[List],
        feature_names: List[str] = None,
        title_prefix: str = "",
        save_dir: str = None
    ):
        if not MATPLOTLIB_AVAILABLE:
            return
        
        n_features = len(mfs)
        for f_idx in range(n_features):
            f_name = feature_names[f_idx] if feature_names else f"Признак {f_idx + 1}"
            save_path = None
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"feature_{f_idx}.png")
            
            MembershipFunctionVisualizer.plot_membership_functions(
                mfs, f_idx, f_name, f"{title_prefix} {f_name}", save_path
            )
    
    @staticmethod
    def plot_comparison(
        mfs_before: List[List],
        mfs_after: List[List],
        feature_idx: int,
        feature_name: str = None,
        save_path: str = None
    ):
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        colors = list(mcolors.TABLEAU_COLORS.values())
        
        x_range = np.linspace(-0.1, 1.1, 1000)
        
        # До
        feature_mfs = mfs_before[feature_idx]
        for term_idx, mf in enumerate(feature_mfs):
            y_values = [mf(x) for x in x_range]
            color = colors[term_idx % len(colors)]
            ax1.plot(x_range, y_values, color=color, linewidth=2,
                    label=f'Терм {term_idx + 1}')
            a, b, c, d = mf.get_params()
            ax1.plot([a, b, c, d], [0, 1, 1, 0], 'o', color=color, markersize=6)
        
        ax1.set_xlabel('Нормализованное значение', fontsize=12)
        ax1.set_ylabel('Степень принадлежности', fontsize=12)
        ax1.set_title('ДО обучения', fontsize=14)
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-0.1, 1.1)
        
        # После
        feature_mfs = mfs_after[feature_idx]
        for term_idx, mf in enumerate(feature_mfs):
            y_values = [mf(x) for x in x_range]
            color = colors[term_idx % len(colors)]
            ax2.plot(x_range, y_values, color=color, linewidth=2,
                    label=f'Терм {term_idx + 1}')
            a, b, c, d = mf.get_params()
            ax2.plot([a, b, c, d], [0, 1, 1, 0], 'o', color=color, markersize=6)
        
        ax2.set_xlabel('Нормализованное значение', fontsize=12)
        ax2.set_ylabel('Степень принадлежности', fontsize=12)
        ax2.set_title('ПОСЛЕ обучения', fontsize=14)
        ax2.legend(loc='upper right', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(-0.1, 1.1)
        
        fig.suptitle(f'Сравнение ФП: {feature_name if feature_name else f"Признак {feature_idx+1}"}',
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def _draw_on_canvas(canvas, feature_mfs, feature_name):
        """Рисует функции принадлежности на переданном холсте matplotlib"""
        import matplotlib.colors as mcolors
        
        ax = canvas.fig.add_subplot(111)
        colors = list(mcolors.TABLEAU_COLORS.values())
        
        # Определяем диапазон X по фактическим параметрам ФП
        all_params = []
        for mf in feature_mfs:
            a, b, c, d = mf.get_params()
            all_params.extend([a, b, c, d])
        
        x_min = min(all_params)
        x_max = max(all_params)
        margin = (x_max - x_min) * 0.1
        x_range = np.linspace(x_min - margin, x_max + margin, 1000)
        
        for term_idx, mf in enumerate(feature_mfs):
            a, b, c, d = mf.get_params()
            y_values = [mf(x) for x in x_range]
            color = colors[term_idx % len(colors)]
            
            # Основная линия трапеции
            ax.plot(x_range, y_values, color=color, linewidth=2,
                    label=f'Терм {term_idx + 1}')
            
            # Точки на углах трапеции
            ax.plot([a, b, c, d], [0, 1, 1, 0], 'o', color=color, markersize=6)
            
            # Пунктирные линии от b и c к оси X
            ax.plot([b, b], [0, 1], '--', color=color, linewidth=1, alpha=0.6)
            ax.plot([c, c], [0, 1], '--', color=color, linewidth=1, alpha=0.6)
            
            # Подписи значений b и c
            y_offset = -0.08  # Смещение под осью X
            ax.annotate(f'{b:.3f}', 
                        xy=(b, 0), 
                        xytext=(b, y_offset),
                        textcoords='data',
                        ha='center', va='top',
                        fontsize=8, color=color,
                        arrowprops=dict(arrowstyle='-', color=color, lw=0.5))
            
            ax.annotate(f'{c:.3f}',
                        xy=(c, 0),
                        xytext=(c, y_offset),
                        textcoords='data',
                        ha='center', va='top',
                        fontsize=8, color=color,
                        arrowprops=dict(arrowstyle='-', color=color, lw=0.5))
        
        ax.set_xlabel('Значение признака', fontsize=10)
        ax.set_ylabel('Степень принадлежности', fontsize=10)
        ax.set_title(f'Функции принадлежности: {feature_name}', fontsize=12)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.2, 1.1)  # Чуть больше места снизу для подписей
        
        canvas.fig.tight_layout()
        canvas.draw()