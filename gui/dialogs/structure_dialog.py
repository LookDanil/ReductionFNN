"""Диалог структуры модели"""
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QDialogButtonBox, QSizePolicy
)


class StructureDialog(QDialog):
    def __init__(self, gradations, rules_count, class_names, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Структура модели")
        self.setMinimumSize(900, 600)
        layout = QVBoxLayout(self)
        info = QLabel(f"Градации: {gradations} | Правил: {rules_count} | Классов: {len(class_names)}")
        info.setStyleSheet("font-weight: bold; font-size: 12px; margin-bottom: 10px;")
        layout.addWidget(info)
        self.figure = Figure(figsize=(12, 6))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self._draw_structure(self.figure, gradations, len(class_names), rules_count)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.accept)
        layout.addWidget(buttons)
    
    def _draw_structure(self, fig, gradations, n_classes, rules_count):
        ax = fig.add_subplot(111)
        n_features = len(gradations)
        input_spacing = 4.0
        grad_spacing = 0.9
        x_min = -0.5
        x_max = (n_features - 1) * input_spacing + 1.5
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(-2, 7)
        ax.axis('off')
        ax.set_aspect('auto')
        input_color = '#4CAF50'
        grad_color = '#2196F3'
        ant_color = '#FF9800'
        cons_color = '#F44336'
        line_color = '#000000'
        
        y_input = 6
        ax.text(x_min + 0.3, y_input + 0.8, 'Входные нейроны', fontsize=10, fontweight='bold', color=input_color)
        for i in range(n_features):
            x = i * input_spacing + 0.5
            circle = plt.Circle((x, y_input), 0.35, color=input_color, ec='white', linewidth=2, zorder=3)
            ax.add_patch(circle)
            ax.text(x, y_input, f'X{i+1}', ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        
        y_grad = 4
        grad_positions = []
        ax.text(x_min + 0.3, y_grad + 0.8, 'Нейроны градаций', fontsize=10, fontweight='bold', color=grad_color)
        for f_idx, g in enumerate(gradations):
            input_x = f_idx * input_spacing + 0.5
            start_x = input_x - (g - 1) * grad_spacing / 2
            for g_idx in range(g):
                x = start_x + g_idx * grad_spacing
                grad_positions.append((x, y_grad, f_idx, g_idx))
                square = plt.Rectangle((x - 0.3, y_grad - 0.3), 0.6, 0.6, color=grad_color, ec='white', linewidth=2, zorder=3)
                ax.add_patch(square)
                ax.text(x, y_grad, f'{g_idx+1}', ha='center', va='center', fontsize=7, color='white', fontweight='bold')
                ax.plot([input_x, x], [y_input - 0.35, y_grad + 0.3], '-', color=line_color, linewidth=0.8, alpha=0.7, zorder=1)
        
        y_ant = 2
        all_x = [p[0] for p in grad_positions]
        ant_x = (min(all_x) + max(all_x)) / 2
        ant_width = (max(all_x) - min(all_x)) * 0.7 + 1.5
        ant_height = 1.2
        ant_rect = FancyBboxPatch((ant_x - ant_width/2, y_ant - ant_height/2), ant_width, ant_height,
                                  boxstyle="round,pad=0.15", color=ant_color, ec='white', linewidth=2.5, zorder=3, alpha=0.9)
        ax.add_patch(ant_rect)
        ax.text(ant_x, y_ant + 0.15, 'Нейроны-антецеденты', ha='center', va='center', fontsize=11, fontweight='bold', color='white')
        ax.text(ant_x, y_ant - 0.25, f'{rules_count} правил', ha='center', va='center', fontsize=14, fontweight='bold', color='white')
        for gx, gy, _, _ in grad_positions:
            ax.plot([gx, ant_x], [gy - 0.3, y_ant + ant_height/2], '-', color=line_color, linewidth=0.6, alpha=0.5, zorder=1)
        
        y_cons = 0
        ax.text(x_min + 0.3, y_cons + 0.8, 'Консеквенты (классы)', fontsize=10, fontweight='bold', color=cons_color)
        cons_spacing = max(1.5, ant_width / max(n_classes, 1) * 0.8)
        cons_start = ant_x - (n_classes - 1) * cons_spacing / 2
        for c in range(n_classes):
            x = cons_start + c * cons_spacing
            triangle = plt.Polygon([(x, y_cons - 0.35), (x - 0.35, y_cons + 0.35), (x + 0.35, y_cons + 0.35)],
                                   color=cons_color, ec='white', linewidth=2, zorder=3)
            ax.add_patch(triangle)
            ax.text(x, y_cons, f'C{c+1}', ha='center', va='center', fontsize=9, color='white', fontweight='bold')
            for offset in [-0.3, 0, 0.3]:
                start_x = ant_x + offset * ant_width/3
                ax.plot([start_x, x], [y_ant - ant_height/2, y_cons + 0.35], '-', color=line_color, linewidth=0.6, alpha=0.4, zorder=1)
        
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=input_color, markersize=12, label='Входной нейрон'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=grad_color, markersize=12, label='Нейрон градации'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=ant_color, markersize=14, label='Блок антецедентов'),
            plt.Line2D([0], [0], marker='^', color='w', markerfacecolor=cons_color, markersize=12, label='Нейрон-консеквент'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.9)
        ax.autoscale_view()
        fig.tight_layout(pad=0.5)