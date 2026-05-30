"""Панель визуализации с графиком работы ГА"""
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PyQt6.QtWidgets import QGroupBox, QVBoxLayout


class VisualizationPanel(QGroupBox):
    """Панель с графиком точности ГА"""
    
    def __init__(self):
        super().__init__("Графики работы генетического алгоритма")
        layout = QVBoxLayout(self)
        
        self.figure = Figure(figsize=(8, 5), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        
        self.train_gens = []
        self.train_values = []
        self.test_gens = []
        self.test_values = []
        
        self.ax = self.figure.add_subplot(111)
        self._init_plot()
    
    def _init_plot(self):
        self.ax.clear()
        self.ax.set_xlabel('Эпоха генетического алгоритма', fontsize=10)
        self.ax.set_ylabel('Точность модели (%)', fontsize=10)
        self.ax.set_title('Точность модели (%)', fontsize=12, fontweight='bold')
        self.ax.grid(True, alpha=0.3)
        
        # Только TRAIN и TEST
        self.train_line, = self.ax.plot([], [], 'g-', linewidth=2, label='Train')
        self.test_line, = self.ax.plot([], [], 'r--', linewidth=2, label='Test')
        self.ax.legend(loc='lower right')
        
        self.canvas.draw()
    
    def add_train_test(self, generation, train_acc, test_acc):
        """Добавить train/test"""
        self.train_gens.append(generation)
        self.train_values.append(train_acc)
        self.test_gens.append(generation)
        self.test_values.append(test_acc)
        self._redraw()
    
    def _redraw(self):
        if self.train_gens:
            self.train_line.set_data(self.train_gens, self.train_values)
        
        if self.test_gens:
            self.test_line.set_data(self.test_gens, self.test_values)
        
        self.ax.relim()
        self.ax.autoscale_view(scalex=True, scaley=True)
        
        all_values = self.train_values + self.test_values
        if all_values:
            y_min = min(all_values)
            y_max = max(all_values)
            margin = (y_max - y_min) * 0.1 if y_max > y_min else 5
            self.ax.set_ylim(max(0, y_min - margin), min(105, y_max + margin))
        
        self.canvas.draw()
    
    def reset(self):
        self.train_gens = []
        self.train_values = []
        self.test_gens = []
        self.test_values = []
        self._init_plot()