"""Вкладка Этапа 2: Редукция модели"""
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QSplitter
)
from PyQt6.QtCore import Qt

from gui.components.reduction_params_panel import ReductionParamsPanel
from gui.components.reduction_results_panel import ReductionResultsPanel
from gui.components.visualization_panel import VisualizationPanel


class Stage2Tab(QWidget):
    """Содержимое вкладки «Редукция модели»"""
    
    def __init__(self, main_window):
        super().__init__()
        self.mw = main_window
        self._build_ui()
    
    def _build_ui(self):
        layout = QHBoxLayout(self)
        left_panel = QVBoxLayout()
        
        # Параметры ГА2
        self.mw.reduction_params_panel = ReductionParamsPanel()
        left_panel.addWidget(self.mw.reduction_params_panel)
        
        # Результаты ГА2
        self.mw.reduction_results_panel = ReductionResultsPanel()
        left_panel.addWidget(self.mw.reduction_results_panel)
        
        # Кнопки НАЧАТЬ / ОСТАНОВИТЬ
        buttons_widget = QWidget()
        buttons_layout = QHBoxLayout(buttons_widget)
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        
        self.mw.btn_start_stage2 = QPushButton("НАЧАТЬ\nредукцию модели")
        self.mw.btn_start_stage2.setObjectName("startButton")
        self.mw.btn_start_stage2.setMinimumHeight(60)
        self.mw.btn_start_stage2.clicked.connect(self.mw._on_start_stage2)
        
        self.mw.btn_stop_stage2 = QPushButton("ОСТАНОВИТЬ\nредукцию модели")
        self.mw.btn_stop_stage2.setObjectName("stopButton")
        self.mw.btn_stop_stage2.setMinimumHeight(60)
        self.mw.btn_stop_stage2.setEnabled(False)
        self.mw.btn_stop_stage2.clicked.connect(self.mw._on_stop_stage2)
        
        buttons_layout.addWidget(self.mw.btn_start_stage2)
        buttons_layout.addWidget(self.mw.btn_stop_stage2)
        left_panel.addWidget(buttons_widget)
        
        # Информационные кнопки
        self.mw.info_buttons_stage2 = []
        info_data = [
            ("Показать структуру редуцированной модели", self.mw._show_structure_stage2),
            ("Показать функции принадлежности", self.mw._show_mf_stage2),
            ("Показать метрики качества", self.mw._show_metrics_stage2),
        ]
        for text, handler in info_data:
            btn = QPushButton(text)
            btn.clicked.connect(handler)
            btn.setEnabled(False)
            self.mw.info_buttons_stage2.append(btn)
            left_panel.addWidget(btn)
        
        left_panel.addStretch()
        
        # График
        right_panel = QVBoxLayout()
        self.mw.visualization_panel_stage2 = VisualizationPanel()
        self.mw.visualization_panel_stage2.setTitle("Графики работы ГА2 (Редукция)")
        right_panel.addWidget(self.mw.visualization_panel_stage2)
        
        left_widget = QWidget()
        left_widget.setLayout(left_panel)
        left_widget.setMaximumWidth(500)
        right_widget = QWidget()
        right_widget.setLayout(right_panel)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter)