"""Вкладка Этапа 3: Обучение модели"""
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QSplitter
)
from PyQt6.QtCore import Qt

from gui.components.tuning_params_panel import TuningParamsPanel
from gui.components.tuning_results_panel import TuningResultsPanel
from gui.components.visualization_panel import VisualizationPanel


class Stage3Tab(QWidget):
    """Содержимое вкладки «Обучение модели»"""
    
    def __init__(self, main_window):
        super().__init__()
        self.mw = main_window
        self._build_ui()
    
    def _build_ui(self):
        layout = QHBoxLayout(self)
        left_panel = QVBoxLayout()
        
        # Параметры ГА3
        self.mw.tuning_params_panel = TuningParamsPanel()
        left_panel.addWidget(self.mw.tuning_params_panel)
        
        # Результаты ГА3
        self.mw.tuning_results_panel = TuningResultsPanel()
        left_panel.addWidget(self.mw.tuning_results_panel)
        
        # Кнопки НАЧАТЬ / ОСТАНОВИТЬ
        buttons_widget = QWidget()
        buttons_layout = QHBoxLayout(buttons_widget)
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        
        self.mw.btn_start_stage3 = QPushButton("НАЧАТЬ\nобучение модели")
        self.mw.btn_start_stage3.setObjectName("startButton")
        self.mw.btn_start_stage3.setMinimumHeight(60)
        self.mw.btn_start_stage3.clicked.connect(self.mw._on_start_stage3)
        
        self.mw.btn_stop_stage3 = QPushButton("ОСТАНОВИТЬ\nобучение модели")
        self.mw.btn_stop_stage3.setObjectName("stopButton")
        self.mw.btn_stop_stage3.setMinimumHeight(60)
        self.mw.btn_stop_stage3.setEnabled(False)
        self.mw.btn_stop_stage3.clicked.connect(self.mw._on_stop_stage3)
        
        buttons_layout.addWidget(self.mw.btn_start_stage3)
        buttons_layout.addWidget(self.mw.btn_stop_stage3)
        left_panel.addWidget(buttons_widget)
        
        # Информационные кнопки
        self.mw.info_buttons_stage3 = []
        info_data = [
            ("Показать структуру модели", self.mw._show_structure_stage3),
            ("Показать функции принадлежности", self.mw._show_mf_stage3),
            ("Показать метрики качества классификации", self.mw._show_metrics_stage3),
        ]
        for text, handler in info_data:
            btn = QPushButton(text)
            btn.clicked.connect(handler)
            btn.setEnabled(False)
            self.mw.info_buttons_stage3.append(btn)
            left_panel.addWidget(btn)
        
        left_panel.addStretch()
        
        # График
        right_panel = QVBoxLayout()
        self.mw.visualization_panel_stage3 = VisualizationPanel()
        self.mw.visualization_panel_stage3.setTitle("Графики работы ГА3 (Обучение)")
        right_panel.addWidget(self.mw.visualization_panel_stage3)
        
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