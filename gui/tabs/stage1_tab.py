"""Вкладка Этапа 1: Преднастройка модели"""
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QSplitter
)
from PyQt6.QtCore import Qt

from gui.components.dataset_panel import DatasetPanel
from gui.components.parameters_panel import ParametersPanel
from gui.components.results_panel import ResultsPanel
from gui.components.visualization_panel import VisualizationPanel


class Stage1Tab(QWidget):
    """Содержимое вкладки «Преднастройка модели»"""
    
    def __init__(self, main_window):
        super().__init__()
        self.mw = main_window  # Ссылка на MainWindow
        self._build_ui()
    
    def _build_ui(self):
        layout = QHBoxLayout(self)
        left_panel = QVBoxLayout()
        
        # Загрузка данных
        self.mw.dataset_panel = DatasetPanel()
        self.mw.dataset_panel.dataset_combo.currentIndexChanged.connect(self.mw._on_dataset_selected)
        self.mw.dataset_panel.features_spin.valueChanged.connect(self.mw._on_features_changed)
        self.mw.dataset_panel.btn_view_data.clicked.connect(self.mw._view_data)
        left_panel.addWidget(self.mw.dataset_panel)
        
        # Параметры ГА
        self.mw.params_panel = ParametersPanel()
        left_panel.addWidget(self.mw.params_panel)
        
        # Результаты
        self.mw.results_panel = ResultsPanel()
        left_panel.addWidget(self.mw.results_panel)
        
        # Кнопки НАЧАТЬ / ОСТАНОВИТЬ
        buttons_widget = QWidget()
        buttons_layout = QHBoxLayout(buttons_widget)
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        
        self.mw.btn_start = QPushButton("НАЧАТЬ\nпреднастройку модели")
        self.mw.btn_start.setObjectName("startButton")
        self.mw.btn_start.setMinimumHeight(60)
        self.mw.btn_start.clicked.connect(self.mw._on_start_stage1)
        
        self.mw.btn_stop = QPushButton("ОСТАНОВИТЬ\nпреднастройку модели")
        self.mw.btn_stop.setObjectName("stopButton")
        self.mw.btn_stop.setMinimumHeight(60)
        self.mw.btn_stop.setEnabled(False)
        self.mw.btn_stop.clicked.connect(self.mw._on_stop_stage1)
        
        buttons_layout.addWidget(self.mw.btn_start)
        buttons_layout.addWidget(self.mw.btn_stop)
        left_panel.addWidget(buttons_widget)
        
        # Информационные кнопки
        self.mw.info_buttons = []
        info_data = [
            ("Показать структуру модели", self.mw._show_structure),
            ("Показать функции принадлежности", self.mw._show_membership_functions),
            ("Показать метрики качества", self.mw._show_metrics),
        ]
        for text, handler in info_data:
            btn = QPushButton(text)
            btn.clicked.connect(handler)
            btn.setEnabled(False)
            self.mw.info_buttons.append(btn)
            left_panel.addWidget(btn)
        
        left_panel.addStretch()
        
        # График
        right_panel = QVBoxLayout()
        self.mw.visualization_panel = VisualizationPanel()
        right_panel.addWidget(self.mw.visualization_panel)
        
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