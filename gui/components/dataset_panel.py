"""Панель загрузки данных"""
from PyQt6.QtWidgets import (
    QGroupBox, QVBoxLayout, QLabel, QComboBox,
    QSpinBox, QHBoxLayout, QPushButton
)
from PyQt6.QtCore import QSignalBlocker


class DatasetPanel(QGroupBox):
    def __init__(self):
        super().__init__("Загрузка данных для анализа")
        layout = QVBoxLayout(self)
        
        self.dataset_combo = QComboBox()
        with QSignalBlocker(self.dataset_combo):
            self.dataset_combo.addItems([
                "Выберите датасет...",
                "Iris (4 признака)",
                "Wine (13 признаков)",
                "Загрузить из TXT...",
            ])
        layout.addWidget(self.dataset_combo)
        
        # Контейнер для спиннера (можно скрывать целиком)
        self.features_container = QHBoxLayout()
        self.features_container.addWidget(QLabel("Признаков:"))
        self.features_spin = QSpinBox()
        self.features_spin.setRange(1, 13)
        self.features_spin.setValue(4)
        self.features_container.addWidget(self.features_spin)
        self.features_container.addStretch()
        layout.addLayout(self.features_container)
        
        self.info_label = QLabel("Данные не загружены")
        self.info_label.setStyleSheet("color: #999; font-style: italic; padding: 5px;")
        layout.addWidget(self.info_label)
        
        # Кнопка просмотра данных (изначально скрыта)
        self.btn_view_data = QPushButton("Посмотреть набор данных")
        self.btn_view_data.setVisible(False)
        layout.addWidget(self.btn_view_data)
        
        layout.addStretch()
    
    def set_info(self, text: str):
        self.info_label.setText(text)
        self.info_label.setStyleSheet("color: #2e7d32; font-weight: bold; padding: 5px;")
    
    def get_n_features(self) -> int:
        return self.features_spin.value()
    
    def show_view_button(self, show: bool = True):
        """Показать/скрыть кнопку просмотра данных"""
        self.btn_view_data.setVisible(show)
    
    def set_features_visible(self, visible: bool):
        """Показать/скрыть выбор количества признаков"""
        # Скрываем/показываем все виджеты в контейнере
        for i in range(self.features_container.count()):
            widget = self.features_container.itemAt(i).widget()
            if widget:
                widget.setVisible(visible)