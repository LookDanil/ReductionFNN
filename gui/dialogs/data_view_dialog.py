"""Диалог просмотра данных"""
import numpy as np
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QTableWidget,
    QTableWidgetItem, QHeaderView, QDialogButtonBox
)
from PyQt6.QtCore import Qt


class DataViewDialog(QDialog):
    def __init__(self, X, y, feature_names, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Просмотр набора данных")
        self.setMinimumSize(800, 500)
        layout = QVBoxLayout(self)
        info_label = QLabel(f"Примеров: {X.shape[0]} | Признаков: {X.shape[1]} | Классов: {len(np.unique(y))}")
        info_label.setStyleSheet("font-weight: bold; font-size: 12px; margin-bottom: 10px;")
        layout.addWidget(info_label)
        table = QTableWidget()
        table.setRowCount(min(X.shape[0], 100))
        table.setColumnCount(X.shape[1] + 1)
        headers = list(feature_names) + ["Класс"]
        table.setHorizontalHeaderLabels(headers)
        for i in range(min(X.shape[0], 100)):
            for j in range(X.shape[1]):
                item = QTableWidgetItem(f"{X[i, j]:.4f}")
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                table.setItem(i, j, item)
            class_item = QTableWidgetItem(str(int(y[i])))
            class_item.setFlags(class_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            table.setItem(i, X.shape[1], class_item)
        table.resizeColumnsToContents()
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(table)
        if X.shape[0] > 100:
            note = QLabel(f"Показаны первые 100 строк из {X.shape[0]}")
            note.setStyleSheet("color: #999; font-style: italic;")
            layout.addWidget(note)
        unique, counts = np.unique(y, return_counts=True)
        stats_text = "Распределение классов:\n"
        for cls, cnt in zip(unique, counts):
            stats_text += f"  Класс {int(cls)}: {cnt} примеров ({cnt/len(y)*100:.1f}%)\n"
        stats_label = QLabel(stats_text)
        stats_label.setStyleSheet("margin-top: 10px;")
        layout.addWidget(stats_label)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)