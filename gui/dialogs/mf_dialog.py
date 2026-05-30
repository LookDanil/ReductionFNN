"""Диалог функций принадлежности"""
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QTabWidget, QWidget,
    QLabel, QFileDialog, QMessageBox, QDialogButtonBox
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from core.fuzzy_system import MembershipFunctionVisualizer


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)


class MFDialog(QDialog):
    def __init__(self, mfs, feature_names, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Функции принадлежности")
        self.setMinimumSize(900, 650)
        layout = QVBoxLayout(self)
        info = QLabel(f"Признаков: {len(mfs)}")
        info.setStyleSheet("font-weight: bold; font-size: 12px; margin-bottom: 10px;")
        layout.addWidget(info)
        tab_widget = QTabWidget()
        for f_idx, feature_mfs in enumerate(mfs):
            tab = QWidget()
            tab_layout = QVBoxLayout(tab)
            canvas = MplCanvas(self, width=10, height=6)
            toolbar = NavigationToolbar(canvas, self)
            tab_layout.addWidget(toolbar)
            f_name = feature_names[f_idx] if feature_names else f"Признак {f_idx + 1}"
            MembershipFunctionVisualizer._draw_on_canvas(canvas, feature_mfs, f_name)
            tab_layout.addWidget(canvas)
            tab_widget.addTab(tab, f_name)
        layout.addWidget(tab_widget)
        buttons = QDialogButtonBox()
        save_btn = buttons.addButton("Сохранить все графики", QDialogButtonBox.ButtonRole.ActionRole)
        save_btn.clicked.connect(lambda: self._save_all(mfs, feature_names))
        close_btn = buttons.addButton(QDialogButtonBox.StandardButton.Close)
        close_btn.clicked.connect(self.accept)
        layout.addWidget(buttons)
    
    def _save_all(self, mfs, feature_names):
        import os
        save_dir = QFileDialog.getExistingDirectory(self, "Выберите папку для сохранения")
        if not save_dir:
            return
        for f_idx, feature_mfs in enumerate(mfs):
            f_name = feature_names[f_idx] if feature_names else f"Признак_{f_idx + 1}"
            save_path = os.path.join(save_dir, f"{f_name}.png")
            MembershipFunctionVisualizer.plot_membership_functions(mfs, f_idx, f_name, save_path=save_path)
        QMessageBox.information(self, "Сохранено", f"Графики сохранены в:\n{save_dir}")