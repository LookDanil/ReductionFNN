"""Диалог метрик качества"""
import numpy as np
import itertools
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QTableWidget,
    QTableWidgetItem, QHeaderView, QDialogButtonBox
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
from core.fnn_model import OptimizedReducedFuzzyNeuralNetwork


class MetricsDialog(QDialog):
    def __init__(self, X_train, y_train, X_test, y_test, membership_funcs, gradations, class_names, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Метрики качества классификации")
        self.setMinimumSize(950, 850)
        layout = QVBoxLayout(self)
        
        # Строим модель
        ranges = [range(g) for g in gradations]
        all_rules = list(itertools.product(*ranges))
        fnn = OptimizedReducedFuzzyNeuralNetwork(
            n_features=len(gradations), n_classes=len(class_names),
            gradations=gradations, membership_funcs=membership_funcs,
            active_rules=all_rules, active_cfs=[np.zeros(len(class_names)) for _ in all_rules]
        )
        
        # CF на TRAIN
        train_act = fnn._vectorized_activation(X_train)
        rule_cfs = np.zeros((len(all_rules), len(class_names)))
        for k in range(len(class_names)):
            class_mask = (y_train == k)
            Nk = np.sum(class_mask)
            if Nk > 0:
                for r_idx in range(len(all_rules)):
                    sum_act = np.sum(train_act[class_mask, r_idx])
                    if sum_act > 0:
                        rule_cfs[r_idx, k] = sum_act / Nk
        fnn.rule_cfs_array = rule_cfs
        
        # Предсказания
        y_train_pred = fnn.predict(X_train)
        y_test_pred = fnn.predict(X_test)
        
        # Информация
        info_label = QLabel(f"Градации: {gradations} | Правил: {int(np.prod(gradations))} | Классов: {len(class_names)}")
        info_label.setStyleSheet("font-weight: bold; font-size: 12px; margin-bottom: 10px;")
        layout.addWidget(info_label)
        
        # ==========================================
        # Общие метрики (macro average)
        # ==========================================
        avg_label = QLabel("Общие метрики (macro average):")
        avg_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        layout.addWidget(avg_label)
        
        table_avg = QTableWidget()
        table_avg.setRowCount(4)
        table_avg.setColumnCount(3)
        table_avg.setHorizontalHeaderLabels(["Метрика", "TRAIN (%)", "TEST (%)"])
        
        train_acc = accuracy_score(y_train, y_train_pred) * 100
        test_acc = accuracy_score(y_test, y_test_pred) * 100
        train_precision = precision_score(y_train, y_train_pred, average='macro', zero_division=0) * 100
        train_recall = recall_score(y_train, y_train_pred, average='macro', zero_division=0) * 100
        train_f1 = f1_score(y_train, y_train_pred, average='macro', zero_division=0) * 100
        test_precision = precision_score(y_test, y_test_pred, average='macro', zero_division=0) * 100
        test_recall = recall_score(y_test, y_test_pred, average='macro', zero_division=0) * 100
        test_f1 = f1_score(y_test, y_test_pred, average='macro', zero_division=0) * 100
        
        metrics = [
            ("Accuracy", train_acc, test_acc),
            ("Precision", train_precision, test_precision),
            ("Recall", train_recall, test_recall),
            ("F1-score", train_f1, test_f1),
        ]
        for i, (name, tr, te) in enumerate(metrics):
            table_avg.setItem(i, 0, QTableWidgetItem(name))
            table_avg.setItem(i, 1, QTableWidgetItem(f"{tr:.2f}"))
            table_avg.setItem(i, 2, QTableWidgetItem(f"{te:.2f}"))
        table_avg.resizeColumnsToContents()
        table_avg.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(table_avg)
        
        # ==========================================
        # Метрики по классам (TRAIN)
        # ==========================================
        train_class_label = QLabel("Метрики по классам (TRAIN):")
        train_class_label.setStyleSheet("font-weight: bold; margin-top: 15px;")
        layout.addWidget(train_class_label)
        
        prec_train_per_class = precision_score(y_train, y_train_pred, average=None, zero_division=0) * 100
        rec_train_per_class = recall_score(y_train, y_train_pred, average=None, zero_division=0) * 100
        f1_train_per_class = f1_score(y_train, y_train_pred, average=None, zero_division=0) * 100
        
        table_train_class = QTableWidget()
        table_train_class.setRowCount(len(class_names))
        table_train_class.setColumnCount(4)
        table_train_class.setHorizontalHeaderLabels(["Класс", "Precision (%)", "Recall (%)", "F1-score (%)"])
        for i, cls_name in enumerate(class_names):
            table_train_class.setItem(i, 0, QTableWidgetItem(str(cls_name)))
            table_train_class.setItem(i, 1, QTableWidgetItem(f"{prec_train_per_class[i]:.2f}" if i < len(prec_train_per_class) else "—"))
            table_train_class.setItem(i, 2, QTableWidgetItem(f"{rec_train_per_class[i]:.2f}" if i < len(rec_train_per_class) else "—"))
            table_train_class.setItem(i, 3, QTableWidgetItem(f"{f1_train_per_class[i]:.2f}" if i < len(f1_train_per_class) else "—"))
        table_train_class.resizeColumnsToContents()
        table_train_class.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(table_train_class)
        
        # ==========================================
        # Метрики по классам (TEST)
        # ==========================================
        test_class_label = QLabel("Метрики по классам (TEST):")
        test_class_label.setStyleSheet("font-weight: bold; margin-top: 15px;")
        layout.addWidget(test_class_label)
        
        prec_test_per_class = precision_score(y_test, y_test_pred, average=None, zero_division=0) * 100
        rec_test_per_class = recall_score(y_test, y_test_pred, average=None, zero_division=0) * 100
        f1_test_per_class = f1_score(y_test, y_test_pred, average=None, zero_division=0) * 100
        
        table_test_class = QTableWidget()
        table_test_class.setRowCount(len(class_names))
        table_test_class.setColumnCount(4)
        table_test_class.setHorizontalHeaderLabels(["Класс", "Precision (%)", "Recall (%)", "F1-score (%)"])
        for i, cls_name in enumerate(class_names):
            table_test_class.setItem(i, 0, QTableWidgetItem(str(cls_name)))
            table_test_class.setItem(i, 1, QTableWidgetItem(f"{prec_test_per_class[i]:.2f}" if i < len(prec_test_per_class) else "—"))
            table_test_class.setItem(i, 2, QTableWidgetItem(f"{rec_test_per_class[i]:.2f}" if i < len(rec_test_per_class) else "—"))
            table_test_class.setItem(i, 3, QTableWidgetItem(f"{f1_test_per_class[i]:.2f}" if i < len(f1_test_per_class) else "—"))
        table_test_class.resizeColumnsToContents()
        table_test_class.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(table_test_class)
        
        # ==========================================
        # Матрицы ошибок
        # ==========================================
        cm_train = confusion_matrix(y_train, y_train_pred)
        cm_test = confusion_matrix(y_test, y_test_pred)
        fig = Figure(figsize=(10, 4))
        ax1 = fig.add_subplot(121)
        sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', ax=ax1, xticklabels=class_names, yticklabels=class_names)
        ax1.set_title('Матрица ошибок — TRAIN')
        ax1.set_xlabel('Классифицированное значение')
        ax1.set_ylabel('Истинное значение')
        ax2 = fig.add_subplot(122)
        sns.heatmap(cm_test, annot=True, fmt='d', cmap='Oranges', ax=ax2, xticklabels=class_names, yticklabels=class_names)
        ax2.set_title('Матрица ошибок — TEST')
        ax2.set_xlabel('Классифицированное значение')
        ax2.set_ylabel('Истинное значение')
        fig.tight_layout()
        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)
        
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.accept)
        layout.addWidget(buttons)