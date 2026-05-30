"""Вкладка Этапа 4: Извлечение нечетких правил"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QLabel, QFileDialog, QMessageBox, QDialog, QDialogButtonBox
)
from PyQt6.QtCore import Qt
import numpy as np


class RulesDialog(QDialog):
    """Диалог для просмотра правил в текстовом виде"""
    def __init__(self, active_rules, active_cfs, feature_names, class_names, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Нечеткие правила в базе знаний")
        self.setMinimumSize(700, 500)
        
        layout = QVBoxLayout(self)
        
        # Формируем текст
        n_features = len(feature_names)
        lines = []
        
        # Заголовок
        headers = list(feature_names[:n_features]) + ["Класс", "CF"]
        lines.append('\t'.join(headers))
        lines.append('-' * 60)
        
        # Правила
        for rule, cf in zip(active_rules, active_cfs):
            terms = [str(t + 1) for t in rule]
            best_class = np.argmax(cf)
            class_name = class_names[best_class]
            cf_value = f"{cf[best_class]:.4f}"
            row = terms + [class_name, cf_value]
            lines.append('\t'.join(row))
        
        text = '\n'.join(lines)
        
        # Текстовое поле
        from PyQt6.QtWidgets import QTextEdit
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setPlainText(text)
        text_edit.setStyleSheet("font-family: Consolas, monospace; font-size: 12px;")
        layout.addWidget(text_edit)
        
        # Информация
        info = QLabel(f"Всего правил: {len(active_rules)}")
        info.setStyleSheet("font-size: 12px; color: #666; padding: 5px;")
        layout.addWidget(info)
        
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class Stage4Tab(QWidget):
    """Содержимое вкладки «Извлечение нечетких правил из модели»"""
    
    def __init__(self, main_window):
        super().__init__()
        self.mw = main_window
        self._active_rules = None
        self._active_cfs = None
        self._build_ui()
    
    def _build_ui(self):
        layout = QVBoxLayout(self)
        
        # Заголовок
        title = QLabel("База знаний")
        title.setStyleSheet("font-size: 18px; font-weight: bold; padding: 10px;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Кнопки
        buttons_layout = QHBoxLayout()
        
        self.btn_extract = QPushButton("ИЗВЛЕЧЬ\nнечеткие правила\nиз модели")
        self.btn_extract.setMinimumHeight(70)
        self.btn_extract.setStyleSheet("font-weight: bold; font-size: 13px;")
        self.btn_extract.clicked.connect(self._extract_rules)
        self.btn_extract.setEnabled(False)
        
        self.btn_show = QPushButton("ПОКАЗАТЬ\nнечеткие правила\nв базе знаний")
        self.btn_show.setMinimumHeight(70)
        self.btn_show.setStyleSheet("font-weight: bold; font-size: 13px;")
        self.btn_show.clicked.connect(self._show_rules)
        self.btn_show.setEnabled(False)
        
        buttons_layout.addWidget(self.btn_extract)
        buttons_layout.addWidget(self.btn_show)
        buttons_layout.addStretch()
        layout.addLayout(buttons_layout)
        
        # Информация
        self.info_label = QLabel("Выполните Этап 1 для извлечения правил.")
        self.info_label.setStyleSheet("font-size: 13px; color: #999; padding: 5px;")
        layout.addWidget(self.info_label)
        
        # Таблица правил (всегда видна, но пустая)
        self.table = QTableWidget()
        self.table.setRowCount(0)
        self.table.setColumnCount(0)
        layout.addWidget(self.table)
    
    def load_rules(self, stage1_results):
        """Загрузка правил после Этапа 1"""
        gradations = stage1_results['gradations']
        membership_funcs = stage1_results['membership_funcs']
        X_train = stage1_results['X_train']
        y_train = stage1_results['y_train']
        
        import itertools
        from core.fnn_model import OptimizedReducedFuzzyNeuralNetwork
        
        ranges = [range(g) for g in gradations]
        all_rules = list(itertools.product(*ranges))
        
        fnn = OptimizedReducedFuzzyNeuralNetwork(
            n_features=len(gradations),
            n_classes=len(self.mw.class_names),
            gradations=gradations,
            membership_funcs=membership_funcs,
            active_rules=all_rules,
            active_cfs=[np.zeros(len(self.mw.class_names)) for _ in all_rules]
        )
        activations = fnn._vectorized_activation(X_train)
        rule_cfs = np.zeros((len(all_rules), len(self.mw.class_names)))
        for k in range(len(self.mw.class_names)):
            class_mask = (y_train == k)
            Nk = np.sum(class_mask)
            if Nk > 0:
                for r_idx in range(len(all_rules)):
                    sum_act = np.sum(activations[class_mask, r_idx])
                    if sum_act > 0:
                        rule_cfs[r_idx, k] = sum_act / Nk
        
        # Только активные правила
        active_rules = []
        active_cfs = []
        for i, rule in enumerate(all_rules):
            if np.max(rule_cfs[i]) > 0:
                active_rules.append(rule)
                active_cfs.append(rule_cfs[i])
        
        self._active_rules = active_rules
        self._active_cfs = active_cfs
        
        # Заполняем таблицу
        n_features = len(gradations)
        n_cols = n_features + 2
        self.table.setRowCount(len(active_rules))
        self.table.setColumnCount(n_cols)
        
        headers = list(self.mw.feature_names[:n_features]) + ["Класс", "CF"]
        self.table.setHorizontalHeaderLabels(headers)
        
        for row_idx, (rule, cf) in enumerate(zip(active_rules, active_cfs)):
            for col_idx, term in enumerate(rule):
                item = QTableWidgetItem(str(term + 1))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(row_idx, col_idx, item)
            
            best_class = np.argmax(cf)
            class_name = self.mw.class_names[best_class]
            class_item = QTableWidgetItem(class_name)
            class_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            class_item.setFlags(class_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row_idx, n_features, class_item)
            
            cf_item = QTableWidgetItem(f"{cf[best_class]:.4f}")
            cf_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            cf_item.setFlags(cf_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row_idx, n_features + 1, cf_item)
        
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        
        self.info_label.setText(f"Правил в базе знаний: {len(active_rules)}")
        self.info_label.setStyleSheet("font-size: 13px; color: #2e7d32; font-weight: bold; padding: 5px;")
        
        self.btn_extract.setEnabled(True)
        self.btn_show.setEnabled(True)
    def load_reduced_rules(self, stage2_results, stage1_results):
        """Загрузка редуцированных правил после Этапа 2"""
        active_rules = stage2_results['active_rules']
        active_cfs = stage2_results['active_cfs']
        
        self._active_rules = active_rules
        self._active_cfs = active_cfs
        
        # Заполняем таблицу
        n_features = len(stage2_results['gradations'])
        n_cols = n_features + 2
        self.table.setRowCount(len(active_rules))
        self.table.setColumnCount(n_cols)
        
        headers = list(self.mw.feature_names[:n_features]) + ["Класс", "CF"]
        self.table.setHorizontalHeaderLabels(headers)
        
        for row_idx, (rule, cf) in enumerate(zip(active_rules, active_cfs)):
            for col_idx, term in enumerate(rule):
                item = QTableWidgetItem(str(term + 1))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(row_idx, col_idx, item)
            
            best_class = np.argmax(cf)
            class_name = self.mw.class_names[best_class]
            class_item = QTableWidgetItem(class_name)
            class_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            class_item.setFlags(class_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row_idx, n_features, class_item)
            
            cf_item = QTableWidgetItem(f"{cf[best_class]:.4f}")
            cf_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            cf_item.setFlags(cf_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row_idx, n_features + 1, cf_item)
        
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        
        self.info_label.setText(f"Правил после редукции: {len(active_rules)}")
        self.info_label.setStyleSheet("font-size: 13px; color: #2e7d32; font-weight: bold; padding: 5px;")
        
        self.btn_extract.setEnabled(True)
        self.btn_show.setEnabled(True)

    def _extract_rules(self):
        """Сохранение правил в файл"""
        if self._active_rules is None:
            QMessageBox.warning(self, "Предупреждение", "Сначала выполните Этап 1")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Извлечь нечеткие правила", "knowledge_base.txt",
            "Text files (*.txt);;CSV files (*.csv);;All files (*.*)"
        )
        if not file_path:
            return
        
        try:
            n_features = len(self.mw.stage1_results['gradations'])
            
            with open(file_path, 'w', encoding='utf-8') as f:
                # === СЕКЦИЯ 1: Параметры функций принадлежности ===
                f.write("# ========================================\n")
                f.write("# Параметры функций принадлежности\n")
                f.write("# Формат: Feature: имя_признака\n")
                f.write("#         Term N: a=X b=X c=X d=X\n")
                f.write("# ========================================\n\n")
                
                membership_funcs = self.mw.stage1_results['membership_funcs']
                for f_idx, feature_mfs in enumerate(membership_funcs):
                    f.write(f"Feature: {self.mw.feature_names[f_idx]}\n")
                    for t_idx, mf in enumerate(feature_mfs):
                        a, b, c, d = mf.get_params()
                        f.write(f"Term {t_idx+1}: a={a:.6f} b={b:.6f} c={c:.6f} d={d:.6f}\n")
                    f.write("\n")
                
                # === СЕКЦИЯ 2: Правила ===
                f.write("# ========================================\n")
                f.write("# База правил\n")
                f.write("# Формат: терм1 терм2 ... термN Класс CF\n")
                f.write("# ========================================\n\n")
                
                headers = list(self.mw.feature_names[:n_features]) + ["Класс", "CF"]
                f.write('\t'.join(headers) + '\n')
                
                for rule, cf in zip(self._active_rules, self._active_cfs):
                    terms = [str(t + 1) for t in rule]
                    best_class = np.argmax(cf)
                    class_name = self.mw.class_names[best_class]
                    cf_value = f"{cf[best_class]:.4f}"
                    row = terms + [class_name, cf_value]
                    f.write('\t'.join(row) + '\n')
            
            QMessageBox.information(self, "Извлечение", f"Правила сохранены в:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить:\n{str(e)}")
    
    def _show_rules(self):
        """Показать правила в отдельном окне"""
        if self._active_rules is None:
            QMessageBox.warning(self, "Предупреждение", "Сначала выполните Этап 1")
            return
        
        n_features = len(self.mw.stage1_results['gradations'])
        dialog = RulesDialog(
            self._active_rules,
            self._active_cfs,
            self.mw.feature_names[:n_features],
            self.mw.class_names,
            self
        )
        dialog.exec()