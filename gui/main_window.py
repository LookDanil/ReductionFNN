"""Главное окно программы"""
import sys
import time
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QMenu, QStatusBar,
    QMessageBox, QLabel, QFileDialog
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction
import numpy as np
import pickle

from gui.styles.themes import LIGHT_QSS, DARK_QSS
from gui.dialogs.log_dialog import LogDialog
from gui.dialogs.data_view_dialog import DataViewDialog
from gui.dialogs.mf_dialog import MFDialog
from gui.dialogs.metrics_dialog import MetricsDialog
from gui.dialogs.structure_dialog import StructureDialog
from gui.tabs.stage1_tab import Stage1Tab
from gui.tabs.stage2_tab import Stage2Tab
from gui.tabs.stage3_tab import Stage3Tab
from gui.workers.stage1_worker import Stage1Worker
from gui.workers.stage2_worker import Stage2Worker
from gui.workers.stage3_worker import Stage3Worker
from core.data_loader import load_dataset


class MainWindow(QMainWindow):
    """Главное окно программы"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Программа формирования баз знаний на основе нейронечеткой редуцированной модели")
        self.setMinimumSize(1200, 800)

        # Данные
        self.X = self.y = self.X_full = self.y_full = None
        self.feature_names = self.feature_names_full = self.class_names = None

        # Результаты этапов
        self.stage1_results = self.stage2_results = self.stage3_results = None
        self.worker = self.worker2 = self.worker3 = None
        self.log_dialog = None

        # Единая модель
        self.fnn_model = None
        self.baseline_train_accuracy = None

        # Буфер логов
        self._log_buffer = []

        # Таймеры
        self._time_timer_stage1 = QTimer()
        self._time_timer_stage1.timeout.connect(self._update_running_time_stage1)
        self._stage_start_time_stage1 = None

        self._time_timer_stage2 = QTimer()
        self._time_timer_stage2.timeout.connect(self._update_running_time_stage2)
        self._stage_start_time_stage2 = None

        self._time_timer_stage3 = QTimer()
        self._time_timer_stage3.timeout.connect(self._update_running_time_stage3)
        self._stage_start_time_stage3 = None

        # Тема
        self._dark_theme = False

        self._setup_menu()
        self._setup_tabs()
        self._setup_statusbar()
        self._apply_theme()
        self.btn_start.setEnabled(False)

    # ==========================================
    # МЕНЮ
    # ==========================================
    def _setup_menu(self):
        menubar = self.menuBar()
        model_menu = menubar.addMenu("Нейронечеткая редуцированная модель")
        model_menu.addAction("Сохранить модель", self._save_model)
        model_menu.addAction("Загрузить модель", self._load_model)
        model_menu.addSeparator()
        model_menu.addAction("Выход", self.close)
        kb_menu = menubar.addMenu("База знаний")
        kb_menu.addAction("Сохранить базу знаний")
        view_menu = menubar.addMenu("Вид")
        self.theme_action = QAction("🌙 Тёмная тема", self)
        self.theme_action.setCheckable(True)
        self.theme_action.triggered.connect(self._toggle_theme)
        view_menu.addAction(self.theme_action)
        help_menu = menubar.addMenu("Справка")
        help_menu.addAction("Показать логи", self._show_logs)
        help_menu.addSeparator()
        help_menu.addAction("О программе", self._show_about)

    def _toggle_theme(self, checked):
        self._dark_theme = checked
        self.theme_action.setText("☀ Светлая тема" if checked else "🌙 Тёмная тема")
        self._apply_theme()

    def _apply_theme(self):
        self.setStyleSheet(DARK_QSS if self._dark_theme else LIGHT_QSS)

    # ==========================================
    # ВКЛАДКИ
    # ==========================================
    def _setup_tabs(self):
        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)

        self.tab_stage1 = Stage1Tab(self)
        self.tab_widget.addTab(self.tab_stage1, "Преднастройка модели")

        self.tab_stage2 = Stage2Tab(self)
        self.tab_widget.addTab(self.tab_stage2, "Редукция модели")
        self.tab_widget.setTabEnabled(1, False)

        self.tab_stage3 = Stage3Tab(self)
        self.tab_widget.addTab(self.tab_stage3, "Обучение модели")
        self.tab_widget.setTabEnabled(2, False)

        from gui.tabs.stage4_tab import Stage4Tab
        self.tab_stage4 = Stage4Tab(self)
        self.tab_widget.addTab(self.tab_stage4, "Извлечение нечетких правил из модели")
        self.tab_widget.setTabEnabled(3, False)

    # ==========================================
    # СТАТУСБАР
    # ==========================================
    def _setup_statusbar(self):
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self.statusbar.showMessage("Готов к работе | Загрузите данные")

    # ==========================================
    # ТАЙМЕРЫ
    # ==========================================
    def _update_running_time_stage1(self):
        if self._stage_start_time_stage1 is not None:
            elapsed = time.time() - self._stage_start_time_stage1
            h, m, s = int(elapsed // 3600), int((elapsed % 3600) // 60), int(elapsed % 60)
            self.results_panel.time_label.setText(f"{h:02d}:{m:02d}:{s:02d}")

    def _update_running_time_stage2(self):
        if self._stage_start_time_stage2 is not None:
            elapsed = time.time() - self._stage_start_time_stage2
            h, m, s = int(elapsed // 3600), int((elapsed % 3600) // 60), int(elapsed % 60)
            self.reduction_results_panel.time_label.setText(f"{h:02d}:{m:02d}:{s:02d}")

    def _update_running_time_stage3(self):
        if self._stage_start_time_stage3 is not None:
            elapsed = time.time() - self._stage_start_time_stage3
            h, m, s = int(elapsed // 3600), int((elapsed % 3600) // 60), int(elapsed % 60)
            self.tuning_results_panel.time_label.setText(f"{h:02d}:{m:02d}:{s:02d}")

    # ==========================================
    # ЗАГРУЗКА ДАННЫХ
    # ==========================================
    def _on_dataset_selected(self, index):
        if index == 0:
            self.btn_start.setEnabled(False)
            self.statusbar.showMessage("Готов к работе | Загрузите данные")
            self.X_full = None
            self.dataset_panel.show_view_button(False)
            self.dataset_panel.set_features_visible(True)
            return
        
        dataset_name = self.dataset_panel.dataset_combo.currentText()
        
        if "TXT" in dataset_name:
            self.dataset_panel.set_features_visible(False)
            self._load_txt_file()
            # Сбрасываем комбобокс, чтобы можно было снова выбрать TXT
            self.dataset_panel.dataset_combo.blockSignals(True)
            self.dataset_panel.dataset_combo.setCurrentIndex(0)
            self.dataset_panel.dataset_combo.blockSignals(False)
            return
        
        self.dataset_panel.set_features_visible(True)
        try:
            if "Iris" in dataset_name:
                self.X_full, self.y_full, self.feature_names_full, self.class_names = load_dataset('iris')
                self.dataset_panel.features_spin.setMaximum(4)
                self.dataset_panel.features_spin.setValue(4)
                self.dataset_panel.show_view_button(True)
            elif "Wine" in dataset_name:
                self.X_full, self.y_full, self.feature_names_full, self.class_names = load_dataset('wine')
                self.dataset_panel.features_spin.setMaximum(13)
                if self.dataset_panel.features_spin.value() > 13:
                    self.dataset_panel.features_spin.setValue(13)
                self.dataset_panel.show_view_button(True)
            self._apply_feature_count()
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))

    def _on_features_changed(self, value):
        if self.X_full is not None:
            self._apply_feature_count()

    def _apply_feature_count(self):
        if self.X_full is None:
            return
        dataset_name = self.dataset_panel.dataset_combo.currentText()
        if "TXT" in dataset_name:
            self.X = self.X_full
            n_features = self.X_full.shape[1]
        else:
            n_features = self.dataset_panel.get_n_features()
            self.X = self.X_full[:, :n_features]
        self.y = self.y_full
        self.feature_names = self.feature_names_full[:n_features]
        total_features = self.X_full.shape[1]
        if "Iris" in dataset_name:
            self.dataset_panel.set_info(f"Iris: 150 примеров, {n_features}/{total_features} признаков, 3 класса")
        elif "Wine" in dataset_name:
            self.dataset_panel.set_info(f"Wine: 178 примеров, {n_features}/{total_features} признаков, 3 класса")
        elif "TXT" in dataset_name:
            self.dataset_panel.set_info(f"Файл: {n_features} признаков, {self.X.shape[0]} примеров")
        self.btn_start.setEnabled(True)
        self.statusbar.showMessage(f"Данные загружены: {self.X.shape[0]} примеров, {self.X.shape[1]} признаков")

    def _load_txt_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите файл данных", "",
            "Text files (*.txt *.csv *.dat);;All files (*.*)"
        )
        if not file_path:
            self.dataset_panel.dataset_combo.setCurrentIndex(0)
            return
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if len(lines) < 2:
                raise ValueError("Файл пуст или содержит только заголовок")
            
            first_line = lines[0].strip()
            has_header = any(c.isalpha() for c in first_line)
            
            sample = lines[1].strip() if has_header else lines[0].strip()
            n_cols = len(sample.split())
            
            data_rows = []
            class_labels = []
            start_row = 1 if has_header else 0
            
            for line in lines[start_row:]:
                parts = line.strip().split()
                if len(parts) != n_cols:
                    continue
                try:
                    row = [float(x) for x in parts]
                    data_rows.append(row)
                except ValueError:
                    try:
                        numerical_part = [float(x) for x in parts[:-1]]
                        class_label = parts[-1]
                        numerical_part.append(class_label)
                        data_rows.append(numerical_part)
                        if class_label not in class_labels:
                            class_labels.append(class_label)
                    except ValueError:
                        continue
            
            if not data_rows:
                raise ValueError("Не удалось разобрать данные")
            
            if class_labels:
                class_to_idx = {label: i for i, label in enumerate(class_labels)}
                self.class_names = class_labels
            else:
                class_to_idx = None
            
            X_list = []
            y_list = []
            
            for row in data_rows:
                if class_to_idx is not None:
                    X_list.append(row[:-1])
                    y_list.append(class_to_idx[row[-1]])
                else:
                    X_list.append(row[:-1])
                    y_list.append(int(row[-1]))
            
            self.X_full = np.array(X_list)
            self.y_full = np.array(y_list)
            
            # Имена признаков
            if has_header:
                headers = first_line.split()
                self.feature_names_full = headers[:-1]
            else:
                self.feature_names_full = [f"Признак {i+1}" for i in range(self.X_full.shape[1])]
            
            # Имена классов
            if not class_labels:
                unique_classes = np.unique(self.y_full)
                self.class_names = [f"Класс {c}" for c in unique_classes]
            
            self.dataset_panel.features_spin.setMaximum(self.X_full.shape[1])
            self.dataset_panel.features_spin.setValue(self.X_full.shape[1])
            self.dataset_panel.show_view_button(True)
            
            self._apply_feature_count()

            # Предупреждение, если много признаков
            if self.X_full.shape[1] > 7:
                QMessageBox.warning(
                    self, "Много признаков",
                    f"Загружено {self.X_full.shape[1]} признаков.\n"
                    f"Рекомендуется использовать не более 6-7 признаков\n"
                    f"во избежание нехватки памяти.\n\n"
                    f"Вы можете уменьшить количество признаков\n"
                    f"с помощью счётчика на панели."
                )
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка загрузки", f"Не удалось загрузить файл:\n{str(e)}")
            self.dataset_panel.dataset_combo.setCurrentIndex(0)

    def _view_data(self):
        if self.X_full is not None and self.y_full is not None:
            dialog = DataViewDialog(self.X_full, self.y_full, self.feature_names_full, self)
            dialog.exec()

    # ==========================================
    # ЭТАП 1
    # ==========================================
    def _on_start_stage1(self):
        config = self.params_panel.get_parameters()

        # ===== ПРОВЕРКА РАЗМЕРА ДАТАСЕТА =====
        n_features = self.X.shape[1] if self.X is not None else 0
        n_samples = self.X.shape[0] if self.X is not None else 0

        # Оценка памяти
        max_rules = 6 ** min(n_features, 10)  # до 10 для оценки
        est_memory_gb = (n_samples * max_rules * 8) / (1024**3)

        if n_features > 7:
            reply = QMessageBox.question(
                self, "Слишком много признаков",
                f"Загружено {n_features} признаков.\n"
                f"Рекомендуется использовать не более 6-7 признаков.\n"
                f"Ожидаемый расход памяти: ~{est_memory_gb:.1f} ГБ\n\n"
                f"Продолжить?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                return
        elif est_memory_gb > 4:
            reply = QMessageBox.question(
                self, "Большой расход памяти",
                f"Ожидаемый расход памяти: ~{est_memory_gb:.1f} ГБ\n"
                f"Это может привести к зависанию или ошибке.\n\n"
                f"Продолжить?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                return
        # ===== КОНЕЦ ПРОВЕРКИ =====

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.dataset_panel.setEnabled(False)
        self.params_panel.setEnabled(False)
        self.statusbar.showMessage("Подготовка...")

        self._stage_start_time_stage1 = time.time()
        self._time_timer_stage1.start(1000)
        self.results_panel.update_results("00:00:00", "—", "—", 0, 0)

        self._log_buffer = []
        self.visualization_panel.reset()
        self.worker = Stage1Worker(self.X, self.y, self.feature_names, self.class_names, config)
        self.worker.status.connect(lambda s: self.statusbar.showMessage(f"Статус: {s}"))
        self.worker.intermediate_result.connect(self._on_intermediate_result)
        self.worker.log.connect(self._on_log_message)
        self.worker.finished.connect(self._on_stage1_finished)
        self.worker.stopped.connect(self._on_stage1_stopped)
        self.worker.error.connect(self._on_stage1_error)
        self.worker.start()

    def _on_intermediate_result(self, data):
        if 'time' in data:
            self.results_panel.time_label.setText(data['time'])
        if 'rules_count' in data and data['rules_count'] != '—':
            self.results_panel.rules_label.setText(str(data['rules_count']))
        if 'antecedents_total' in data and data['antecedents_total'] != '—':
            self.results_panel.antecedents_label.setText(str(data['antecedents_total']))

        gen = data.get('generation', 0)
        train_val = data.get('train_accuracy', '—')
        test_val = data.get('test_accuracy', '—')

        if gen > 0:
            try:
                t = float(train_val) if train_val != '—' else 0
                te = float(test_val) if test_val != '—' else 0
                if t > 0:
                    self.visualization_panel.add_train_test(gen, t, te)
            except (ValueError, TypeError):
                pass

        if train_val != '—':
            self.results_panel.train_acc_label.setText(str(train_val))
        if test_val != '—':
            self.results_panel.test_acc_label.setText(str(test_val))

    def _on_log_message(self, message):
        self._log_buffer.append(message)
        if self.log_dialog is None:
            self.log_dialog = LogDialog(self)
        if self.log_dialog.isVisible():
            self.log_dialog.append_log(message)

    def _on_stop_stage1(self):
        if self.worker and self.worker.isRunning():
            self.statusbar.showMessage("Останавливаем...")
            self.btn_stop.setEnabled(False)
            self.worker.stop()
    def _on_stage1_error(self, error_msg):
        if "Нехватка памяти" in error_msg or "MemoryError" in error_msg or "Unable to allocate" in error_msg:
            QMessageBox.critical(
                self, "Нехватка памяти",
                "Программе не хватило оперативной памяти.\n\n"
                "Рекомендации:\n"
                "• Уменьшите количество признаков (4-6)\n"
                "• Используйте меньше примеров\n"
                "• Закройте другие программы"
            )
        else:
            QMessageBox.critical(self, "Ошибка", error_msg)
        self._on_stage1_stopped()

    def _on_stage1_finished(self, results):
        self._time_timer_stage1.stop()
        self.stage1_results = results
        self.fnn_model = results['fnn_model']
        self.baseline_train_accuracy = results['train_accuracy'] / 100

        self.results_panel.update_results(
            results['time'],
            results['rules_count'],
            results['antecedents_total'],
            results['train_accuracy'],
            results['test_accuracy']
        )
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.dataset_panel.setEnabled(True)
        self.params_panel.setEnabled(True)
        self.tab_widget.setTabEnabled(1, True)
        self.tab_widget.setTabEnabled(3, True)

        for btn in self.info_buttons:
            btn.setEnabled(True)

        self.statusbar.showMessage(f"Готово! Train: {results['train_accuracy']:.2f}% | Test: {results['test_accuracy']:.2f}%")
        QMessageBox.information(self, "Этап 1 завершён",
            f"Точность TRAIN: {results['train_accuracy']:.2f}%\n"
            f"Точность TEST: {results['test_accuracy']:.2f}%\n"
            f"Правил: {results['rules_count']}\n"
            f"Антецедентов: {results['antecedents_total']}"
        )

    def _on_stage1_stopped(self):
        self._time_timer_stage1.stop()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.dataset_panel.setEnabled(True)
        self.params_panel.setEnabled(True)
        self.statusbar.showMessage("Готов к работе | Выполнение остановлено")
        if self.stage1_results:
            for btn in self.info_buttons:
                btn.setEnabled(True)

    # ==========================================
    # ЭТАП 2
    # ==========================================
    def _on_start_stage2(self):
        if self.stage1_results is None:
            QMessageBox.warning(self, "Предупреждение", "Сначала выполните Этап 1")
            return

        config = self.reduction_params_panel.get_parameters()
        results = self.stage1_results

        self.btn_start_stage2.setEnabled(False)
        self.btn_stop_stage2.setEnabled(True)
        self.reduction_params_panel.setEnabled(False)
        self.statusbar.showMessage("Выполняется Этап 2...")

        self._stage_start_time_stage2 = time.time()
        self._time_timer_stage2.start(1000)
        self.reduction_results_panel.update_results("00:00:00", "—", "—", 0, 0)

        self.visualization_panel_stage2.reset()

        self.worker2 = Stage2Worker(
            self.fnn_model,
            results['X_train'], results['y_train'],
            results['X_test'], results['y_test'],
            self.class_names,
            config,
            baseline_accuracy=self.baseline_train_accuracy
        )
        self.worker2.status.connect(lambda s: self.statusbar.showMessage(f"Статус: {s}"))
        self.worker2.intermediate_result.connect(self._on_stage2_intermediate)
        self.worker2.log.connect(self._on_log_message)
        self.worker2.finished.connect(self._on_stage2_finished)
        self.worker2.stopped.connect(self._on_stage2_stopped)
        self.worker2.error.connect(lambda e: QMessageBox.critical(self, "Ошибка", e))
        self.worker2.start()

    def _on_stage2_intermediate(self, data):
        gen = data.get('generation', 0)
        if gen is not None and gen > 0:
            train_acc = data.get('train_accuracy', 0)
            test_acc = data.get('test_accuracy', 0)
            if train_acc and train_acc > 0:
                self.reduction_results_panel.train_acc_label.setText(f"{train_acc:.2f}")
            if test_acc and test_acc > 0:
                self.reduction_results_panel.test_acc_label.setText(f"{test_acc:.2f}")
            if train_acc and train_acc > 0:
                self.visualization_panel_stage2.add_train_test(gen, train_acc, test_acc)

    def _on_stop_stage2(self):
        if self.worker2 and self.worker2.isRunning():
            self.statusbar.showMessage("Останавливаем...")
            self.btn_stop_stage2.setEnabled(False)
            self.worker2.stop()

    def _on_stage2_finished(self, results):
        self._time_timer_stage2.stop()
        self.stage2_results = results
        self.fnn_model = results['fnn_model']

        self.tab_stage4.load_reduced_rules(results, self.stage1_results)

        rules_after_count = 0
        for cf in results['active_cfs']:
            rules_after_count += int(np.sum(cf > 0))

        antecedents_count = len(results['active_cfs'])

        self.reduction_results_panel.update_results(
            results['time'],
            rules_after_count,
            antecedents_count,
            results['final_accuracy'] * 100,
            results['test_accuracy']
        )

        self.btn_start_stage2.setEnabled(True)
        self.btn_stop_stage2.setEnabled(False)
        self.reduction_params_panel.setEnabled(True)
        self.tab_widget.setTabEnabled(2, True)

        for btn in self.info_buttons_stage2:
            btn.setEnabled(True)

        self.statusbar.showMessage(
            f"Этап 2 завершён! Правил: {rules_after_count} | "
            f"Train: {results['final_accuracy']*100:.2f}% | Test: {results['test_accuracy']:.2f}%"
        )

        QMessageBox.information(self, "Этап 2 завершён",
            f"Редукция завершена!\n\n"
            f"Антецедентов: {results['total_rules']} → {antecedents_count}\n"
            f"Правил: {rules_after_count}\n"
            f"Удалено антецедентов: {results['removed_pct']:.1f}%\n\n"
            f"Точность TRAIN: {results['final_accuracy']*100:.2f}%\n"
            f"Точность TEST: {results['test_accuracy']:.2f}%"
        )

    def _on_stage2_stopped(self):
        self._time_timer_stage2.stop()
        self.btn_start_stage2.setEnabled(True)
        self.btn_stop_stage2.setEnabled(False)
        self.reduction_params_panel.setEnabled(True)
        self.statusbar.showMessage("Готов к работе | Выполнение остановлено")

    # ==========================================
    # ЭТАП 3
    # ==========================================
    def _on_start_stage3(self):
        if self.stage2_results is None:
            QMessageBox.warning(self, "Предупреждение", "Сначала выполните Этап 2")
            return

        config = self.tuning_params_panel.get_parameters()

        self.btn_start_stage3.setEnabled(False)
        self.btn_stop_stage3.setEnabled(True)
        self.tuning_params_panel.setEnabled(False)
        self.statusbar.showMessage("Выполняется Этап 3...")

        self._stage_start_time_stage3 = time.time()
        self._time_timer_stage3.start(1000)
        self.tuning_results_panel.update_results("00:00:00", "—", "—", 0, 0)

        self.visualization_panel_stage3.reset()

        self.worker3 = Stage3Worker(
            self.fnn_model,
            self.stage1_results['X_train'], self.stage1_results['y_train'],
            self.stage1_results['X_test'], self.stage1_results['y_test'],
            config
        )
        self.worker3.status.connect(lambda s: self.statusbar.showMessage(f"Статус: {s}"))
        self.worker3.progress.connect(lambda ep, max_ep, acc: None)
        self.worker3.log.connect(self._on_log_message)
        self.worker3.finished.connect(self._on_stage3_finished)
        self.worker3.stopped.connect(self._on_stage3_stopped)
        self.worker3.error.connect(lambda e: QMessageBox.critical(self, "Ошибка", e))
        self.worker3.start()

    def _on_stop_stage3(self):
        if self.worker3 and self.worker3.isRunning():
            self.statusbar.showMessage("Останавливаем...")
            self.btn_stop_stage3.setEnabled(False)
            self.worker3.stop()

    def _on_stage3_finished(self, results):
        self._time_timer_stage3.stop()
        self.stage3_results = results
        self.fnn_model = results['fnn_model']

        # Правила после Этапа 2 (если есть) или после Этапа 1
        rules_count = 0
        for cf in self.fnn_model.active_cfs:
            rules_count += int(np.sum(cf > 0))

        antecedents_count = len(self.fnn_model.active_rules)

        self.tuning_results_panel.update_results(
            results['time'],
            rules_count,
            antecedents_count,
            results['final_train_accuracy'],
            results['final_test_accuracy']
        )
        self.btn_start_stage3.setEnabled(True)
        self.btn_stop_stage3.setEnabled(False)
        self.tuning_params_panel.setEnabled(True)
        self.tab_widget.setTabEnabled(3, True)

        for btn in self.info_buttons_stage3:
            btn.setEnabled(True)

        self.statusbar.showMessage(
            f"Этап 3 завершён! Train: {results['final_train_accuracy']:.2f}% | "
            f"Test: {results['final_test_accuracy']:.2f}%"
        )
        QMessageBox.information(self, "Этап 3 завершён",
            f"Обучение завершено!\n\n"
            f"Точность TRAIN: {results['initial_train_accuracy']:.2f}% → {results['final_train_accuracy']:.2f}%\n"
            f"Точность TEST: {results['initial_test_accuracy']:.2f}% → {results['final_test_accuracy']:.2f}%\n"
            f"Улучшение TRAIN: {results['final_train_accuracy'] - results['initial_train_accuracy']:+.2f}%\n"
            f"Улучшение TEST: {results['final_test_accuracy'] - results['initial_test_accuracy']:+.2f}%"
        )

    def _on_stage3_stopped(self):
        self._time_timer_stage3.stop()
        self.btn_start_stage3.setEnabled(True)
        self.btn_stop_stage3.setEnabled(False)
        self.tuning_params_panel.setEnabled(True)
        self.statusbar.showMessage("Готов к работе | Выполнение остановлено")

    # ==========================================
    # ИНФОРМАЦИОННЫЕ КНОПКИ
    # ==========================================
    def _show_structure(self):
        if self.stage1_results is None: return
        r = self.stage1_results
        StructureDialog(r['gradations'], r['antecedents_total'], self.class_names, self).exec()

    def _show_membership_functions(self):
        if self.stage1_results is None: return
        MFDialog(self.stage1_results['membership_funcs'], self.feature_names, self).exec()

    def _show_metrics(self):
        if self.stage1_results is None: return
        r = self.stage1_results
        MetricsDialog(r['X_train'], r['y_train'], r['X_test'], r['y_test'],
                      r['membership_funcs'], r['gradations'], self.class_names, self).exec()

    def _show_structure_stage2(self):
        if self.stage2_results is None: return
        r = self.stage2_results
        antecedents = len(r['active_cfs'])
        StructureDialog(r['gradations'], antecedents, self.class_names, self).exec()

    def _show_mf_stage2(self):
        if self.stage1_results is None: return
        MFDialog(self.stage1_results['membership_funcs'], self.feature_names, self).exec()

    def _show_metrics_stage2(self):
        if self.stage2_results is None: return
        r2 = self.stage2_results
        r1 = self.stage1_results
        MetricsDialog(r1['X_train'], r1['y_train'], r1['X_test'], r1['y_test'],
                      r1['membership_funcs'], r2['gradations'], self.class_names, self).exec()

    def _show_structure_stage3(self):
        if self.stage2_results is None: return
        r = self.stage2_results
        antecedents = len(r['active_cfs'])
        StructureDialog(r['gradations'], antecedents, self.class_names, self).exec()

    def _show_mf_stage3(self):
        if self.stage3_results is not None and 'final_mfs' in self.stage3_results:
            mfs = self.stage3_results['final_mfs']
        elif self.stage1_results is not None:
            mfs = self.stage1_results['membership_funcs']
        else:
            return
        MFDialog(mfs, self.feature_names, self).exec()

    def _show_metrics_stage3(self):
        if self.stage1_results is None: return
        r1 = self.stage1_results
        MetricsDialog(r1['X_train'], r1['y_train'], r1['X_test'], r1['y_test'],
                      r1['membership_funcs'], r1['gradations'], self.class_names, self).exec()

    # ==========================================
    # ДИАЛОГИ
    # ==========================================
    def _show_logs(self):
        if self.log_dialog is None:
            self.log_dialog = LogDialog(self)
        self.log_dialog.log_text.clear()
        for msg in self._log_buffer:
            self.log_dialog.log_text.append(msg)
        self.log_dialog.show()
        self.log_dialog.raise_()

    # ==========================================
    # СОХРАНЕНИЕ / ЗАГРУЗКА МОДЕЛИ
    # ==========================================
    def _save_model(self):
        if self.fnn_model is None:
            QMessageBox.warning(self, "Предупреждение", "Нет обученной модели.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить модель", "model.pkl",
            "Pickle files (*.pkl);;All files (*.*)"
        )
        if not file_path:
            return

        try:
            if self.stage3_results is not None:
                stage = 3
            elif self.stage2_results is not None:
                stage = 2
            elif self.stage1_results is not None:
                stage = 1
            else:
                stage = 1

            data = {
                'fnn_model': self.fnn_model,
                'stage': stage,
                'class_names': self.class_names,
                'feature_names': self.feature_names,
            }

            if self.stage1_results:
                data['X_train'] = self.stage1_results.get('X_train')
                data['y_train'] = self.stage1_results.get('y_train')
                data['X_test'] = self.stage1_results.get('X_test')
                data['y_test'] = self.stage1_results.get('y_test')
                data['gradations'] = self.stage1_results.get('gradations')
                data['rules_count'] = self.stage1_results.get('rules_count')
                data['active_antecedents'] = self.stage1_results.get('active_antecedents')
                data['train_accuracy'] = self.stage1_results.get('train_accuracy')
                data['test_accuracy'] = self.stage1_results.get('test_accuracy')

            if self.stage2_results:
                data['rules_after'] = self.stage2_results.get('rules_after')
                data['removed_pct'] = self.stage2_results.get('removed_pct')
                data['final_accuracy_stage2'] = self.stage2_results.get('final_accuracy')
                data['test_accuracy_stage2'] = self.stage2_results.get('test_accuracy')

            if self.stage3_results:
                data['final_train_accuracy'] = self.stage3_results.get('final_train_accuracy')
                data['final_test_accuracy'] = self.stage3_results.get('final_test_accuracy')
                data['initial_train_accuracy'] = self.stage3_results.get('initial_train_accuracy')
                data['initial_test_accuracy'] = self.stage3_results.get('initial_test_accuracy')

            with open(file_path, 'wb') as f:
                pickle.dump(data, f)

            stage_names = {1: "Этап 1 (Преднастройка)", 2: "Этап 2 (Редукция)", 3: "Этап 3 (Обучение)"}
            QMessageBox.information(self, "Сохранено",
                f"Модель сохранена ({stage_names[stage]})\nФайл: {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить:\n{str(e)}")

    def _load_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Загрузить модель", "",
            "Pickle files (*.pkl);;All files (*.*)"
        )
        if not file_path:
            return

        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)

            self.fnn_model = data['fnn_model']
            self.class_names = data['class_names']
            self.feature_names = data['feature_names']
            stage = data.get('stage', 1)

            self.stage1_results = {
                'fnn_model': self.fnn_model,
                'X_train': data['X_train'],
                'y_train': data['y_train'],
                'X_test': data['X_test'],
                'y_test': data['y_test'],
                'X_all': data['X_train'],
                'y_all': data['y_train'],
                'gradations': data['gradations'],
                'membership_funcs': self.fnn_model.membership_funcs,
                'rules_count': data['rules_count'],
                'antecedents_total': data.get('antecedents_total', 0),
                'train_accuracy': data['train_accuracy'],
                'test_accuracy': data['test_accuracy'],
                'fitness': data['train_accuracy'] / 100 if data['train_accuracy'] else 0,
            }

            self.baseline_train_accuracy = data['train_accuracy'] / 100 if data['train_accuracy'] else 0

            if stage >= 2 and 'rules_after' in data:
                self.stage2_results = {
                    'fnn_model': self.fnn_model,
                    'rules_after': data['rules_after'],
                    'removed_pct': data['removed_pct'],
                    'final_accuracy': data['final_accuracy_stage2'],
                    'test_accuracy': data['test_accuracy_stage2'],
                    'gradations': data['gradations'],
                    'membership_funcs': self.fnn_model.membership_funcs,
                    'active_rules': self.fnn_model.active_rules,
                    'active_cfs': self.fnn_model.active_cfs,
                    'total_rules': data['rules_count'],
                }
                antecedents = len(self.fnn_model.active_rules)
                self.reduction_results_panel.update_results(
                    "—", data['rules_after'], antecedents,
                    data['final_accuracy_stage2'] * 100 if data['final_accuracy_stage2'] else 0,
                    data['test_accuracy_stage2']
                )
                for btn in self.info_buttons_stage2:
                    btn.setEnabled(True)

            if stage >= 3 and 'final_train_accuracy' in data:
                self.stage3_results = {
                    'fnn_model': self.fnn_model,
                    'final_train_accuracy': data['final_train_accuracy'],
                    'final_test_accuracy': data['final_test_accuracy'],
                    'initial_train_accuracy': data['initial_train_accuracy'],
                    'initial_test_accuracy': data['initial_test_accuracy'],
                }
                rules_count = 0
                for cf in self.fnn_model.active_cfs:
                    rules_count += int(np.sum(cf > 0))
                antecedents = len(self.fnn_model.active_rules)
                self.tuning_results_panel.update_results(
                    "—", rules_count, antecedents,
                    data['final_train_accuracy'], data['final_test_accuracy']
                )
                for btn in self.info_buttons_stage3:
                    btn.setEnabled(True)

            antecedents_stage1 = data.get('active_antecedents', 0)
            self.results_panel.update_results(
                "—", data['rules_count'], antecedents_stage1,
                data['train_accuracy'], data['test_accuracy']
            )

            self.tab_widget.setTabEnabled(1, True)
            if stage >= 2:
                self.tab_widget.setTabEnabled(2, True)
            if stage >= 3:
                self.tab_widget.setTabEnabled(3, True)

            for btn in self.info_buttons:
                btn.setEnabled(True)

            stage_names = {1: "Этап 1 (Преднастройка)", 2: "Этап 2 (Редукция)", 3: "Этап 3 (Обучение)"}
            QMessageBox.information(self, "Загружено",
                f"Модель загружена ({stage_names[stage]})\n"
                f"Градации: {data['gradations']}\n"
                f"Правил: {data['rules_count']}\n"
                f"Антецедентов: {antecedents_stage1}\n"
                f"Точность TRAIN: {data['train_accuracy']:.2f}%")

            self.statusbar.showMessage(f"Модель загружена ({stage_names[stage]})")

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить:\n{str(e)}")

    def _show_about(self):
        QMessageBox.about(self, "О программе", "Нейронечеткая редуцированная модель\nВерсия 1.0")


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()