"""Рабочий поток для Этапа 2: Редукция модели"""
import time
import traceback
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import QApplication
import numpy as np

from core.ga_stage2 import GeneticAntecedentReducer
from sklearn.metrics import accuracy_score


class Stage2Worker(QThread):
    progress = pyqtSignal(int, int, float)
    intermediate_result = pyqtSignal(dict)
    status = pyqtSignal(str)
    log = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    stopped = pyqtSignal()
    
    def __init__(self, fnn_model, X_train, y_train, X_test, y_test, class_names, config, 
                 baseline_accuracy=None):
        super().__init__()
        self.fnn_model = fnn_model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.class_names = class_names
        self.config = config
        self._is_running = True
        self._start_time = time.time()
        self._optimizer = None
        self._last_accuracy = -1
        self._last_test_acc = 0.0
        self.baseline_accuracy = baseline_accuracy
        self._last_generation = 0
    
    def run(self):
        self._start_time = time.time()
        try:
            if not self._is_running:
                self.stopped.emit()
                return
            
            gradations = self.fnn_model.gradations
            membership_funcs = self.fnn_model.membership_funcs
            all_rules = self.fnn_model.active_rules
            
            rule_cfs_array = self.fnn_model.rule_cfs_array
            all_cfs = [rule_cfs_array[i] for i in range(len(all_rules))]
            
            self.status.emit("Фильтрация правил...")
            self.log.emit("Отсеивание неактивных правил (CF=0)...")
            
            active_mask = np.max(rule_cfs_array, axis=1) > 0
            active_rules = [all_rules[i] for i in range(len(all_rules)) if active_mask[i]]
            active_cfs = [all_cfs[i] for i in range(len(all_rules)) if active_mask[i]]
            
            self.log.emit(f"Всего правил: {len(all_rules)}")
            self.log.emit(f"Активных правил: {len(active_rules)}")
            self.log.emit(f"Удалено неактивных: {len(all_rules) - len(active_rules)}")
            
            if not self._is_running:
                self.stopped.emit()
                return
            
            if self.baseline_accuracy is None:
                y_pred_base = self.fnn_model.predict(self.X_train)
                baseline_accuracy = accuracy_score(self.y_train, y_pred_base)
            else:
                baseline_accuracy = self.baseline_accuracy

            self.log.emit(f"Baseline accuracy (TRAIN): {baseline_accuracy:.4f}")
            
            def on_progress(generation, max_generations, accuracy):
                if self._is_running:
                    self._last_generation = generation
                    self.progress.emit(generation, max_generations, accuracy)
                    elapsed = time.time() - self._start_time
                    h, m, s = int(elapsed // 3600), int((elapsed % 3600) // 60), int(elapsed % 60)
                    
                    # Используем лучшую FNN из ГА2 для TEST
                    test_acc = self._last_test_acc
                    if abs(accuracy - self._last_accuracy) > 1e-6:
                        self._last_accuracy = accuracy
                        if self._optimizer and self._optimizer.get_best_fnn():
                            best_fnn = self._optimizer.get_best_fnn()
                            test_acc = accuracy_score(self.y_test, best_fnn.predict(self.X_test)) * 100
                            self._last_test_acc = test_acc
                    
                    self.intermediate_result.emit({
                        'time': f"{h:02d}:{m:02d}:{s:02d}",
                        'generation': generation,
                        'train_accuracy': accuracy * 100,
                        'test_accuracy': test_acc,
                    })
                    QApplication.processEvents()
            
            def on_log(msg):
                self.log.emit(msg)
            
            self.status.emit("Создание оптимизатора...")
            self.log.emit("Инициализация ГА2...")
            
            if not self._is_running:
                self.stopped.emit()
                return
            
            self._optimizer = GeneticAntecedentReducer(
                self.X_train, self.y_train,
                gradations, membership_funcs,
                self.class_names, baseline_accuracy, self.config,
                progress_callback=on_progress,
                log_callback=on_log,
                active_rules=active_rules,
                active_cfs=active_cfs
            )
            
            if not self._is_running:
                self.stopped.emit()
                return
            
            self._optimizer.precompute()
            self._optimizer.accuracy_cache.clear()
            
            if not self._is_running:
                self.stopped.emit()
                return
            
            self.log.emit("Предрасчёт завершён")
            
            self.status.emit("Редукция правил...")
            self.log.emit("Запуск редукции правил...")
            
            best_chrom, passive_count, final_accuracy, history, reduced_fnn = self._optimizer.run()
            
            if not self._is_running:
                self.log.emit("Остановлено пользователем")
                self.stopped.emit()
                return
            
            self.fnn_model = reduced_fnn
            
            elapsed = time.time() - self._start_time
            h, m, s = int(elapsed // 3600), int((elapsed % 3600) // 60), int(elapsed % 60)
            
            rules_after = len(self.fnn_model.active_rules)
            removed_pct = (len(active_rules) - rules_after) / len(active_rules) * 100
            
            train_acc = final_accuracy * 100
            test_acc = accuracy_score(self.y_test, self.fnn_model.predict(self.X_test)) * 100
            
            results = {
                'best_chrom': best_chrom,
                'passive_count': passive_count,
                'final_accuracy': final_accuracy,
                'test_accuracy': test_acc,
                'active_rules': self.fnn_model.active_rules,
                'active_cfs': self.fnn_model.active_cfs,
                'time': f"{h:02d}:{m:02d}:{s:02d}",
                'total_rules': len(active_rules),
                'rules_after': rules_after,
                'removed_pct': removed_pct,
                'gradations': gradations,
                'membership_funcs': membership_funcs,
                'baseline_accuracy': baseline_accuracy,
                'fnn_model': self.fnn_model,
            }
            
            self.log.emit(f"Этап 2 завершён за {results['time']}")
            self.log.emit(f"Правил: {len(active_rules)} → {rules_after} (удалено {removed_pct:.1f}%)")
            self.log.emit(f"Точность TRAIN: {train_acc:.2f}%")
            self.log.emit(f"Точность TEST: {test_acc:.2f}%")
            self.status.emit("Готово")
            
            self.intermediate_result.emit({
                'time': results['time'],
                'generation': self._last_generation,
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
            })
            
            self.finished.emit(results)
            
        except Exception as e:
            self.error.emit(f"{traceback.format_exc()}")
    
    def stop(self):
        self._is_running = False