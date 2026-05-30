"""Рабочий поток для Этапа 1"""
import time
import traceback
from PyQt6.QtCore import QThread, pyqtSignal
import numpy as np
import itertools

from core.data_loader import split_data_by_bootstrap
from core.ga_stage1 import GeneticGradationOptimizer
from sklearn.metrics import accuracy_score


class Stage1Worker(QThread):
    progress = pyqtSignal(int, int, float)
    intermediate_result = pyqtSignal(dict)
    status = pyqtSignal(str)
    log = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    stopped = pyqtSignal()
    
    def __init__(self, X, y, feature_names, class_names, config):
        super().__init__()
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.class_names = class_names
        self.config = config
        self._is_running = True
        self._start_time = time.time()
        self._optimizer = None
        self._current_gradations = None
        self._last_fitness = -1
        self._current_train_acc = 0
        self._current_test_acc = 0
        self._metrics_computed = False
        self._last_generation = 0
    
    def run(self):
        self._start_time = time.time()
        try:
            if not self._is_running:
                self.stopped.emit()
                return
            
            self.status.emit("Разбиение данных...")
            self.log.emit("Разбиение данных методом бутстрэпа...")
            X_train, X_test, y_train, y_test, X_all, y_all = split_data_by_bootstrap(
                self.X, self.y, random_state=42
            )
            self.log.emit(f"ALL (исходный): {len(y_all)} примеров")
            self.log.emit(f"TRAIN (бутстрэп): {len(y_train)} примеров")
            self.log.emit(f"TEST (не попавшие): {len(y_test)} примеров")
            
            if not self._is_running:
                self.log.emit("Остановлено пользователем")
                self.stopped.emit()
                return
            
            self.status.emit("Инициализация ГА1...")
            self.log.emit("Инициализация ГА1 на TRAIN...")
            
            def on_progress(generation, max_generations, fitness):
                if self._is_running:
                    self._last_generation = generation
                    self.progress.emit(generation, max_generations, fitness)
                    elapsed = time.time() - self._start_time
                    h, m, s = int(elapsed // 3600), int((elapsed % 3600) // 60), int(elapsed % 60)
                    
                    antecedents_total = str(int(np.prod(self._current_gradations))) if self._current_gradations else '—'
                    
                    # Всегда получаем актуальные метрики из лучшей FNN
                    if self._optimizer:
                        try:
                            best_fnn = self._optimizer.get_best_fnn()
                            if best_fnn is not None:
                                train_acc = accuracy_score(y_train, best_fnn.predict(X_train)) * 100
                                test_acc = accuracy_score(y_test, best_fnn.predict(X_test)) * 100
                                self._current_train_acc = train_acc
                                self._current_test_acc = test_acc
                                self._metrics_computed = True
                        except Exception:
                            pass
                    
                    self._last_fitness = fitness
                    
                    show_train = f"{self._current_train_acc:.2f}" if self._metrics_computed else "—"
                    show_test = f"{self._current_test_acc:.2f}" if self._metrics_computed else "—"
                    
                    self.intermediate_result.emit({
                        'time': f"{h:02d}:{m:02d}:{s:02d}",
                        'generation': generation,
                        'fitness': fitness * 100,
                        'rules_count': '—',
                        'antecedents_total': antecedents_total,
                        'train_accuracy': show_train,
                        'test_accuracy': show_test,
                    })
            
            def on_log(msg):
                self.log.emit(msg)
                if "градации=" in msg:
                    try:
                        start = msg.index("[") + 1
                        end = msg.index("]")
                        grad_str = msg[start:end]
                        self._current_gradations = [int(x.strip()) for x in grad_str.split(',')]
                    except Exception:
                        pass
            
            if not self._is_running:
                self.stopped.emit()
                return
            
            self._optimizer = GeneticGradationOptimizer(
                X_train, y_train,
                self.class_names,
                self.config,
                progress_callback=on_progress,
                log_callback=on_log
            )
            
            if not self._is_running:
                self._optimizer.request_stop()
                self.stopped.emit()
                return
            
            self.status.emit("Оптимизация градаций...")
            self.log.emit("Запуск оптимизации градаций на TRAIN...")
            
            gradations, fitness_train, _, membership_funcs, fnn_model = self._optimizer.run()
            
            if not self._is_running or self._optimizer._stop_requested:
                self.log.emit("Остановлено пользователем")
                self.stopped.emit()
                return
            
            self.status.emit("Финальная оценка...")
            self.log.emit("Вычисление точности на train/test...")
            
            train_acc = accuracy_score(y_train, fnn_model.predict(X_train)) * 100
            test_acc = accuracy_score(y_test, fnn_model.predict(X_test)) * 100
            
            # Статистика
            rule_cfs = fnn_model.rule_cfs_array
            total_antecedents = int(np.prod(gradations))
            active_antecedents = int(np.sum(np.max(rule_cfs, axis=1) > 0))
            inactive_antecedents = total_antecedents - active_antecedents
            
            rules_cf_positive = 0
            rules_cf_zero = 0
            for cf in rule_cfs:
                for c in cf:
                    if c > 0:
                        rules_cf_positive += 1
                    else:
                        rules_cf_zero += 1
            total_rules_count = rules_cf_positive + rules_cf_zero
            
            self.log.emit(f"Точность TRAIN: {train_acc:.2f}%")
            self.log.emit(f"Точность TEST: {test_acc:.2f}%")
            self.log.emit(f"--- Статистика модели ---")
            self.log.emit(f"Нейронов-антецедентов всего: {total_antecedents}")
            self.log.emit(f"Нейронов-антецедентов активных (CF>0): {active_antecedents}")
            self.log.emit(f"Нейронов-антецедентов неактивных (CF=0): {inactive_antecedents}")
            self.log.emit(f"Всего правил (антецеденты × классы): {total_rules_count}")
            self.log.emit(f"Правил с CF>0: {rules_cf_positive}")
            self.log.emit(f"Правил с CF=0: {rules_cf_zero}")
            
            elapsed = time.time() - self._start_time
            h, m, s = int(elapsed // 3600), int((elapsed % 3600) // 60), int(elapsed % 60)
            
            results = {
                'gradations': gradations,
                'fitness': fitness_train,
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'time': f"{h:02d}:{m:02d}:{s:02d}",
                'rules_count': rules_cf_positive,
                'antecedents_total': total_antecedents,
                'membership_funcs': membership_funcs,
                'X_train': X_train, 'y_train': y_train,
                'X_test': X_test, 'y_test': y_test,
                'X_all': X_all, 'y_all': y_all,
                'fnn_model': fnn_model,
            }
            
            self.intermediate_result.emit({
                'time': results['time'],
                'generation': self._last_generation,
                'fitness': fitness_train * 100,
                'rules_count': str(rules_cf_positive),
                'antecedents_total': str(total_antecedents),
                'train_accuracy': f"{train_acc:.2f}",
                'test_accuracy': f"{test_acc:.2f}",
            })
            
            self.log.emit(f"Этап 1 завершён за {results['time']}")
            self.status.emit("Готово")
            self.finished.emit(results)
        
        except MemoryError:
            self.error.emit("Нехватка памяти! Слишком большой датасет для обработки.\n"
                            "Уменьшите количество признаков (рекомендуется 4-6).")
        except Exception as e:
            self.error.emit(f"{traceback.format_exc()}")
    
    def stop(self):
        self._is_running = False
        if self._optimizer is not None:
            self._optimizer.request_stop()