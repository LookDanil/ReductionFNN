"""Рабочий поток для Этапа 3: Обучение модели"""
import time
import traceback
from PyQt6.QtCore import QThread, pyqtSignal
import numpy as np
from sklearn.metrics import accuracy_score

from core.ga_stage3 import ParallelGA3CyclicTuner


class Stage3Worker(QThread):
    progress = pyqtSignal(int, int, float)
    log = pyqtSignal(str)
    status = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    stopped = pyqtSignal()
    
    def __init__(self, fnn_model, X_train, y_train, X_test, y_test, config):
        super().__init__()
        self.fnn_model = fnn_model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.config = config
        self._is_running = True
        self._start_time = time.time()
    
    def run(self):
        self._start_time = time.time()
        try:
            self.status.emit("Инициализация ГА3...")
            self.log.emit("Инициализация ГА3 (Тонкая настройка ФП)...")
            
            # Используем predict + accuracy_score
            y_pred_train = self.fnn_model.predict(self.X_train)
            initial_train_acc = accuracy_score(self.y_train, y_pred_train)
            
            y_pred_test = self.fnn_model.predict(self.X_test)
            initial_test_acc = accuracy_score(self.y_test, y_pred_test)
            
            self.log.emit(f"Начальная точность TRAIN: {initial_train_acc*100:.2f}%")
            self.log.emit(f"Начальная точность TEST: {initial_test_acc*100:.2f}%")
            
            def on_progress(epoch, max_epochs, accuracy):
                if self._is_running:
                    self.progress.emit(epoch, max_epochs, accuracy)
            
            def on_log(msg):
                self.log.emit(msg)
            
            tuner = ParallelGA3CyclicTuner(
                fnn=self.fnn_model,
                X_all=self.X_train,
                y_all=self.y_train,
                config=self.config,
                progress_callback=on_progress,
                log_callback=on_log
            )
            
            self.status.emit("Обучение модели...")
            self.log.emit("Запуск тонкой настройки ФП...")
            
            final_mfs, final_train_acc, history = tuner.run()
            
            if not self._is_running:
                self.log.emit("Остановлено пользователем")
                self.stopped.emit()
                return
            
            # Финальная оценка через predict
            y_pred_train = self.fnn_model.predict(self.X_train)
            final_train_acc = accuracy_score(self.y_train, y_pred_train)
            
            y_pred_test = self.fnn_model.predict(self.X_test)
            final_test_acc = accuracy_score(self.y_test, y_pred_test)
            
            elapsed = time.time() - self._start_time
            h, m, s = int(elapsed // 3600), int((elapsed % 3600) // 60), int(elapsed % 60)
            
            results = {
                'time': f"{h:02d}:{m:02d}:{s:02d}",
                'epochs': len(history.get('epoch', [])),
                'initial_train_accuracy': initial_train_acc * 100,
                'final_train_accuracy': final_train_acc * 100,
                'initial_test_accuracy': initial_test_acc * 100,
                'final_test_accuracy': final_test_acc * 100,
                'final_mfs': final_mfs,
                'history': history,
                'fnn_model': self.fnn_model,
            }
            
            self.log.emit(f"Этап 3 завершён за {results['time']}")
            self.log.emit(f"TRAIN: {initial_train_acc*100:.2f}% → {final_train_acc*100:.2f}%")
            self.log.emit(f"TEST: {initial_test_acc*100:.2f}% → {final_test_acc*100:.2f}%")
            self.status.emit("Готово")
            self.finished.emit(results)
            
        except Exception as e:
            self.error.emit(f"{traceback.format_exc()}")
    
    def stop(self):
        self._is_running = False