"""Панель результатов ГА3 (Обучение модели)"""
from PyQt6.QtWidgets import QGroupBox, QGridLayout, QLabel


class TuningResultsPanel(QGroupBox):
    def __init__(self):
        super().__init__("Результаты работы ГА")
        layout = QGridLayout(self)
        
        layout.addWidget(QLabel("Время работы (чч:мм:сс):"), 0, 0)
        self.time_label = QLabel("00:00:00")
        layout.addWidget(self.time_label, 0, 1)
        
        layout.addWidget(QLabel("Число правил в модели:"), 1, 0)
        self.rules_label = QLabel("—")
        layout.addWidget(self.rules_label, 1, 1)
        
        layout.addWidget(QLabel("Число нейронов-антецедентов:"), 2, 0)
        self.antecedents_label = QLabel("—")
        layout.addWidget(self.antecedents_label, 2, 1)
        
        layout.addWidget(QLabel("Точность модели, % (train):"), 3, 0)
        self.train_acc_label = QLabel("—")
        layout.addWidget(self.train_acc_label, 3, 1)
        
        layout.addWidget(QLabel("Точность модели, % (test):"), 4, 0)
        self.test_acc_label = QLabel("—")
        layout.addWidget(self.test_acc_label, 4, 1)
    
    def update_results(self, time_str, rules, antecedents, train_acc, test_acc):
        self.time_label.setText(time_str)
        self.rules_label.setText(str(rules))
        self.antecedents_label.setText(str(antecedents))
        self.train_acc_label.setText(f"{train_acc:.2f}")
        self.test_acc_label.setText(f"{test_acc:.2f}")