"""Панель параметров ГА3 (Обучение модели)"""
from PyQt6.QtWidgets import QGroupBox, QFormLayout, QSpinBox


class TuningParamsPanel(QGroupBox):
    def __init__(self):
        super().__init__("Параметры ГА")
        layout = QFormLayout(self)
        
        self.m_bits = QSpinBox()
        self.m_bits.setRange(2, 6)
        self.m_bits.setValue(3)
        layout.addRow("Точность дискретизации (m):", self.m_bits)
        
        self.pop_size = QSpinBox()
        self.pop_size.setRange(2, 500)
        self.pop_size.setValue(16)
        layout.addRow("Число хромосом в популяции:", self.pop_size)
        
        self.stall_epochs = QSpinBox()
        self.stall_epochs.setRange(1, 50)
        self.stall_epochs.setValue(2)
        layout.addRow("Эпох стагнации:", self.stall_epochs)
    
    def get_parameters(self):
        return {
            'm_bits': self.m_bits.value(),
            'population_size': self.pop_size.value(),
            'max_generations': 1000,       # Фиксированное значение
            'max_epochs': 1000,            # Фиксированное значение
            'stall_epochs': self.stall_epochs.value(),
            'tournament_size': 3,
            'crossover_prob': 0.8,
            'mutation_prob': 0.1,
            'stall_generations': 5,
            'verbose': True
        }