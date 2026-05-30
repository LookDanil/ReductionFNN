"""Панель параметров ГА2 (Редукция модели)"""
from PyQt6.QtWidgets import QGroupBox, QFormLayout, QSpinBox


class ReductionParamsPanel(QGroupBox):
    def __init__(self):
        super().__init__("Параметры ГА")
        layout = QFormLayout(self)
        
        self.pop_size = QSpinBox()
        self.pop_size.setRange(2, 1000)
        self.pop_size.setValue(20)
        layout.addRow("Число хромосом в популяции:", self.pop_size)
        
        self.tournament_size = QSpinBox()
        self.tournament_size.setRange(2, 100)
        self.tournament_size.setValue(2)
        layout.addRow("Число хромосом в турнирном отборе:", self.tournament_size)
        
        self.crossover_points = QSpinBox()
        self.crossover_points.setRange(1, 10)
        self.crossover_points.setValue(2)
        layout.addRow("Число этапов кроссинговера:", self.crossover_points)
        
        self.stall_generations = QSpinBox()
        self.stall_generations.setRange(1, 10000)
        self.stall_generations.setValue(10)
        layout.addRow("Число эпох холостой работы ГА:", self.stall_generations)
    
    def get_parameters(self):
        return {
            'population_size': self.pop_size.value(),
            'tournament_size': self.tournament_size.value(),
            'crossover_points': self.crossover_points.value(),
            'stall_generations': self.stall_generations.value(),
            'max_generations': 10000,  # Большое число, чтобы не мешало
            'verbose': True
        }