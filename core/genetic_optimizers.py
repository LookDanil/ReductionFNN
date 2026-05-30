"""
Модуль генетических оптимизаторов (ГА1, ГА2, ГА3)
Все ГА работают на полной выборке
Train/Test разделение только для финальной оценки
БЫЛ РЕФАКТОРЕН В ДРУГИЕ ФАЙЛ
"""

import numpy as np
import random
import time
import itertools
from typing import List, Tuple, Dict, Optional, Callable
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

from sklearn.cluster import KMeans

from core.fuzzy_system import TrapezoidalMF
from core.fnn_model import OptimizedReducedFuzzyNeuralNetwork

# Параллельные вычисления
try:
    from joblib import Parallel, delayed
    import multiprocessing
    JOBLIB_AVAILABLE = True
    N_CORES = multiprocessing.cpu_count()
except ImportError:
    JOBLIB_AVAILABLE = False
    N_CORES = 1

N_PROCESSES = min(N_CORES, 8)
USE_PROCESSES = True


# ============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ ПАРАЛЛЕЛЬНЫХ ПРОЦЕССОВ
# ============================================================================

def _serialize_mfs(mfs: List[List]) -> List[List[Tuple]]:
    """Сериализует MF для передачи в процессы"""
    data = []
    for f_mfs in mfs:
        f_data = []
        for mf in f_mfs:
            f_data.append(mf.get_params())
        data.append(f_data)
    return data


def _deserialize_mfs(data: List[List[Tuple]]) -> List[List]:
    """Десериализует MF из данных"""
    mfs = []
    for f_data in data:
        f_mfs = []
        for params in f_data:
            f_mfs.append(TrapezoidalMF(*params))
        mfs.append(f_mfs)
    return mfs


def _evaluate_single_value_process(args):
    """Глобальная функция для оценки одного значения параметра в отдельном процессе"""
    (value, feature_idx, grad_idx, param_name,
     mfs_data, X_all, y_all,
     active_rules, active_cfs, n_classes) = args

    mfs = _deserialize_mfs(mfs_data)

    a, b, c, d = mfs[feature_idx][grad_idx].get_params()
    if param_name == 'a': a = value
    elif param_name == 'b': b = value
    elif param_name == 'c': c = value
    elif param_name == 'd': d = value
    mfs[feature_idx][grad_idx] = TrapezoidalMF(a, b, c, d)

    gradations = [len(f_mfs) for f_mfs in mfs]
    temp_fnn = OptimizedReducedFuzzyNeuralNetwork(
        n_features=len(mfs),
        n_classes=n_classes,
        gradations=gradations,
        membership_funcs=mfs,
        active_rules=active_rules,
        active_cfs=active_cfs
    )

    return temp_fnn.evaluate(X_all, y_all)


# ============================================================================
# ЭТАП 1: ОПТИМИЗАЦИЯ ГРАДАЦИЙ (ГА1) - НА ВСЕЙ ВЫБОРКЕ
# ============================================================================

class GradationChromosome:
    GRADATION_VALUES = [3, 4, 5, 6]
    BITS_PER_GRADATION = 2

    @staticmethod
    def encode_gradation(k: int) -> tuple:
        index = GradationChromosome.GRADATION_VALUES.index(k)
        return tuple((index >> bit) & 1 for bit in range(1, -1, -1))

    @staticmethod
    def decode_gradation(cgi: tuple) -> int:
        index = (cgi[0] << 1) | cgi[1]
        return GradationChromosome.GRADATION_VALUES[index]

    @staticmethod
    def create_g_chromosome(gradations: List[int]) -> tuple:
        chrom_parts = []
        for k in gradations:
            chrom_parts.extend(GradationChromosome.encode_gradation(k))
        return tuple(chrom_parts)

    @staticmethod
    def decode_g_chromosome(g_chromosome: tuple, n_features: int) -> List[int]:
        gradations = []
        for i in range(n_features):
            start = i * GradationChromosome.BITS_PER_GRADATION
            cgi = g_chromosome[start:start + GradationChromosome.BITS_PER_GRADATION]
            gradations.append(GradationChromosome.decode_gradation(cgi))
        return gradations


class GeneticGradationOptimizer:
    """ЭТАП 1: Оптимизация градаций на ВСЕЙ выборке"""

    def __init__(self, X_all, y_all, class_names, config,
                 progress_callback=None, log_callback=None):
        self.X_all = X_all
        self.y_all = y_all
        self.Nall = len(y_all)
        self.class_names = class_names
        self.n_features = X_all.shape[1]
        self.n_classes = len(class_names)

        self.population_size = config.get('population_size', 20)
        self.tournament_size = config.get('tournament_size', 2)
        self.crossover_points = config.get('crossover_points', 2)
        self.stall_generations = config.get('stall_generations', 10)
        self.max_generations = config.get('max_generations', 10000)
        self.verbose = config.get('verbose', True)

        self.progress_callback = progress_callback
        self.log_callback = log_callback
        self._stop_requested = False

        self.chromosome_length = self.n_features * GradationChromosome.BITS_PER_GRADATION
        self.mutation_prob = 1.0 / self.chromosome_length if self.chromosome_length > 0 else 0.1

        self.fitness_cache = {}
        self.mf_cache = {}

        self.class_data = []
        self.Nk = np.zeros(self.n_classes)
        for k in range(self.n_classes):
            indices = np.where(self.y_all == k)[0]
            self.class_data.append(self.X_all[indices])
            self.Nk[k] = len(indices)

    def _log(self, msg):
        if self.log_callback:
            self.log_callback(msg)
        elif self.verbose:
            print(msg)
    def get_mfs_for_gradations(self, gradations):
        """Получить MF для заданных градаций (из кэша или построить)"""
        return self._build_membership_functions(gradations)

    def get_mfs_for_gradations(self, gradations):
        """Получить MF для заданных градаций (из кэша или построить)"""
        return self._build_membership_functions(gradations)
    
    def request_stop(self):
        """Запросить остановку оптимизатора"""
        self._stop_requested = True

    def _build_membership_functions(self, gradations: List[int]):
        cache_key = tuple(gradations)
        if cache_key in self.mf_cache:
            return self.mf_cache[cache_key]

        membership_funcs = []
        for feature_idx in range(self.n_features):
            k = gradations[feature_idx]
            feature_values = self.X_all[:, feature_idx]

            X_reshaped = feature_values.reshape(-1, 1)
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_reshaped)
            centers = kmeans.cluster_centers_.flatten()

            cluster_bounds = []
            for i in range(k):
                cluster_data = feature_values[labels == i]
                if len(cluster_data) > 0:
                    cluster_bounds.append((np.min(cluster_data), np.max(cluster_data)))
                else:
                    cluster_bounds.append((centers[i] - 0.1, centers[i] + 0.1))

            sorted_indices = np.argsort(centers)
            cluster_bounds = [cluster_bounds[i] for i in sorted_indices]

            X_min, X_max = np.min(feature_values), np.max(feature_values)
            data_range = X_max - X_min

            feature_mfs = []
            for grad_idx in range(k):
                b, c = cluster_bounds[grad_idx]

                if k == 1:
                    a, d = X_min, X_max
                elif grad_idx == 0:
                    a = X_min
                    d = cluster_bounds[1][0] if k > 1 else X_max
                elif grad_idx == k - 1:
                    a = cluster_bounds[k-2][1] if k > 1 else X_min
                    d = X_max
                else:
                    a = cluster_bounds[grad_idx-1][1]
                    d = cluster_bounds[grad_idx+1][0]

                if a > b:
                    a = max(X_min, b - 0.05 * data_range)
                if d < c:
                    d = min(X_max, c + 0.05 * data_range)

                feature_mfs.append(TrapezoidalMF(a, b, c, d))

            membership_funcs.append(feature_mfs)

        self.mf_cache[cache_key] = membership_funcs
        return membership_funcs

    def _calculate_activation(self, x: np.ndarray, rule: tuple, mfs):
        min_act = 1.0
        for feat_idx, grad_idx in enumerate(rule):
            act = mfs[feat_idx][grad_idx](x[feat_idx])
            if act < min_act:
                min_act = act
                if min_act == 0:
                    break
        return min_act

    def _has_activation(self, class_data, rule, mfs):
        for x in class_data:
            if self._calculate_activation(x, rule, mfs) > 0:
                return True
        return False

    def _calculate_cf(self, class_data, rule, mfs, Nk):
        if Nk == 0:
            return 0.0
        sum_act = sum(self._calculate_activation(x, rule, mfs) for x in class_data)
        return sum_act / Nk

    def _evaluate_fitness(self, gradations, mfs):
        """Оценка фитнеса на ВСЕЙ выборке (MIN-композиция, без нормализации)"""
        ranges = [range(g) for g in gradations]
        
        active_rules = []
        active_cfs = []
        
        for rule in itertools.product(*ranges):
            rule_cfs = np.zeros(self.n_classes)
            has_valid = False
            
            for k in range(self.n_classes):
                if self._has_activation(self.class_data[k], rule, mfs):
                    cf = self._calculate_cf(self.class_data[k], rule, mfs, self.Nk[k])
                    if cf > 0:
                        rule_cfs[k] = cf
                        has_valid = True
            
            if has_valid:
                active_rules.append(rule)
                active_cfs.append(rule_cfs)
        
        if not active_rules:
            return 0.0
        
        correct = 0
        for sample_idx in range(self.Nall):
            x = self.X_all[sample_idx]
            true_class = self.y_all[sample_idx]
            
            confidences = np.zeros(self.n_classes)
            
            for rule, cfs in zip(active_rules, active_cfs):
                act = self._calculate_activation(x, rule, mfs)  # MIN-композиция
                if act > 0:
                    confidences += act * cfs  # Сумма act×CF
            
            if np.max(confidences) > 0:
                if np.argmax(confidences) == true_class:
                    correct += 1
            else:
                if np.random.randint(self.n_classes) == true_class:
                    correct += 1
        
        return correct / self.Nall

    def calculate_fitness(self, chromosome):
        if self._stop_requested:
            return 0.0, 0, [], []

        if chromosome in self.fitness_cache:
            return self.fitness_cache[chromosome]

        gradations = GradationChromosome.decode_g_chromosome(chromosome, self.n_features)
        mfs = self._build_membership_functions(gradations)

        if self._stop_requested:
            return 0.0, 0, gradations, []

        fitness = self._evaluate_fitness(gradations, mfs)

        result = (fitness, sum(gradations), gradations, mfs)
        self.fitness_cache[chromosome] = result
        return result

    def create_initial_population(self):
        all_gradations = [GradationChromosome.GRADATION_VALUES for _ in range(self.n_features)]
        all_combinations = list(itertools.product(*all_gradations))
        total_combinations = len(all_combinations)

        segment_size = total_combinations // self.population_size
        remainder = total_combinations % self.population_size

        population = []
        start_idx = 0

        for i in range(self.population_size):
            if self._stop_requested:
                break

            current_segment_size = segment_size + (1 if i < remainder else 0)
            end_idx = start_idx + current_segment_size

            segment_combinations = all_combinations[start_idx:end_idx]
            selected_combo = random.choice(segment_combinations)

            chromosome = GradationChromosome.create_g_chromosome(list(selected_combo))
            population.append(chromosome)

            start_idx = end_idx

        return population

    def tournament_selection(self, evaluated_pop):
        candidates = random.sample(evaluated_pop, min(self.tournament_size, len(evaluated_pop)))
        return max(candidates, key=lambda x: (x[1], -x[2]))

    def crossover(self, p1, p2):
        length = len(p1)
        if length <= 1:
            return p1, p2

        points = sorted(random.sample(range(1, length), min(self.crossover_points, length-1)))
        points = [0] + points + [length]

        child1, child2 = [], []
        for i in range(len(points)-1):
            start, end = points[i], points[i+1]
            if i % 2 == 0:
                child1.extend(p1[start:end])
                child2.extend(p2[start:end])
            else:
                child1.extend(p2[start:end])
                child2.extend(p1[start:end])

        return tuple(child1), tuple(child2)

    def mutate(self, chrom):
        mutated = list(chrom)
        mutated_flag = False
        for i in range(len(mutated)):
            if random.random() < self.mutation_prob:
                mutated[i] = 1 - mutated[i]
                mutated_flag = True

        if not mutated_flag:
            return chrom, False

        try:
            gradations = GradationChromosome.decode_g_chromosome(tuple(mutated), self.n_features)
            for j in range(len(gradations)):
                if gradations[j] < 3:
                    gradations[j] = 3
                elif gradations[j] > 6:
                    gradations[j] = 6
            return GradationChromosome.create_g_chromosome(gradations), True
        except:
            return chrom, False

    def evaluate_on_training(self, gradations, mfs):
        """Оценка точности на всей выборке (тот же метод)"""
        return self._evaluate_fitness(gradations, mfs)

    def run(self):
        """Запуск ГА1"""
        self._log("\n" + "="*60)
        self._log("ЭТАП 1: ОПТИМИЗАЦИЯ ГРАДАЦИЙ (ГА1) - НА ВСЕЙ ВЫБОРКЕ")
        self._log("="*60)
        self._log(f"  Параметры ГА:")
        self._log(f"    • Популяция: {self.population_size}")
        self._log(f"    • Турнир: {self.tournament_size}")
        self._log(f"    • Кроссовер: {self.crossover_points} точек")
        self._log(f"    • Стагнация: {self.stall_generations} поколений")
        self._log(f"    • Признаков: {self.n_features}")
        self._log(f"    • Классов: {self.n_classes}")
        self._log(f"    • Примеров (ALL): {self.Nall}")

        train_accuracy = 0.0

        if self._stop_requested:
            self._log("Оптимизация остановлена пользователем")
            return [], 0.0, 0.0, []

        self._log("\n  Фаза 1: Создание начальной популяции...")
        population = self.create_initial_population()

        if self._stop_requested or not population:
            self._log("Оптимизация остановлена пользователем")
            return [], 0.0, 0.0, []

        self._log(f"  Фаза 2: Оценка {len(population)} хромосом...")
        evaluated = []
        for idx, chrom in enumerate(population):
            if self._stop_requested:
                break
            fitness, total_mf, grad, mfs = self.calculate_fitness(chrom)
            evaluated.append((chrom, fitness, total_mf, grad, mfs))
            self._log(f"    Хромосома {idx+1}/{len(population)}: fitness={fitness:.4f}, градации={grad}")

        if self._stop_requested or not evaluated:
            self._log("Оптимизация остановлена пользователем")
            return [], 0.0, 0.0, []

        best_chrom, best_fitness, best_mf, best_grad, best_mfs = max(
            evaluated, key=lambda x: (x[1], -x[2])
        )
        best_overall_fitness = best_fitness
        best_overall_grad = best_grad.copy()
        best_overall_mfs = best_mfs

        self._log(f"\n  ✓ Начальная популяция оценена:")
        self._log(f"    • Лучший fitness: {best_fitness:.4f}")
        self._log(f"    • Лучшие градации: {best_overall_grad}")
        self._log(f"    • Правил: {int(np.prod(best_overall_grad))}")
        self._log(f"    • Сумма градаций: {best_mf}")

        stall_counter = 0
        generation = 1

        self._log(f"\n  Фаза 3: Эволюция (поколения 1-{self.max_generations})...")
        self._log(f"  {'─'*50}")

        while (generation <= self.max_generations and 
            stall_counter < self.stall_generations and 
            not self._stop_requested):

            parent1 = self.tournament_selection(evaluated)
            parent2 = self.tournament_selection(evaluated)

            child1, child2 = self.crossover(parent1[0], parent2[0])
            child1, mutated1 = self.mutate(child1)
            child2, mutated2 = self.mutate(child2)

            f1, mf1, g1, mfs1 = self.calculate_fitness(child1)
            f2, mf2, g2, mfs2 = self.calculate_fitness(child2)

            new_evaluated = evaluated + [(child1, f1, mf1, g1, mfs1), (child2, f2, mf2, g2, mfs2)]
            new_evaluated.sort(key=lambda x: (x[1], -x[2]), reverse=True)
            new_evaluated = new_evaluated[:self.population_size]

            current_best = new_evaluated[0]
            improved = False
            if current_best[1] > best_overall_fitness or \
            (abs(current_best[1] - best_overall_fitness) < 1e-10 and current_best[2] < best_mf):
                best_overall_fitness = current_best[1]
                best_overall_grad = current_best[3].copy()
                best_overall_mfs = current_best[4]
                best_mf = current_best[2]
                stall_counter = 0
                improved = True
                self._log(f"  🔥 Поколение {generation}: УЛУЧШЕНИЕ!")
                self._log(f"      fitness={best_overall_fitness:.4f}, градации={best_overall_grad}")
                self._log(f"      правил={int(np.prod(best_overall_grad))}, сумма градаций={best_mf}")
            else:
                stall_counter += 1
                # Логируем каждое поколение
                if generation <= 5 or generation % 5 == 0:
                    self._log(f"  •  Поколение {generation}: fitness={best_overall_fitness:.4f}, "
                            f"градации={best_overall_grad}, стагнация={stall_counter}/{self.stall_generations}")

            if self.progress_callback:
                self.progress_callback(generation, self.max_generations, best_overall_fitness)

            evaluated = new_evaluated
            generation += 1

        if self._stop_requested:
            self._log(f"\n  ⚠ Оптимизация остановлена пользователем на поколении {generation-1}")

        self._log(f"\n  {'─'*50}")
        self._log(f"  Фаза 4: Финальная оценка...")
        train_accuracy = self.evaluate_on_training(best_overall_grad, best_overall_mfs)

        self._log(f"\n{'='*60}")
        self._log(f"РЕЗУЛЬТАТ ЭТАПА 1:")
        self._log(f"{'='*60}")
        self._log(f"  • Выполнено поколений: {generation-1}")
        self._log(f"  • Причина останова: {'стагнация' if stall_counter >= self.stall_generations else 'достигнут макс. поколений'}")
        self._log(f"  • Оптимальные градации: {best_overall_grad}")
        self._log(f"  • Точность на всей выборке: {best_overall_fitness:.4f} ({best_overall_fitness*100:.2f}%)")
        self._log(f"  • Точность на обучении: {train_accuracy:.4f}")
        self._log(f"  • Количество правил: {int(np.prod(best_overall_grad)):,}")
        self._log(f"  • Сумма градаций: {best_mf}")

        return best_overall_grad, best_overall_fitness, train_accuracy, best_overall_mfs


# ============================================================================
# ЭТАП 2: РЕДУКЦИЯ НЕЙРОНОВ-АНТЕЦЕДЕНТОВ (ГА2) - НА ВСЕЙ ВЫБОРКЕ
# ============================================================================

class AntecedentChromosome:
    @staticmethod
    def check_gradation_coverage(chromosome: tuple, gradations: List[int]) -> Tuple[bool, Dict]:
        n_features = len(gradations)
        grad_ranges = [range(g) for g in gradations]

        coverage = {}
        for f_idx, g in enumerate(gradations):
            for grad_idx in range(g):
                coverage[(f_idx, grad_idx)] = False

        rule_index = 0
        for rule in itertools.product(*grad_ranges):
            if chromosome[rule_index] == 1:
                for f_idx, grad_idx in enumerate(rule):
                    coverage[(f_idx, grad_idx)] = True
            rule_index += 1

        is_fully_covered = all(coverage.values())
        return is_fully_covered, coverage


class GeneticAntecedentReducer:
    """ЭТАП 2: Редукция правил на ВСЕЙ выборке"""

    def __init__(self, X_all, y_all, gradations: List[int],
                 membership_funcs: List[List], class_names: List[str],
                 baseline_accuracy: float, config: Dict,
                 progress_callback=None, log_callback=None):

        self.X_all = X_all
        self.y_all = y_all
        self.gradations = gradations
        self.membership_funcs = membership_funcs
        self.class_names = class_names
        self.n_features = len(gradations)
        self.n_classes = len(class_names)
        self.baseline_accuracy = baseline_accuracy
        self.Nall = len(y_all)

        self.N = int(np.prod(gradations))

        self.population_size = config.get('population_size', 20)
        self.tournament_size = config.get('tournament_size', 2)
        self.crossover_points = config.get('crossover_points', 2)
        self.stall_generations = config.get('stall_generations', 10)
        self.max_generations = config.get('max_generations', 30)
        self.mutation_prob = config.get('mutation_prob', 1.0 / self.N)
        self.verbose = config.get('verbose', True)

        self.progress_callback = progress_callback
        self.log_callback = log_callback

        self.accuracy_cache = {}

        self.all_rules = self._generate_all_rules()
        #self._precompute()

    def _log(self, msg):
        if self.log_callback:
            self.log_callback(msg)
        elif self.verbose:
            print(msg)

    def _generate_all_rules(self) -> List[tuple]:
        ranges = [range(g) for g in self.gradations]
        return list(itertools.product(*ranges))
    

    def _precompute(self):
        self._log(f"\nПредварительный расчёт активаций...")
        self._log(f"  • Правил: {self.N:,}")
        self._log(f"  • Примеров во всей выборке: {self.Nall:,}")
        
        # Временно сохраняем логи в буфер
        temp_fnn = OptimizedReducedFuzzyNeuralNetwork(
            n_features=self.n_features,
            n_classes=self.n_classes,
            gradations=self.gradations,
            membership_funcs=self.membership_funcs,
            active_rules=self.all_rules,
            active_cfs=[np.zeros(self.n_classes) for _ in self.all_rules]
        )
        
        self.all_activations = temp_fnn._vectorized_activation(self.X_all)
        
        # Обработка событий GUI
        from PyQt6.QtWidgets import QApplication
        QApplication.processEvents()
        
        self.rule_cfs = np.zeros((self.N, self.n_classes))
        for k in range(self.n_classes):
            class_mask = (self.y_all == k)
            class_activations = self.all_activations[class_mask]
            Nk = np.sum(class_mask)
            if Nk > 0:
                for rule_idx in range(self.N):
                    sum_act = np.sum(class_activations[:, rule_idx])
                    if sum_act > 0:
                        self.rule_cfs[rule_idx, k] = sum_act / Nk
            # Обработка событий GUI после каждого класса
            QApplication.processEvents()
        
        self._log(f"  ✓ Предрасчёт завершён")

    def precompute(self):
            """Публичный метод для предрасчёта (вызывается после подключения колбэков)"""
            self._precompute()
    def evaluate_accuracy(self, chromosome: tuple) -> float:
        cache_key = chromosome
        if cache_key in self.accuracy_cache:
            return self.accuracy_cache[cache_key]

        active_indices = [i for i, active in enumerate(chromosome) if active == 1]

        if not active_indices:
            self.accuracy_cache[cache_key] = 0.0
            return 0.0

        correct = 0
        for sample_idx in range(self.Nall):
            true_class = self.y_all[sample_idx]

            confidences = np.zeros(self.n_classes)

            for rule_idx in active_indices:
                act = self.all_activations[sample_idx, rule_idx]
                if act > 0:
                    confidences += act * self.rule_cfs[rule_idx]

            if np.max(confidences) > 0:
                if np.argmax(confidences) == true_class:
                    correct += 1

        accuracy = correct / self.Nall
        self.accuracy_cache[cache_key] = accuracy
        return accuracy

    def calculate_fitness(self, chromosome: tuple) -> Tuple[float, int, float]:
        passive_count = chromosome.count(0)
        accuracy = self.evaluate_accuracy(chromosome)

        if accuracy < self.baseline_accuracy - 1e-10:
            return 0.0, passive_count, accuracy

        return float(passive_count), passive_count, accuracy

    def create_initial_population(self) -> List[tuple]:
        self._log(f"\nСОЗДАНИЕ НАЧАЛЬНОЙ ПОПУЛЯЦИИ ГА2")
        population = []

        full_chromosome = tuple([1] * self.N)
        population.append(full_chromosome)

        for i in range(self.population_size - 1):
            density = 0.1 + (i / max(1, self.population_size - 2)) * 0.8
            target_ones = max(1, int(self.N * density))

            chromosome_list = [0] * self.N
            ones_indices = random.sample(range(self.N), target_ones)
            for idx in ones_indices:
                chromosome_list[idx] = 1

            chromosome = tuple(chromosome_list)
            is_covered, _ = AntecedentChromosome.check_gradation_coverage(chromosome, self.gradations)
            if is_covered:
                population.append(chromosome)
            else:
                population.append(full_chromosome)

        return population[:self.population_size]

    def tournament_selection(self, evaluated_pop: List[Tuple]) -> Tuple:
        candidates = random.sample(evaluated_pop, min(self.tournament_size, len(evaluated_pop)))
        return max(candidates, key=lambda x: (x[1], x[3]))

    def crossover(self, p1: tuple, p2: tuple) -> Tuple[tuple, tuple]:
        length = len(p1)
        if length <= 1:
            return p1, p2

        points = sorted(random.sample(range(1, length), min(self.crossover_points, length-1)))
        points = [0] + points + [length]

        child1, child2 = [], []
        for i in range(len(points)-1):
            start, end = points[i], points[i+1]
            if i % 2 == 0:
                child1.extend(p1[start:end])
                child2.extend(p2[start:end])
            else:
                child1.extend(p2[start:end])
                child2.extend(p1[start:end])

        child1_tuple = tuple(child1)
        child2_tuple = tuple(child2)

        is_covered1, _ = AntecedentChromosome.check_gradation_coverage(child1_tuple, self.gradations)
        is_covered2, _ = AntecedentChromosome.check_gradation_coverage(child2_tuple, self.gradations)

        if not is_covered1:
            child1_tuple = tuple([1] * self.N)
        if not is_covered2:
            child2_tuple = tuple([1] * self.N)

        return child1_tuple, child2_tuple

    def mutate(self, chromosome: tuple) -> tuple:
        mutated = list(chromosome)
        mutated_flag = False

        for i in range(len(mutated)):
            if random.random() < self.mutation_prob:
                mutated[i] = 1 - mutated[i]
                mutated_flag = True

        if not mutated_flag:
            return chromosome

        mutated_tuple = tuple(mutated)
        is_covered, _ = AntecedentChromosome.check_gradation_coverage(mutated_tuple, self.gradations)

        if sum(mutated_tuple) == 0 or not is_covered:
            return chromosome

        return mutated_tuple

    def reduction(self, population: List[Tuple]) -> List[Tuple]:
        if len(population) <= self.population_size:
            return population
        sorted_pop = sorted(population, key=lambda x: (x[1], x[3]), reverse=True)
        return sorted_pop[:self.population_size]

    def run(self) -> Tuple[tuple, int, float, Dict]:
        self._log("\n" + "="*60)
        self._log("ЭТАП 2: РЕДУКЦИЯ НЕЙРОНОВ-АНТЕЦЕДЕНТОВ (ГА2) - НА ВСЕЙ ВЫБОРКЕ")
        self._log("="*60)

        from PyQt6.QtWidgets import QApplication

        population = self.create_initial_population()
        evaluated = []
        for chrom in population:
            fitness, passive, accuracy = self.calculate_fitness(chrom)
            evaluated.append((chrom, fitness, passive, accuracy))
        QApplication.processEvents()

        best_chrom, best_fitness, best_passive, best_accuracy = max(
            evaluated, key=lambda x: (x[1], x[3])
        )
        best_overall_chrom = best_chrom
        best_overall_fitness = best_fitness
        best_overall_passive = best_passive
        best_overall_accuracy = best_accuracy

        stall_counter = 0
        generation = 1
        history = {'best_fitness': [], 'best_accuracy': [], 'best_passive': []}

        while generation <= self.max_generations and stall_counter < self.stall_generations:
            parent1 = self.tournament_selection(evaluated)
            parent2 = self.tournament_selection(evaluated)

            child1, child2 = self.crossover(parent1[0], parent2[0])
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)

            fitness1, passive1, accuracy1 = self.calculate_fitness(child1)
            fitness2, passive2, accuracy2 = self.calculate_fitness(child2)

            evaluated.append((child1, fitness1, passive1, accuracy1))
            evaluated.append((child2, fitness2, passive2, accuracy2))
            evaluated = self.reduction(evaluated)

            current_best = max(evaluated, key=lambda x: (x[1], x[3]))

            if (current_best[1] > best_overall_fitness or
                (abs(current_best[1] - best_overall_fitness) < 1e-10 and current_best[3] > best_overall_accuracy)):
                best_overall_chrom = current_best[0]
                best_overall_fitness = current_best[1]
                best_overall_passive = current_best[2]
                best_overall_accuracy = current_best[3]
                stall_counter = 0
            else:
                stall_counter += 1

            if self.progress_callback:
                self.progress_callback(generation, self.max_generations, best_overall_accuracy)

            # Даём GUI обновиться каждые 2 поколения
            if generation % 2 == 0:
                QApplication.processEvents()

            generation += 1

        reduction_pct = best_overall_passive / self.N * 100
        self._log(f"\nРЕЗУЛЬТАТ ЭТАПА 2:")
        self._log(f"  • Пассивных нейронов: {best_overall_passive} ({reduction_pct:.1f}%)")
        self._log(f"  • Активных нейронов: {self.N - best_overall_passive}")
        self._log(f"  • Точность на всей выборке: {best_overall_accuracy:.4f}")

        return best_overall_chrom, best_overall_passive, best_overall_accuracy, history

    def get_active_rules(self, chromosome: tuple) -> Tuple[List[int], List[tuple], List[np.ndarray]]:
        active_indices = [i for i, active in enumerate(chromosome) if active == 1]
        active_rules = [self.all_rules[i] for i in active_indices]
        active_cfs = [self.rule_cfs[i] for i in active_indices]
        return active_indices, active_rules, active_cfs


# ============================================================================
# ЭТАП 3: ПАРАЛЛЕЛЬНОЕ ОБУЧЕНИЕ (ГА3) - НА ВСЕЙ ВЫБОРКЕ
# ============================================================================

class ParallelSingleParameterGA3:
    """Параллельный ГА для одного параметра MF"""

    def __init__(self, param_min: float, param_max: float, m_bits: int = 4):
        self.param_min = param_min
        self.param_max = param_max
        self.m_bits = m_bits
        self.n_points = 2 ** m_bits
        self.possible_values = self._discretize_range()

    def _discretize_range(self) -> np.ndarray:
        all_points = np.linspace(self.param_min, self.param_max, self.n_points + 2)
        return all_points[1:-1]

    def encode(self, value_idx: int) -> tuple:
        bits = []
        for i in range(self.m_bits - 1, -1, -1):
            bits.append((value_idx >> i) & 1)
        return tuple(bits)

    def decode(self, chromosome: tuple) -> float:
        value_idx = 0
        for bit in chromosome:
            value_idx = (value_idx << 1) | bit
        value_idx = min(value_idx, self.n_points - 1)
        return self.possible_values[value_idx]

    def create_initial_population(self, pop_size: int) -> List[tuple]:
        population = []
        if pop_size >= self.n_points:
            for i in range(self.n_points):
                population.append(self.encode(i))
            while len(population) < pop_size:
                population.append(self.encode(random.randint(0, self.n_points - 1)))
        else:
            selected_indices = random.sample(range(self.n_points), pop_size)
            for idx in selected_indices:
                population.append(self.encode(idx))
        return population

    def mutate(self, chromosome: tuple, mutation_prob: float) -> tuple:
        mutated = list(chromosome)
        for i in range(len(mutated)):
            if random.random() < mutation_prob:
                mutated[i] = 1 - mutated[i]
        return tuple(mutated)

    def crossover(self, p1: tuple, p2: tuple, crossover_prob: float) -> Tuple[tuple, tuple]:
        if random.random() < crossover_prob and self.m_bits > 1:
            point = random.randint(1, self.m_bits - 1)
            c1 = p1[:point] + p2[point:]
            c2 = p2[:point] + p1[point:]
            return c1, c2
        return p1, p2

    def _tournament_selection(self, population: List, fitnesses: List, k: int) -> tuple:
        indices = random.sample(range(len(population)), min(k, len(population)))
        best_idx = max(indices, key=lambda i: fitnesses[i])
        return population[best_idx]

    def _parallel_evaluate(self, population: List[tuple], eval_context: Dict) -> List[float]:
        values = [self.decode(chrom) for chrom in population]
        mfs_data = _serialize_mfs(eval_context['mfs'])

        args_list = []
        for value in values:
            args = (
                value,
                eval_context['feature_idx'],
                eval_context['grad_idx'],
                eval_context['param_name'],
                mfs_data,
                eval_context['X_all'],
                eval_context['y_all'],
                eval_context['active_rules'],
                eval_context['active_cfs'],
                eval_context['n_classes']
            )
            args_list.append(args)

        with ProcessPoolExecutor(max_workers=N_PROCESSES) as executor:
            futures = [executor.submit(_evaluate_single_value_process, args) for args in args_list]
            fitnesses = [future.result() for future in futures]

        return fitnesses

    def run(self, eval_context: Dict, population_size: int = 20,
            max_generations: int = 30, tournament_size: int = 3,
            crossover_prob: float = 0.8, mutation_prob: float = 0.1,
            stall_generations: int = 10, verbose: bool = False) -> Tuple[float, float]:

        population = self.create_initial_population(population_size)
        fitnesses = self._parallel_evaluate(population, eval_context)

        best_idx = np.argmax(fitnesses)
        best_chrom = population[best_idx]
        best_fitness = fitnesses[best_idx]
        best_value = self.decode(best_chrom)

        stall_counter = 0
        generation = 1

        while generation <= max_generations and stall_counter < stall_generations:
            new_population = []
            elite_idx = np.argmax(fitnesses)
            new_population.append(population[elite_idx])

            while len(new_population) < population_size:
                p1 = self._tournament_selection(population, fitnesses, tournament_size)
                p2 = self._tournament_selection(population, fitnesses, tournament_size)
                c1, c2 = self.crossover(p1, p2, crossover_prob)
                c1 = self.mutate(c1, mutation_prob)
                c2 = self.mutate(c2, mutation_prob)
                new_population.append(c1)
                if len(new_population) < population_size:
                    new_population.append(c2)

            population = new_population
            fitnesses = self._parallel_evaluate(population, eval_context)

            current_best_idx = np.argmax(fitnesses)
            current_best_fitness = fitnesses[current_best_idx]

            if current_best_fitness > best_fitness + 1e-6:
                best_fitness = current_best_fitness
                best_chrom = population[current_best_idx]
                best_value = self.decode(best_chrom)
                stall_counter = 0
            else:
                stall_counter += 1

            generation += 1

        return best_value, best_fitness


class ParallelGA3CyclicTuner:
    """ЭТАП 3: Параллельное циклическое обучение на ВСЕЙ выборке"""

    def __init__(self, fnn: OptimizedReducedFuzzyNeuralNetwork,
                 X_all: np.ndarray, y_all: np.ndarray,
                 config: Dict, progress_callback=None, log_callback=None):
        self.fnn = fnn
        self.X_all = X_all
        self.y_all = y_all

        self.m_bits = config.get('m_bits', 3)
        self.population_size = config.get('population_size', 16)
        self.max_generations = config.get('max_generations', 15)
        self.tournament_size = config.get('tournament_size', 3)
        self.crossover_prob = config.get('crossover_prob', 0.8)
        self.mutation_prob = config.get('mutation_prob', 0.1)
        self.stall_generations = config.get('stall_generations', 5)
        self.max_epochs = config.get('max_epochs', 10)
        self.stall_epochs = config.get('stall_epochs', 2)
        self.verbose = config.get('verbose', True)

        self.progress_callback = progress_callback
        self.log_callback = log_callback

        self._evaluation_cache = {}
        self.best_all_accuracy = 0
        self.best_mfs = None

    def _log(self, msg):
        if self.log_callback:
            self.log_callback(msg)
        elif self.verbose:
            print(msg)

    def _get_all_tunable_params(self) -> List[Dict]:
        tunable_params = []
        for f_idx, feature_mfs in enumerate(self.fnn.membership_funcs):
            n_terms = len(feature_mfs)
            for g_idx, mf in enumerate(feature_mfs):
                a, b, c, d = mf.get_params()
                if g_idx == 0 and n_terms > 1:
                    next_mf = feature_mfs[g_idx + 1]
                    next_a, next_b, next_c, next_d = next_mf.get_params()
                    tunable_params.append({'feature_idx': f_idx, 'grad_idx': g_idx, 'param_name': 'c', 'current_value': c, 'min_value': b, 'max_value': next_b})
                    tunable_params.append({'feature_idx': f_idx, 'grad_idx': g_idx, 'param_name': 'd', 'current_value': d, 'min_value': c, 'max_value': next_c})
                elif g_idx == n_terms - 1 and n_terms > 1:
                    prev_mf = feature_mfs[g_idx - 1]
                    prev_a, prev_b, prev_c, prev_d = prev_mf.get_params()
                    tunable_params.append({'feature_idx': f_idx, 'grad_idx': g_idx, 'param_name': 'a', 'current_value': a, 'min_value': prev_c, 'max_value': b})
                    tunable_params.append({'feature_idx': f_idx, 'grad_idx': g_idx, 'param_name': 'b', 'current_value': b, 'min_value': a, 'max_value': c})
                elif 0 < g_idx < n_terms - 1:
                    prev_mf = feature_mfs[g_idx - 1]
                    next_mf = feature_mfs[g_idx + 1]
                    prev_a, prev_b, prev_c, prev_d = prev_mf.get_params()
                    next_a, next_b, next_c, next_d = next_mf.get_params()
                    tunable_params.extend([
                        {'feature_idx': f_idx, 'grad_idx': g_idx, 'param_name': 'a', 'current_value': a, 'min_value': prev_c, 'max_value': b},
                        {'feature_idx': f_idx, 'grad_idx': g_idx, 'param_name': 'b', 'current_value': b, 'min_value': a, 'max_value': c},
                        {'feature_idx': f_idx, 'grad_idx': g_idx, 'param_name': 'c', 'current_value': c, 'min_value': b, 'max_value': next_a},
                        {'feature_idx': f_idx, 'grad_idx': g_idx, 'param_name': 'd', 'current_value': d, 'min_value': c, 'max_value': next_b}
                    ])
        return tunable_params

    def run(self) -> Tuple[List[List], float, Dict]:
        self._log("\n" + "="*60)
        self._log("ЭТАП 3: ПАРАЛЛЕЛЬНОЕ ОБУЧЕНИЕ НА ВСЕЙ ВЫБОРКЕ")
        self._log("="*60)
        return self.fnn.membership_funcs, 0.0, {}