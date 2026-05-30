"""ГА1: Оптимизация градаций"""
import numpy as np
import random
import time
import itertools
from typing import List, Tuple
from sklearn.cluster import KMeans
from core.fuzzy_system import TrapezoidalMF
from core.ga_common import GradationChromosome
from core.fnn_model import OptimizedReducedFuzzyNeuralNetwork


class GeneticGradationOptimizer:
    """ЭТАП 1: Оптимизация градаций на TRAIN"""

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

        # Лучшая модель на текущий момент
        self._best_overall_fnn = None

    def _log(self, msg):
        if self.log_callback:
            self.log_callback(msg)
        elif self.verbose:
            print(msg)

    def request_stop(self):
        self._stop_requested = True

    def get_mfs_for_gradations(self, gradations):
        return self._build_membership_functions(gradations)

    def get_best_fnn(self):
        """Возвращает лучшую FNN на текущий момент"""
        return self._best_overall_fnn

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

    def _build_fnn_for_gradations(self, gradations, mfs):
        """Построить полную FNN для заданных градаций"""
        ranges = [range(g) for g in gradations]
        all_rules = list(itertools.product(*ranges))

        fnn = OptimizedReducedFuzzyNeuralNetwork(
            n_features=self.n_features,
            n_classes=self.n_classes,
            gradations=gradations,
            membership_funcs=mfs,
            active_rules=all_rules,
            active_cfs=[np.zeros(self.n_classes) for _ in all_rules]
        )

        activations = fnn._vectorized_activation(self.X_all)
        rule_cfs = np.zeros((len(all_rules), self.n_classes))
        for k in range(self.n_classes):
            class_mask = (self.y_all == k)
            class_activations = activations[class_mask]
            Nk = np.sum(class_mask)
            if Nk > 0:
                for r_idx in range(len(all_rules)):
                    sum_act = np.sum(class_activations[:, r_idx])
                    if sum_act > 0:
                        rule_cfs[r_idx, k] = sum_act / Nk

        fnn.rule_cfs_array = rule_cfs
        return fnn

    def _evaluate_fitness(self, gradations, mfs):
        """Оценка фитнеса на TRAIN через FNN"""
        fnn = self._build_fnn_for_gradations(gradations, mfs)
        return fnn.evaluate(self.X_all, self.y_all)

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
        return max(candidates, key=lambda x: x[1])

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

    def run(self):
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
        best_overall_fnn = None
        self._best_overall_fnn = None

        if self._stop_requested:
            self._log("Оптимизация остановлена пользователем")
            return [], 0.0, 0.0, [], None

        self._log("\n  Фаза 1: Создание начальной популяции...")
        population = self.create_initial_population()

        if self._stop_requested or not population:
            self._log("Оптимизация остановлена пользователем")
            return [], 0.0, 0.0, [], None

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
            return [], 0.0, 0.0, [], None

        best_idx = max(range(len(evaluated)), key=lambda i: evaluated[i][1])
        best_chrom, best_fitness, best_mf, best_grad, best_mfs = evaluated[best_idx]
        best_overall_fitness = best_fitness
        best_overall_grad = best_grad.copy()
        best_overall_mfs = best_mfs
        best_overall_fnn = self._build_fnn_for_gradations(best_overall_grad, best_overall_mfs)
        self._best_overall_fnn = best_overall_fnn

        self._log(f"\n  ✓ Начальная популяция оценена:")
        self._log(f"    • Лучший fitness: {best_fitness:.4f}")
        self._log(f"    • Лучшие градации: {best_overall_grad}")
        self._log(f"    • Правил: {int(np.prod(best_overall_grad))}")

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
            child1, _ = self.mutate(child1)
            child2, _ = self.mutate(child2)
            f1, mf1, g1, mfs1 = self.calculate_fitness(child1)
            f2, mf2, g2, mfs2 = self.calculate_fitness(child2)

            child1_tuple = (child1, f1, mf1, g1, mfs1)
            child2_tuple = (child2, f2, mf2, g2, mfs2)

            evaluated.append(child1_tuple)
            evaluated.append(child2_tuple)

            evaluated.sort(key=lambda x: x[1], reverse=True)

            worst1 = evaluated[-1]
            worst2 = evaluated[-2]

            evaluated = evaluated[:self.population_size]

            child1_survived = child1_tuple not in [worst1, worst2]
            child2_survived = child2_tuple not in [worst1, worst2]

            if not child1_survived and not child2_survived:
                stall_counter += 1
            else:
                stall_counter = 0

            current_best = evaluated[0]
            if current_best[1] > best_overall_fitness:
                best_overall_fitness = current_best[1]
                best_overall_grad = current_best[3].copy()
                best_overall_mfs = current_best[4]
                best_overall_fnn = self._build_fnn_for_gradations(best_overall_grad, best_overall_mfs)
                self._best_overall_fnn = best_overall_fnn
                self._log(f"  🔥 Поколение {generation}: УЛУЧШЕНИЕ!")
                self._log(f"      fitness={best_overall_fitness:.4f}, градации={best_overall_grad}")

            if generation <= 5 or generation % 5 == 0:
                self._log(f"  •  Поколение {generation}: fitness={best_overall_fitness:.4f}, "
                         f"стагнация={stall_counter}/{self.stall_generations}")

            if self.progress_callback:
                self.progress_callback(generation, self.max_generations, best_overall_fitness)

            generation += 1

        if self._stop_requested:
            self._log(f"\n  ⚠ Оптимизация остановлена пользователем на поколении {generation-1}")

        self._log(f"\n  {'─'*50}")
        self._log(f"  Финальная оценка (используем лучшую модель из ГА)...")
        train_accuracy = best_overall_fnn.evaluate(self.X_all, self.y_all) if best_overall_fnn else 0.0

        self._log(f"\n{'='*60}")
        self._log(f"РЕЗУЛЬТАТ ЭТАПА 1:")
        self._log(f"{'='*60}")
        self._log(f"  • Выполнено поколений: {generation-1}")
        self._log(f"  • Причина останова: {'стагнация' if stall_counter >= self.stall_generations else 'остановлен пользователем'}")
        self._log(f"  • Оптимальные градации: {best_overall_grad}")
        self._log(f"  • Точность на всей выборке: {best_overall_fitness:.4f} ({best_overall_fitness*100:.2f}%)")
        self._log(f"  • Количество правил: {int(np.prod(best_overall_grad)):,}")

        return best_overall_grad, best_overall_fitness, train_accuracy, best_overall_mfs, best_overall_fnn