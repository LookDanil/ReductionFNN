"""ГА2: Редукция нейронов-антецедентов"""
import numpy as np
import random
import itertools
from typing import List, Tuple, Dict
from core.fnn_model import OptimizedReducedFuzzyNeuralNetwork
from PyQt6.QtWidgets import QApplication


class GeneticAntecedentReducer:
    """ЭТАП 2: Редукция правил на TRAIN"""

    def __init__(self, X_all, y_all, gradations, membership_funcs,
                 class_names, baseline_accuracy, config,
                 progress_callback=None, log_callback=None,
                 active_rules=None, active_cfs=None):
        
        self.X_all = X_all
        self.y_all = y_all
        self.gradations = gradations
        self.membership_funcs = membership_funcs
        self.class_names = class_names
        self.n_features = len(gradations)
        self.n_classes = len(class_names)
        self.baseline_accuracy = baseline_accuracy
        self.Nall = len(y_all)

        if active_rules is not None and active_cfs is not None:
            self.all_rules = active_rules
            self.rule_cfs = np.array(active_cfs)
            self.N = len(active_rules)
            self._precomputed = True
        else:
            self.all_rules = self._generate_all_rules()
            self.N = len(self.all_rules)
            self._precomputed = False

        self.population_size = config.get('population_size', 20)
        self.tournament_size = config.get('tournament_size', 2)
        self.crossover_points = config.get('crossover_points', 2)
        self.stall_generations = config.get('stall_generations', 10)
        self.max_generations = config.get('max_generations', 10000)
        self.mutation_prob = config.get('mutation_prob', 1.0 / self.N if self.N > 0 else 0.01)
        self.verbose = config.get('verbose', True)

        self.progress_callback = progress_callback
        self.log_callback = log_callback

        self.accuracy_cache = {}
        self._best_overall_chrom = None
        self._best_overall_fnn = None

    def get_best_fnn(self):
        """Возвращает лучшую FNN на текущий момент"""
        return self._best_overall_fnn

    def precompute(self):
        if not self._precomputed:
            self._precompute()
        else:
            self._log("  ✓ Правила уже отфильтрованы, предрасчёт не требуется")

    def _log(self, msg):
        if self.log_callback:
            self.log_callback(msg)
        elif self.verbose:
            print(msg)

    def _generate_all_rules(self):
        ranges = [range(g) for g in self.gradations]
        return list(itertools.product(*ranges))

    def _precompute(self):
        self._log(f"\nПредварительный расчёт активаций...")
        self._log(f"  • Правил: {self.N:,}")
        self._log(f"  • Примеров во всей выборке: {self.Nall:,}")

        temp_fnn = OptimizedReducedFuzzyNeuralNetwork(
            n_features=self.n_features, n_classes=self.n_classes,
            gradations=self.gradations, membership_funcs=self.membership_funcs,
            active_rules=self.all_rules,
            active_cfs=[np.zeros(self.n_classes) for _ in self.all_rules]
        )
        self.all_activations = temp_fnn._vectorized_activation(self.X_all)
        QApplication.processEvents()

        if not self._precomputed:
            self.rule_cfs = np.zeros((self.N, self.n_classes))
            for k in range(self.n_classes):
                class_mask = (self.y_all == k)
                Nk = np.sum(class_mask)
                if Nk > 0:
                    sum_acts = np.sum(self.all_activations[class_mask], axis=0)
                    mask = sum_acts > 0
                    self.rule_cfs[mask, k] = sum_acts[mask] / Nk

        self._log(f"  ✓ Предрасчёт завершён")

    def _build_fnn_for_chromosome(self, chromosome):
        """Построить FNN для хромосомы"""
        active_indices = [i for i, a in enumerate(chromosome) if a == 1]
        if not active_indices:
            return None
        
        active_rules = [self.all_rules[i] for i in active_indices]
        active_cfs = [self.rule_cfs[i] for i in active_indices]
        
        return OptimizedReducedFuzzyNeuralNetwork(
            n_features=self.n_features,
            n_classes=self.n_classes,
            gradations=self.gradations,
            membership_funcs=self.membership_funcs,
            active_rules=active_rules,
            active_cfs=active_cfs
        )

    def _check_active_coverage(self, chromosome):
        coverage = {}
        for f_idx, g in enumerate(self.gradations):
            for grad_idx in range(g):
                coverage[(f_idx, grad_idx)] = False

        for rule_idx, active in enumerate(chromosome):
            if active == 1:
                rule = self.all_rules[rule_idx]
                for f_idx, grad_idx in enumerate(rule):
                    coverage[(f_idx, grad_idx)] = True

        return all(coverage.values())

    def evaluate_accuracy(self, chromosome):
        cache_key = chromosome
        if cache_key in self.accuracy_cache:
            return self.accuracy_cache[cache_key]

        fnn = self._build_fnn_for_chromosome(chromosome)
        if fnn is None:
            self.accuracy_cache[cache_key] = 0.0
            return 0.0

        from sklearn.metrics import accuracy_score
        y_pred = fnn.predict(self.X_all)
        accuracy = accuracy_score(self.y_all, y_pred)

        self.accuracy_cache[cache_key] = accuracy
        return accuracy

    def calculate_fitness(self, chromosome):
        passive_count = chromosome.count(0)
        accuracy = self.evaluate_accuracy(chromosome)

        if accuracy > self.baseline_accuracy:
            self.baseline_accuracy = accuracy
            self._log(f"  🔥 Новый baseline: {accuracy:.4f} (улучшение точности!)")

        if accuracy < self.baseline_accuracy - 1e-10:
            return 0.0, passive_count, accuracy

        return float(passive_count), passive_count, accuracy

    def create_initial_population(self):
        population = [tuple([1] * self.N)]

        for i in range(self.population_size - 1):
            density = 0.1 + (i / max(1, self.population_size - 2)) * 0.8
            target_ones = max(1, int(self.N * density))

            chromosome_list = [0] * self.N
            ones_indices = random.sample(range(self.N), target_ones)
            for idx in ones_indices:
                chromosome_list[idx] = 1

            chromosome = tuple(chromosome_list)
            is_covered = self._check_active_coverage(chromosome)

            if is_covered:
                population.append(chromosome)
            else:
                population.append(population[0])

        return population[:self.population_size]

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
        child1_tuple = tuple(child1)
        child2_tuple = tuple(child2)
        is_covered1 = self._check_active_coverage(child1_tuple)
        is_covered2 = self._check_active_coverage(child2_tuple)
        if not is_covered1:
            child1_tuple = tuple([1] * self.N)
        if not is_covered2:
            child2_tuple = tuple([1] * self.N)
        return child1_tuple, child2_tuple

    def mutate(self, chromosome):
        mutated = list(chromosome)
        mutated_flag = False
        for i in range(len(mutated)):
            if random.random() < self.mutation_prob:
                mutated[i] = 1 - mutated[i]
                mutated_flag = True
        if not mutated_flag:
            return chromosome
        mutated_tuple = tuple(mutated)
        is_covered = self._check_active_coverage(mutated_tuple)
        if sum(mutated_tuple) == 0 or not is_covered:
            return chromosome
        return mutated_tuple

    def reduction(self, population):
        if len(population) <= self.population_size:
            return population
        sorted_pop = sorted(population, key=lambda x: x[1], reverse=True)
        return sorted_pop[:self.population_size]

    def run(self) -> Tuple[tuple, int, float, Dict, OptimizedReducedFuzzyNeuralNetwork]:
        self._log("\n" + "="*60)
        self._log("ЭТАП 2: РЕДУКЦИЯ НЕЙРОНОВ-АНТЕЦЕДЕНТОВ (ГА2) - НА TRAIN")
        self._log("="*60)
        self._log(f"Начальная точность (baseline): {self.baseline_accuracy:.4f}")
        self._log(f"Активных правил: {self.N}")

        from PyQt6.QtWidgets import QApplication

        population = self.create_initial_population()
        evaluated = []
        for chrom in population:
            fitness, passive, accuracy = self.calculate_fitness(chrom)
            evaluated.append((chrom, fitness, passive, accuracy))
        QApplication.processEvents()

        best_idx = max(range(len(evaluated)), key=lambda i: evaluated[i][1])
        best_chrom, best_fitness, best_passive, best_accuracy = evaluated[best_idx]
        best_overall_chrom = best_chrom
        best_overall_fitness = best_fitness
        best_overall_passive = best_passive
        best_overall_accuracy = best_accuracy

        self._best_overall_chrom = best_chrom
        self._best_overall_fnn = self._build_fnn_for_chromosome(best_chrom)

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

            child1_tuple = (child1, fitness1, passive1, accuracy1)
            child2_tuple = (child2, fitness2, passive2, accuracy2)

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
                best_overall_chrom = current_best[0]
                best_overall_fitness = current_best[1]
                best_overall_passive = current_best[2]
                best_overall_accuracy = current_best[3]
                self._best_overall_chrom = current_best[0]
                self._best_overall_fnn = self._build_fnn_for_chromosome(current_best[0])

                self._log(f"  🔥 Поколение {generation}: улучшение!")
                self._log(f"     Пассивных: {best_overall_passive} ({best_overall_passive/self.N*100:.1f}%)")
                self._log(f"     Точность: {best_overall_accuracy:.4f}")

            if generation <= 5 or generation % 10 == 0:
                self._log(f"  •  Поколение {generation}: пассивных={best_overall_fitness}, "
                         f"стагнация={stall_counter}/{self.stall_generations}")

            if self.progress_callback:
                self.progress_callback(generation, self.max_generations, best_overall_accuracy)

            if generation % 2 == 0:
                QApplication.processEvents()

            generation += 1

        max_passive = max(evaluated, key=lambda x: x[1])[1]
        best_of_best = max([e for e in evaluated if abs(e[1] - max_passive) < 1e-10], key=lambda x: x[3])

        best_overall_chrom = best_of_best[0]
        best_overall_passive = best_of_best[2]
        best_overall_accuracy = best_of_best[3]
        self._best_overall_fnn = self._build_fnn_for_chromosome(best_overall_chrom)

        reduction_pct = best_overall_passive / self.N * 100
        self._log(f"\nРЕЗУЛЬТАТ ЭТАПА 2:")
        self._log(f"  • Пассивных нейронов: {best_overall_passive} ({reduction_pct:.1f}%)")
        self._log(f"  • Активных нейронов: {self.N - best_overall_passive}")
        self._log(f"  • Точность (TRAIN): {best_overall_accuracy:.4f}")
        self._log(f"  • Baseline (конечный): {self.baseline_accuracy:.4f}")

        result_fnn = self._best_overall_fnn

        return best_overall_chrom, best_overall_passive, best_overall_accuracy, history, result_fnn

    def get_active_rules(self, chromosome):
        active_indices = [i for i, active in enumerate(chromosome) if active == 1]
        active_rules = [self.all_rules[i] for i in active_indices]
        active_cfs = [self.rule_cfs[i] for i in active_indices]
        return active_indices, active_rules, active_cfs