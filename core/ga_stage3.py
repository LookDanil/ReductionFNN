"""ГА3: Параллельная тонкая настройка ФП"""
import numpy as np
import random
import time
from typing import List, Tuple, Dict
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor
from core.fuzzy_system import TrapezoidalMF
from core.fnn_model import OptimizedReducedFuzzyNeuralNetwork
from core.ga_utils import serialize_mfs, evaluate_single_value_process
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
try:
    import multiprocessing
    N_CORES = multiprocessing.cpu_count()
except:
    N_CORES = 1

N_PROCESSES = max(1, min(N_CORES - 2, 4))


class ParallelSingleParameterGA3:
    def __init__(self, param_min, param_max, m_bits=4):
        self.param_min = param_min
        self.param_max = param_max
        self.m_bits = m_bits
        self.n_points = 2 ** m_bits
        self.possible_values = self._discretize_range()

    def _discretize_range(self):
        all_points = np.linspace(self.param_min, self.param_max, self.n_points + 2)
        return all_points[1:-1]

    def encode(self, value_idx):
        bits = []
        for i in range(self.m_bits - 1, -1, -1):
            bits.append((value_idx >> i) & 1)
        return tuple(bits)

    def decode(self, chromosome):
        value_idx = 0
        for bit in chromosome:
            value_idx = (value_idx << 1) | bit
        value_idx = min(value_idx, self.n_points - 1)
        return self.possible_values[value_idx]

    def create_initial_population(self, pop_size):
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

    def mutate(self, chromosome, mutation_prob):
        mutated = list(chromosome)
        for i in range(len(mutated)):
            if random.random() < mutation_prob:
                mutated[i] = 1 - mutated[i]
        return tuple(mutated)

    def crossover(self, p1, p2, crossover_prob):
        if random.random() < crossover_prob and self.m_bits > 1:
            point = random.randint(1, self.m_bits - 1)
            return p1[:point] + p2[point:], p2[:point] + p1[point:]
        return p1, p2

    def _tournament_selection(self, population, fitnesses, k):
        indices = random.sample(range(len(population)), min(k, len(population)))
        best_idx = max(indices, key=lambda i: fitnesses[i])
        return population[best_idx]

    def _parallel_evaluate(self, population, eval_context, progress_callback=None):
        values = [self.decode(chrom) for chrom in population]
        mfs_data = serialize_mfs(eval_context['mfs'])
        
        args_list = []
        for value in values:
            args = (value, eval_context['feature_idx'], eval_context['grad_idx'],
                    eval_context['param_name'], mfs_data, eval_context['X_all'],
                    eval_context['y_all'], eval_context['active_rules'],
                    eval_context['active_cfs'], eval_context['n_classes'])
            args_list.append(args)
        
        fitnesses = [None] * len(args_list)
        
        try:
            with ProcessPoolExecutor(max_workers=N_PROCESSES) as executor:
                futures = [executor.submit(evaluate_single_value_process, args) for args in args_list]
                
                import time
                from PyQt6.QtWidgets import QApplication
                
                done_count = 0
                while done_count < len(futures):
                    done_count = 0
                    for i, f in enumerate(futures):
                        if fitnesses[i] is not None:
                            done_count += 1
                        elif f.done():
                            try:
                                fitnesses[i] = f.result()
                            except Exception:
                                fitnesses[i] = 0.0
                            done_count += 1
                    
                    if progress_callback:
                        progress_callback(done_count, len(futures))
                    
                    QApplication.processEvents()
                    
                    if done_count < len(futures):
                        time.sleep(0.02)
        except Exception as e:
            # Если ProcessPoolExecutor не сработал — выполняем последовательно
            for i, args in enumerate(args_list):
                fitnesses[i] = evaluate_single_value_process(args)
        
        return fitnesses

    def run(self, eval_context, population_size=20, max_generations=30,
            tournament_size=3, crossover_prob=0.8, mutation_prob=0.1,
            stall_generations=10, verbose=False, progress_callback=None):
        
        population = self.create_initial_population(population_size)
        fitnesses = self._parallel_evaluate(population, eval_context, progress_callback)
        
        best_idx = np.argmax(fitnesses)
        best_chrom = population[best_idx]
        best_fitness = fitnesses[best_idx]
        best_value = self.decode(best_chrom)
        
        stall_counter = 0
        generation = 1
        
        while generation <= max_generations and stall_counter < stall_generations:
            new_population = [population[np.argmax(fitnesses)]]
            
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
            
            # Асинхронная оценка с прогрессом
            if progress_callback:
                progress_callback(0, population_size, msg=f"Поколение {generation}/{max_generations}")
            fitnesses = self._parallel_evaluate(population, eval_context, progress_callback)
            
            current_best_idx = np.argmax(fitnesses)
            if fitnesses[current_best_idx] > best_fitness + 1e-6:
                best_fitness = fitnesses[current_best_idx]
                best_chrom = population[current_best_idx]
                best_value = self.decode(best_chrom)
                stall_counter = 0
            else:
                stall_counter += 1
            
            generation += 1
        
        return best_value, best_fitness


class ParallelGA3CyclicTuner:
    def __init__(self, fnn, X_all, y_all, config, progress_callback=None, log_callback=None):
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

    def _get_all_tunable_params(self):
        tunable_params = []
        for f_idx, feature_mfs in enumerate(self.fnn.membership_funcs):
            n_terms = len(feature_mfs)
            for g_idx, mf in enumerate(feature_mfs):
                a, b, c, d = mf.get_params()
                if g_idx == 0 and n_terms > 1:
                    next_mf = feature_mfs[g_idx + 1]
                    nb = next_mf.get_params()[1]
                    nc = next_mf.get_params()[2]
                    tunable_params.append({'feature_idx': f_idx, 'grad_idx': g_idx, 'param_name': 'c', 'current_value': c, 'min_value': b, 'max_value': nb})
                    tunable_params.append({'feature_idx': f_idx, 'grad_idx': g_idx, 'param_name': 'd', 'current_value': d, 'min_value': c, 'max_value': nc})
                elif g_idx == n_terms - 1 and n_terms > 1:
                    prev_mf = feature_mfs[g_idx - 1]
                    pc = prev_mf.get_params()[2]
                    tunable_params.append({'feature_idx': f_idx, 'grad_idx': g_idx, 'param_name': 'a', 'current_value': a, 'min_value': pc, 'max_value': b})
                    tunable_params.append({'feature_idx': f_idx, 'grad_idx': g_idx, 'param_name': 'b', 'current_value': b, 'min_value': a, 'max_value': c})
                elif 0 < g_idx < n_terms - 1:
                    prev_mf = feature_mfs[g_idx - 1]
                    next_mf = feature_mfs[g_idx + 1]
                    pc = prev_mf.get_params()[2]
                    na = next_mf.get_params()[0]
                    nb = next_mf.get_params()[1]
                    tunable_params.extend([
                        {'feature_idx': f_idx, 'grad_idx': g_idx, 'param_name': 'a', 'current_value': a, 'min_value': pc, 'max_value': b},
                        {'feature_idx': f_idx, 'grad_idx': g_idx, 'param_name': 'b', 'current_value': b, 'min_value': a, 'max_value': c},
                        {'feature_idx': f_idx, 'grad_idx': g_idx, 'param_name': 'c', 'current_value': c, 'min_value': b, 'max_value': na},
                        {'feature_idx': f_idx, 'grad_idx': g_idx, 'param_name': 'd', 'current_value': d, 'min_value': c, 'max_value': nb}
                    ])
        return tunable_params

    def _create_eval_context(self, param_info):
        return {
            'feature_idx': param_info['feature_idx'],
            'grad_idx': param_info['grad_idx'],
            'param_name': param_info['param_name'],
            'mfs': self.fnn.membership_funcs,
            'X_all': self.X_all,
            'y_all': self.y_all,
            'active_rules': self.fnn.active_rules,
            'active_cfs': self.fnn.active_cfs,
            'n_classes': self.fnn.n_classes
        }

    def _tune_single_parameter(self, param_info, current_all_acc):
        ga3 = ParallelSingleParameterGA3(param_info['min_value'], param_info['max_value'], self.m_bits)
        eval_context = self._create_eval_context(param_info)
        
        def on_ga3_progress(done, total, msg=None):
            # Прокидываем прогресс наверх
            pass  # Можно добавить логирование если нужно
        
        best_value, new_all_acc = ga3.run(
            eval_context=eval_context, population_size=self.population_size,
            max_generations=self.max_generations, tournament_size=self.tournament_size,
            crossover_prob=self.crossover_prob, mutation_prob=self.mutation_prob,
            stall_generations=self.stall_generations, verbose=False,
            progress_callback=on_ga3_progress
        )
        
        if new_all_acc <= current_all_acc + 1e-6:
            return False, current_all_acc
        
        mf = self.fnn.membership_funcs[param_info['feature_idx']][param_info['grad_idx']]
        old_params = mf.get_params()
        a, b, c, d = old_params
        
        if param_info['param_name'] == 'a': a = best_value
        elif param_info['param_name'] == 'b': b = best_value
        elif param_info['param_name'] == 'c': c = best_value
        elif param_info['param_name'] == 'd': d = best_value
        
        self.fnn.membership_funcs[param_info['feature_idx']][param_info['grad_idx']] = TrapezoidalMF(a, b, c, d)
        
        if new_all_acc > self.best_all_accuracy:
            self.best_all_accuracy = new_all_acc
            self.best_mfs = deepcopy(self.fnn.membership_funcs)
            param_info['current_value'] = best_value
            return True, new_all_acc
        else:
            self.fnn.membership_funcs[param_info['feature_idx']][param_info['grad_idx']] = TrapezoidalMF(*old_params)
            return False, current_all_acc

    def run(self):
        self._log("\n" + "="*60)
        self._log("ЭТАП 3: ПАРАЛЛЕЛЬНОЕ ОБУЧЕНИЕ НА ВСЕЙ ВЫБОРКЕ")
        self._log("="*60)
        
        from PyQt6.QtWidgets import QApplication
        
        all_params = self._get_all_tunable_params()
        
        self._log(f"\nНастраиваемых параметров: {len(all_params)}")
        self._log(f"Точность дискретизации: m={self.m_bits} ({2**self.m_bits} точек)")
        self._log(f"Параллельных процессов: {N_PROCESSES} (1 ядро для GUI)")
        
        initial_all_acc = self.fnn.evaluate(self.X_all, self.y_all)
        
        self.best_all_accuracy = initial_all_acc
        self.best_mfs = deepcopy(self.fnn.membership_funcs)
        current_all_acc = initial_all_acc
        
        self._log(f"Начальная точность: {initial_all_acc:.4f}")
        
        history = {'epoch': [], 'accuracy': [], 'improvements': []}
        
        epoch = 0
        stall_counter = 0
        start_time = time.time()
        total_improvements = 0
        
        while epoch < self.max_epochs and stall_counter < self.stall_epochs:
            epoch += 1
            epoch_start = time.time()
            epoch_best_before = self.best_all_accuracy
            epoch_improved = False
            epoch_improvements = 0
            
            self._log(f"\n{'='*50}")
            self._log(f"ЭПОХА {epoch}/{self.max_epochs}")
            self._log(f"  Текущая точность: {self.best_all_accuracy:.4f}")
            self._log(f"  Стагнация: {stall_counter}/{self.stall_epochs}")
            self._log(f"{'='*50}")
            
            param_order = random.sample(all_params, len(all_params))
            self._evaluation_cache.clear()
            
            for param_idx, param_info in enumerate(param_order):
                # Лог каждые 16 параметров
                if param_idx % 16 == 0:
                    pct = (param_idx + 1) / len(all_params) * 100
                    self._log(f"  Прогресс: {param_idx+1}/{len(all_params)} ({pct:.0f}%)")
                
                # Настройка одного параметра
                improved, current_all_acc = self._tune_single_parameter(param_info, current_all_acc)
                
                # Обновляем GUI после каждого параметра
                QApplication.processEvents()
                
                if improved:
                    epoch_improved = True
                    epoch_improvements += 1
                    total_improvements += 1
                    self._log(f"  ✓ Улучшение #{epoch_improvements}! "
                            f"Точность: {self.best_all_accuracy:.4f} "
                            f"(+{(self.best_all_accuracy - epoch_best_before)*100:.2f}%)")
            
            epoch_time = time.time() - epoch_start
            
            history['epoch'].append(epoch)
            history['accuracy'].append(self.best_all_accuracy)
            history['improvements'].append(epoch_improved)
            
            self._log(f"\n{'─'*50}")
            self._log(f"ИТОГ ЭПОХИ {epoch}:")
            self._log(f"  Точность: {self.best_all_accuracy:.4f}")
            self._log(f"  Улучшений: {epoch_improvements}")
            self._log(f"  Время: {epoch_time:.1f} сек")
            
            if self.best_all_accuracy > epoch_best_before:
                stall_counter = 0
                self._log(f"  Статус: 🟢 Улучшение")
            else:
                stall_counter += 1
                self._log(f"  Статус: 🔴 Стагнация ({stall_counter}/{self.stall_epochs})")
            
            if self.progress_callback:
                self.progress_callback(epoch, self.max_epochs, self.best_all_accuracy)
        
        total_time = time.time() - start_time
        
        self.fnn.membership_funcs = self.best_mfs
        
        final_acc = self.fnn.evaluate(self.X_all, self.y_all)
        
        self._log(f"\n{'='*60}")
        self._log(f"ОБУЧЕНИЕ ЗАВЕРШЕНО")
        self._log(f"{'='*60}")
        self._log(f"  Начальная точность: {initial_all_acc:.4f} ({initial_all_acc*100:.2f}%)")
        self._log(f"  Финальная точность: {final_acc:.4f} ({final_acc*100:.2f}%)")
        self._log(f"  Общее улучшение: {(final_acc - initial_all_acc)*100:+.2f}%")
        self._log(f"  Всего улучшений: {total_improvements}")
        self._log(f"  Выполнено эпох: {epoch}")
        self._log(f"  Общее время: {total_time:.1f} сек ({total_time/60:.1f} мин)")
        
        if final_acc > initial_all_acc:
            self._log(f"\n  ✓ Модель улучшена!")
        elif abs(final_acc - initial_all_acc) < 1e-6:
            self._log(f"\n  ✓ Точность сохранена")
        else:
            self._log(f"\n  ⚠ Точность снизилась (должна быть восстановлена лучшая)")
        
        return self.best_mfs, final_acc, history