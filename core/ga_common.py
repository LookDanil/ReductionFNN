"""Общие классы для ГА: хромосомы"""
from typing import List, Tuple, Dict
import itertools


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