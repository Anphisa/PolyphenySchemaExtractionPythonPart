### Global config class (singleton)
from valentine.algorithms import JaccardLevenMatcherColNamesOnly, Coma

class Config():
    def __init__(self):
        self.jl_colnames_only_threshold = 0.8
        self.sample_size = 5
        self.valentine_algo = JaccardLevenMatcherColNamesOnly(self.jl_colnames_only_threshold)
        self.show_n_best_mappings = 3