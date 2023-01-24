### Global config class (singleton)
from valentine.algorithms import JaccardLevenMatcherColNamesOnly, Coma

class Config():
    def __init__(self):
        self.polypheny_ip_address = "http://127.0.0.1"
        self.polypheny_port = "20598"
        self.jl_colnames_only_threshold = 0.8
        self.matching_threshold = 0.5
        self.sample_size = 5
        self.valentine_algo = JaccardLevenMatcherColNamesOnly(self.jl_colnames_only_threshold)
        self.show_n_best_mappings = 3

    def valentine_algo_string(self):
        if isinstance(self.valentine_algo, JaccardLevenMatcherColNamesOnly):
            return "Jaccard-Levenshtein (column names only, threshold: " + str(self.jl_colnames_only_threshold) + ")"