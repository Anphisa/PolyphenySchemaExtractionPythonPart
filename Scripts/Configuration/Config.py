### Global config class (singleton)
from valentine.algorithms import JaccardLevenMatcherColNamesOnly, Coma

class Config():
    def __init__(self):
        # todo: get ip address and port from polypheny and fallback to 127.0.0.1
        self.polypheny_ip_address = "http://127.0.0.1"
        self.polypheny_port = "20598"
        self.jl_colnames_only_threshold = 0.8
        self.matching_threshold = 0.5
        self.sample_size = 5
        self.valentine_algo = JaccardLevenMatcherColNamesOnly(self.jl_colnames_only_threshold)
        self.show_n_best_mappings = 3
        self.random_sample = False

    def valentine_algo_string(self):
        if isinstance(self.valentine_algo, JaccardLevenMatcherColNamesOnly):
            return "Jaccard-Levenshtein (column names only, threshold: " + str(self.jl_colnames_only_threshold) + ")"
        elif isinstance(self.valentine_algo, Coma):
            return "Coma-OPT"

    def set_valentine_algo(self, algo_string):
        # Given an algorithm string, set Valentine algorithm
        if algo_string == "JaccardLevenMatcherColNamesOnly":
            self.valentine_algo = JaccardLevenMatcherColNamesOnly(self.jl_colnames_only_threshold)
        elif algo_string == "Coma":
            self.valentine_algo = Coma()
        return "Set valentine algorithm to " + self.valentine_algo_string()

    def __str__(self):
        config_string = "Polypheny IP address: " + self.polypheny_ip_address + "\r\n" + \
            "Polypheny port: " + str(self.polypheny_port) + "\r\n" + \
            "JL col names only threshold: " + str(self.jl_colnames_only_threshold) + "\r\n" + \
            "Matching threshold: " + str(self.matching_threshold) + "\r\n" + \
            "Sample size: " + str(self.sample_size) + "\r\n" + \
            "Random samples: " + str(self.random_sample) + "\r\n" + \
            "Valentine algorithm: " + self.valentine_algo_string() + "\r\n" + \
            "Show n best mappings: " + str(self.show_n_best_mappings)
        return config_string
