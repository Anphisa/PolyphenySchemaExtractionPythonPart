### Global config class (singleton)
from valentine.algorithms import JaccardLevenMatcherColNamesOnly, Coma, JaccardLevenMatcher, Cupid, SimilarityFlooding, DistributionBased

class Config():
    def __init__(self):
        # todo: get ip address and port from polypheny and fallback to 127.0.0.1
        self.polypheny_ip_address = "http://127.0.0.1"
        self.polypheny_port = "20598"
        self.jl_threshold = 0.8
        self.matching_threshold = 0.5
        self.sample_size = 5
        #self.valentine_algo = JaccardLevenMatcherColNamesOnly(self.jl_threshold)
        #self.str_valentine_algo = "JaccardLevenMatcherColNamesOnly"
        self.str_valentine_algo = "automatic"
        self.valentine_algo = None
        self.show_n_best_mappings = 3
        self.random_sample = False

    def valentine_algo_string(self):
        if not self.valentine_algo:
            return "automatic"
        if isinstance(self.valentine_algo, JaccardLevenMatcherColNamesOnly):
            return "Jaccard-Levenshtein (column names only, threshold: " + str(self.jl_threshold) + ")"
        elif isinstance(self.valentine_algo, Coma):
            return "Coma"
        elif isinstance(self.valentine_algo, Cupid):
            return "Cupid"
        elif isinstance(self.valentine_algo, DistributionBased):
            return "DistributionBased"
        elif isinstance(self.valentine_algo, JaccardLevenMatcher):
            return "JaccardLevenMatcher"
        elif isinstance(self.valentine_algo, SimilarityFlooding):
            return "SimilarityFlooding"

    def set_valentine_algo(self, algo_string):
        # Given an algorithm string, set Valentine algorithm
        if algo_string == "JaccardLevenMatcherColNamesOnly":
            self.str_valentine_algo = "JaccardLevenMatcherColNamesOnly"
            self.valentine_algo = JaccardLevenMatcherColNamesOnly(self.jl_threshold)
        elif algo_string == "Coma":
            self.str_valentine_algo = "Coma"
            self.valentine_algo = Coma()
        elif algo_string == "Coma_OPT":
            self.str_valentine_algo = "Coma"
            self.valentine_algo = Coma()
        elif algo_string == "COMA_OPT_INST":
            self.str_valentine_algo = "Coma_OPT_INST"
            self.valentine_algo = Coma("COMA_OPT_INST")
        elif algo_string == "Cupid":
            self.str_valentine_algo = "Cupid"
            self.valentine_algo = Cupid()
        elif algo_string == "DistributionBased":
            self.str_valentine_algo = "DistributionBased"
            self.valentine_algo = DistributionBased()
        elif algo_string == "JaccardLevenMatcher":
            self.str_valentine_algo = "JaccardLevenMatcher"
            self.valentine_algo = JaccardLevenMatcher(self.jl_threshold)
        elif algo_string == "SimilarityFlooding":
            self.str_valentine_algo = "SimilarityFlooding"
            self.valentine_algo = SimilarityFlooding()
        elif algo_string == "automatic":
            self.str_valentine_algo = "automatic"
            self.valentine_algo = None
        return "Set valentine algorithm to " + self.valentine_algo_string()

    def set_valentine_algo_parameters(self, parameter_strings):
        parameter_strings = parameter_strings.split(",")
        params = [p.split(":")[0] for p in parameter_strings]
        values = [p.split(":")[1] for p in parameter_strings]
        for i, p in enumerate(params):
            v = values[i]
            if p == "max_n":
                self.valentine_algo = Coma(max_n=v)
            if p == "strategy":
                self.valentine_algo = Coma(strategy=v)
            if p == "w_struct":
                self.valentine_algo = Cupid(w_struct=v)
            if p == "leaf_w_struct":
                self.valentine_algo = Cupid(leaf_w_struct=v)
            if p == "th_accept":
                self.valentine_algo = Cupid(th_accept=v)
            if p == "threshold1":
                self.valentine_algo = DistributionBased(threshold1=v)
            if p == "threshold2":
                self.valentine_algo = DistributionBased(threshold2=v)
            if p == "threshold_leven":
                self.jl_threshold = v
            if p == "coeff_policy":
                self.valentine_algo = SimilarityFlooding(coeff_policy=v)
            if p == "formula":
                self.valentine_algo = SimilarityFlooding(formula=v)

    def __str__(self):
        config_string = "Polypheny IP address: " + self.polypheny_ip_address + "\r\n" + \
            "Polypheny port: " + str(self.polypheny_port) + "\r\n" + \
            "JL col names only threshold: " + str(self.jl_threshold) + "\r\n" + \
            "Matching threshold: " + str(self.matching_threshold) + "\r\n" + \
                        "Sample size: " + str(self.sample_size) + "\r\n" + \
                        "Random samples: " + str(self.random_sample) + "\r\n" + \
                        "Valentine algorithm: " + self.valentine_algo_string() + "\r\n" + \
                        "Show n best mappings: " + str(self.show_n_best_mappings)
        return config_string
