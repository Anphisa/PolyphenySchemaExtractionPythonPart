# This class knows all column comparison strategies and makes decisions about which comparisons to make (given two columns)
import LevenshteinColumnCompare
from Sample import Sample
from collections import Counter

class ColumnCompareStrategy():
    def __init__(self, host, port, column1: str, table1: str, column2: str, table2: str, sample_size: int):
        self.column1 = column1
        self.column2 = column2
        self.table1 = table1
        self.table2 = table2

        # Samples from both columns (currently un-random samples)
        sample1 = Sample(host, port, column1, table1, sample_size).take_set_sample()
        self.sample1 = sample1.extract_sample()
        sample2 = Sample(host, port, column2, table2, sample_size).take_set_sample()
        self.sample2 = sample2.extract_sample()

        # all known column compare strategies
        self.strategies = [LevenshteinColumnCompare.LevenshteinColumnCompare]
        # estimate costs of all known column compare strategies
        self.strategy_costs = {}
        for strategy in self.strategies:
            est_costs = strategy.estimate_costs(self.sample1, self.sample2)
            self.strategy_costs[strategy] = {"est_costs": est_costs}

    def top_n_strategies(self, n):
        """Return top n strategies by cost

        :param n: how many strategies to return
        :return: list of top n strategy methods
        """
        top_n_strategies = sorted(self.strategy_costs.items(), key=lambda x: x[1]['est_costs'], reverse=False)[:n]
        return top_n_strategies

    def below_threshold_strategies(self, cost_threshold):
        """Return all strategies that have a cost below a certain threshold

        :param cost_threshold: Cost below which a strategy should be returned
        :return: list of strategy methods with cost below threshold
        """
        below_threshold_strategies = []
        for strategy in self.strategy_costs:
            if self.strategy_costs[strategy] <= cost_threshold:
                below_threshold_strategies.append(strategy)
        return below_threshold_strategies


test = ColumnCompareStrategy("http://127.0.0.1", "20598", "deptno", "depts", "deptno", "emps", 10)
print(test.top_n_strategies(5))