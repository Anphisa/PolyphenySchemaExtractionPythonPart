### Matching wrappers

class Matching():
    @staticmethod
    def chooseValentineAlgorithm(dfs):
        # Given pandas dataframes (probably filled with samples from Polypheny), choose the most suitable Valentine
        # algorithm for matching.
        # This is derived from tests on dataframe combinations and seeing which features lead to which Valentine algorithm
        # performing best.
        # todo: put logic here ;)
        return "JaccardLevenMatcherColNamesOnly"