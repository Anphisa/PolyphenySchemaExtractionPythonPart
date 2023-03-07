### Matching wrappers
from fuzzywuzzy import fuzz

class Matching():
    @staticmethod
    def chooseValentineAlgorithm(dfs):
        # Given pandas dataframes (probably filled with samples from Polypheny), choose the most suitable Valentine
        # algorithm for matching.
        # This is derived from tests on dataframe combinations and seeing which features lead to which Valentine algorithm
        # performing best. (see master's thesis)

        # Collect statistics from dfs. We assume that there are two, because this is the dataset that I trained the tree on.
        df_names = list(dfs.keys())
        df1 = dfs[df_names[0]]
        df2 = dfs[df_names[1]]

        # avg_fuzzy_col_match: For every column in a df: Does it have a direct correspondence in other df?
        thresh = 80
        d1_fuzzy = {}
        for col1 in df1:
            d1_fuzzy[col1] = False
            for col2 in df2:
                if fuzz.ratio(col1, col2) >= thresh:
                    d1_fuzzy[col1] = True
                    break
        d2_fuzzy = {}
        for col2 in df2:
            d2_fuzzy[col2] = False
            for col1 in df1:
                if fuzz.ratio(col1, col2) >= thresh:
                    d2_fuzzy[col2] = True
                    break
        fuzzy_col_matches = [sum(d1_fuzzy.values()), sum(d2_fuzzy.values())]
        avg_fuzzy_col_match = sum(fuzzy_col_matches) / len(fuzzy_col_matches)

        n_col_max = max(len(df1), len(df2))
        n_col_diff = abs(len(df1) - len(df2))

        matcher_type = "schema_based"
        if avg_fuzzy_col_match <= 1.75:
            if n_col_max <= 13:
                if not n_col_diff > 3:
                    matcher_type = "instance_based"

        # Heuristic based on appropriate schema matcher class, as described in thesis
        if matcher_type == "schema_based":
            return {"algorithm_string": "JaccardLevenMatcherColNamesOnly",
                    "algorithm_arguments" : "",
                    "explanation": "Schema-based algorithm selected (default: JaccardLevenMatcherColNamesOnly)."}
        else:
            return {"algorithm_string": "COMA_OPT_INST",
                    "algorithm_arguments" : "",
                    "explanation": "Instance-based algorithm selected (avg_fuzzy_col_match <= 1.75 and n_col_max <= 13 and not n_col_diff > 3.): COMA_OPT_INST."}