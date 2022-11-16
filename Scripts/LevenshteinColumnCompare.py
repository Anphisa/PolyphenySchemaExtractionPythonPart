from ColumnCompare import ColumnCompare

class LevenshteinColumnCompare(ColumnCompare):
    def applicable(self):
        # String distance methods are always applicable
        return True

    def levenshtein_distance(self, column1: list, column2: list):
        # Edit distance (number of inserts, deletions, changes)
        distances = []
        normalized_distances = []
        for val1 in column1:
            for val2 in column2:
                levenshtein_distance = levenshtein.distance(string(val1), string(val2))
                distances.append(levenshtein_distance)
                normalized_distances.append(levenshtein_distance / max(len(s) for s in [val1, val2]))
        return distances, normalized_distances

    def levenshtein_metric(self, column1: list, column2: list):
        # 0 if equal, 1 if maximally unequal.
        levenshtein_distances, levenshtein_normalized_distances = self.levenshtein_distance(column1, column2)
        return sum(levenshtein_normalized_distances) / len(levenshtein_normalized_distances)

    def compare(self) -> float:
        return self.levenshtein_metric(self.sample1, self.sample2)

    @staticmethod
    def estimate_costs(sample1, sample2):
        # Should be roughly n x m
        return len(sample1) * len(sample2)