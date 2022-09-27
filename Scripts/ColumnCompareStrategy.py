# This class knows all column comparison strategies and makes decisions about which comparisons to make (given two columns)
import ColumnCompare
import Sample

class ColumnCompareStrategy():
    def __init__(self, host, port, column1: str, table1: str, column2: str, table2: str, sample_size: int):
        self.column1 = column1
        self.column2 = column2
        self.table1 = table1
        self.table2 = table2

        # all known column compare strategies
        self.strategies = [ColumnCompare.StringCompare]

        # Samples from both columns
        self.sample1 = Sample(host, port, column1, table1, sample_size)
        self.sample2 = Sample(host, port, column2, table2, sample_size)

    # for each known strategy, estimate costs
    # (maybe with a global sample?)
    #