# Given two columns (or fields) from tables (or relations), compare them for similarity
import abc

class ColumnCompare(abc.ABC):
    def __init__(self, column1 : str, table1 : str, column2 : str, table2 : str):
        self.column1 = column1
        self.column2 = column2
        self.table1 = table1
        self.table2 = table2

    @property
    def identical(self):
        # Return true if columns are identical
        if self.column1 == self.column2 and self.table1 == self.table2:
            return True
        return False

    @abc.abstractmethod
    def applicable(self) -> bool:
        # Return true if strategy is applicable to data format, etc (?)
        pass

    @abc.abstractmethod
    def estimate_costs(self, sample1, sample2):
        # Estimate cost of executing this strategy given a sample of the two columns
        pass

    @abc.abstractmethod
    def compare(self):
        # Execute comparison (TODO: Split by sample/exhaustive comparison?)
        pass