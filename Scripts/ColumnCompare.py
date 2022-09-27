# Given two columns (or fields) from tables (or relations), compare them for similarity
import abc

class ColumnCompare(abc.ABC):
    def __init__(self, column1 : str, table1 : str, column2 : str, table2 : str):
        self.column1 = column1
        self.column2 = column2
        self.table1 = table1
        self.table2 = table2

    @abc.abstractmethod
    def applicable(self) -> bool:
        # Return true if strategy is applicable to data format, etc (?)
        pass

    @abc.abstractmethod
    def estimate_costs(self, sample):
        # Estimate cost of executing this strategy given a sample of the two columns
        pass

    @abc.abstractmethod
    def compare(self):
        # Execute comparison (TODO: Split by sample/exhaustive comparison?)
        pass


class StringCompare(ColumnCompare):
    def applicable(self):
        return True

    def compare(self) -> float:
        return 1.0

    def estimate_costs(self, sample):
        return 1.0


def main() -> None:
    test = StringCompare(1, 2, 3, 4)
    print(test.compare())

if __name__ == "__main__":
    main()