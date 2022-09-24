# Given two columns (or fields) from tables (or relations), compare them for similarity
# Strategy pattern (?)

class ColumnCompare:
    def __init__(self, column1, column2, compare_strategy = "CompareStrategy"):
        self._compare_strategy_ = compare_strategy
        self.column1 = column1
        self.column2 = column2

    @property
    def compare_strategy(self) -> "CompareStrategy":
        return self._compare_strategy_

    @compare_strategy.setter
    def compare_strategy(self, compare_strategy: "CompareStrategy") -> None:
        self._compare_strategy_ = compare_strategy

class CompareStrategy(abc.ABC):
    @abc.abstractmethod
    def get_act_price(self, raw_price: float) -> float:
        raise NotImplementedError

class StringStrategy(CompareStrategy):
    def get_act_price(self, raw_price: float) -> float:
        return raw_price

def main() -> None:
    string_strategy = StringStrategy()

    test = ColumnCompare(column1, column2, string_strategy)

    # Something like this... Compare the columns according to strategy comparison
    # Add the comparison results to some list or something?
    # ... TO think about
    # https://en.wikipedia.org/wiki/Strategy_pattern Python code
    test.compare()

if __name__ == "__main__":
    main()