# Compare primary keys and find primary key <-> foreign key relationships
from itertools import chain
from StringSimilarity import StringSimilarity

class PkFkFinder:
    def __init__(self, cursor, tables: dict, sample_size: int):
        """Find primary key <-> foreign key relationships between relational tables.

        :param cursor: cursor with connection to Polypheny
        :param tables: dictionary with structure {"table name": "primary key name"}
        :param sample_size: number of samples that are drawn from a table for comparison
        """
        self.cursor = cursor
        self.tables = tables
        self.sample_size = sample_size
        self.column_name_similarity = StringSimilarity(min_similarity=0.2)


    def sample_values(self, cursor, table_name: str, column_name: str, sample_size: int):
        """Take a sample of values in a column."""
        cursor.execute("SELECT "+ column_name + " FROM " + table_name)
        result = cursor.fetchmany(sample_size)
        return result

    def check_fk(self, cursor, table_name: str, pk_name: str, sample_entries: list):
        """Boolean checking if a list is a sub-sample of a pk column.

        :param cursor: cursor with connection to Polypheny
        :param table_name: table name to check pk column of
        :param pk_name: name of primary key column
        :param sample_entries: a list with samples which may derive of pk column
        :return: True if sample_entries can all be found within pk. False otherwise.
        """
        try:
            flat_sample_entries = list(chain.from_iterable(sample_entries))
            sample_entries_SQL_style = "".join(str(flat_sample_entries))[1:-1]
            query = "SELECT COUNT(" + pk_name + ") FROM " + table_name + " WHERE " + pk_name + " IN (" + sample_entries_SQL_style + ")"
            cursor.execute(query)
            result = cursor.fetchall()
            if result[0][0] == len(set(flat_sample_entries)):
                # All sample entries are present in pk column of table_name
                return True
            else:
                # At least one sample entry is not present in pk column of table_name. It can't be the fk.
                return False
        except:
            raise RuntimeError("PkFkFinder.py:check_fk failed")

    def above_similarity_threshold_pk_comparison(self):
        """

        :return: Dictionary of {table 1: {table 2 where pk1 is in: fk2 column}}
        """
        table_relationships = {}
        # TODO: table_relationships should be a class variable
        # TODO: Write a method to add pk-fks to table_relationships
        for table in self.tables:
            for column_name in table["columnNames"]:
                for compare_table in self.tables:
                    compare_pk_name = compare_table["primaryKey"]
                    if len(compare_pk_name) > 1:
                        raise RuntimeError("PkFkFinder.py: Not implemented for composite primary keys. Failed for primary keys",
                                           compare_pk_name, "in", compare_table["tableName"])
                    else:
                        # There's only one primary key, so we unlist it
                        compare_pk_name = compare_pk_name[0]
                    if table == compare_table:
                        continue
                    known_fks = table["foreignKeys"]
                    if known_fks:
                        # TODO: Time this?
                        fksThisComparison = [fk for fk in known_fks if fk["foreignTableName"] == compare_table["tableName"]]
                        foreignTableColumnNames = list(chain.from_iterable([ft["foreignTableColumnNames"] for ft in fksThisComparison]))
                        columnNames = list(chain.from_iterable([ft["columnNames"] for ft in fksThisComparison]))
                        if compare_pk_name in foreignTableColumnNames and column_name in columnNames:
                            # This pk-fk relationship is already known! Add it to relationships dict, don't compare columns
                            if table["tableName"] in table_relationships:
                                table_relationships[table["tableName"]].append([{"columnName": column_name,
                                                                                 "foreignTableName": compare_table["tableName"],
                                                                                 "foreignColumnNames": compare_pk_name}])
                            else:
                                table_relationships[table["tableName"]] = [{"columName": column_name,
                                                                            "foreignTableName": compare_table["tableName"],
                                                                            "foreignColumnNames": compare_pk_name}]
                            continue
                    if not self.column_name_similarity.bool_string_distance([column_name, compare_pk_name]):
                        # The column names are not similar enough to be compared
                        # print("Skipping comparison of", table["tableName"], column_name, "with", compare_table["tableName"], compare_pk_name)
                        continue
                    sample = self.sample_values(self.cursor, table["tableName"], column_name, self.sample_size)
                    is_table_pk_fk_for_compare = self.check_fk(self.cursor, compare_table["tableName"], compare_pk_name, sample)
                    if is_table_pk_fk_for_compare:
                        if table["tableName"] in table_relationships:
                            table_relationships[table["tableName"]].append([{"columName" : column_name,
                                                                            "foreignTableName": compare_table["tableName"],
                                                                            "foreignColumnNames": compare_pk_name}])
                        else:
                            table_relationships[table["tableName"]] = [{"columName" : column_name,
                                                                        "foreignTableName": compare_table["tableName"],
                                                                        "foreignColumnNames": compare_pk_name}]
        return table_relationships