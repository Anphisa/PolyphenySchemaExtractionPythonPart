# Compare primary keys and find primary key <-> foreign key relationships
import logging
import requests
import ast
import Sample
from itertools import chain
from StringSimilarity import StringSimilarity

class PkFkFinder:
    def __init__(self, host, port, tables: dict, sample_size: int):
        """Find primary key <-> foreign key relationships between relational tables.

        :param tables: dictionary with structure {"table name": "primary key name"}
        :param sample_size: number of samples that are drawn from a table for comparison
        """
        self.host = host
        self.port = port
        self.tables = tables
        self.sample_size = sample_size
        self.column_name_similarity = StringSimilarity(min_similarity=0.2)

    def check_fk(self, host, port, table_name: str, pk_name: str, sample_entries: list):
        """Boolean checking if a list is a sub-sample of a pk column.

        :param table_name: table name to check pk column of
        :param pk_name: name of primary key column
        :param sample_entries: a list with samples which may derive of pk column
        :return: True if sample_entries can all be found within pk. False otherwise.
        """
        try:
            flat_sample_entries = list(chain.from_iterable(sample_entries))
            sample_entries_SQL_style = "".join(str(flat_sample_entries))[1:-1]
            query = "SELECT COUNT(" + pk_name + ") FROM " + table_name + " WHERE " + pk_name + " IN (" + sample_entries_SQL_style + ")"
            logging.info("Checking if a list is subsample of a PK column: " + query)
            URL = host + ":" + port + "/query"
            DATA = {'querylanguage': 'SQL',
                    'query': query}
            result = requests.post(url=URL, data=DATA)
            result_content = ast.literal_eval(result.content.decode("utf-8"))
            if result_content[0][0] == len(set(flat_sample_entries)):
                # All sample entries are present in pk column of table_name
                return True
            else:
                # At least one sample entry is not present in pk column of table_name. It can't be the fk.
                return False
        except:
            raise RuntimeError("PkFkFinder.py:check_fk failed")

    def above_similarity_threshold_pk_comparison(self):
        """Return likely pk-fk relationships via a sampling procedure.

        :return: Dictionary of {table 1: {table 2 where pk1 is in: fk2 column}}
        """
        logging.info("Looking for pk-fk relationships above similarity threshold")
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
                        fksThisComparison = [fk for fk in known_fks if fk["foreignTableName"] == compare_table["tableName"]]
                        foreignTableColumnNames = list(chain.from_iterable([ft["foreignTableColumnNames"] for ft in fksThisComparison]))
                        columnNames = list(chain.from_iterable([ft["columnNames"] for ft in fksThisComparison]))
                        if compare_pk_name in foreignTableColumnNames and column_name in columnNames:
                            # This pk-fk relationship is already known! Add it to relationships dict, don't compare columns
                            if table["tableName"] in table_relationships:
                                table_relationships[table["tableName"]].append([{"columnName": column_name,
                                                                                 "foreignTableName": compare_table["tableName"],
                                                                                 "foreignColumnNames": compare_pk_name,
                                                                                 "knownPkFkRelationship": True}])
                            else:
                                table_relationships[table["tableName"]] = [{"columnName": column_name,
                                                                            "foreignTableName": compare_table["tableName"],
                                                                            "foreignColumnNames": compare_pk_name,
                                                                            "knownPkFkRelationship": True}]
                            continue
                    if not self.column_name_similarity.bool_string_distance([column_name, compare_pk_name]):
                        # The column names are not similar enough to be compared
                        # print("Skipping comparison of", table["tableName"], column_name, "with", compare_table["tableName"], compare_pk_name)
                        continue
                    sample = Sample.Sample(self.host, self.port, column_name, table["tableName"], self.sample_size).take_sample()
                    sample_list = ast.literal_eval(sample.content.decode('utf-8'))
                    is_table_pk_fk_for_compare = self.check_fk(self.host, self.port, compare_table["tableName"], compare_pk_name, sample_list)
                    if is_table_pk_fk_for_compare:
                        if table["tableName"] in table_relationships:
                            table_relationships[table["tableName"]].append([{"columnName" : column_name,
                                                                            "foreignTableName": compare_table["tableName"],
                                                                            "foreignColumnNames": compare_pk_name,
                                                                             "knownPkFkRelationship": False}])
                        else:
                            table_relationships[table["tableName"]] = [{"columnName" : column_name,
                                                                        "foreignTableName": compare_table["tableName"],
                                                                        "foreignColumnNames": compare_pk_name,
                                                                        "knownPkFkRelationship": False}]
        logging.info("Finished looking for pk-fk relationships above similarity threshold. Found: " + str(table_relationships))
        return table_relationships

    def validate_pk_fk_relationships(self, table_relationships : dict):
        """Validate sampled pk-fk relationship by exhaustively testing their validity

        :param table_relationships: Table relationships as found by above_similarity_threshold_pk_comparison
        :return: A dict of pk fk relationships validated by an exhaustive check of values
        """
        validated_table_relationships = {}
        for rel in table_relationships:
            table_name = rel
            rels_info = table_relationships[rel]
            for rel_info in rels_info:
                known_pk_fk_relationship = rel_info["knownPkFkRelationship"]
                column_name = rel_info["columnName"]
                other_table_name = rel_info["foreignTableName"]
                other_column_name = rel_info["foreignColumnNames"]
                if known_pk_fk_relationship:
                    # We don't validate known relationships, we believe in them
                    if table_name in validated_table_relationships:
                        validated_table_relationships[table_name].append([{"columnName": column_name,
                                                                           "foreignTableName": other_table_name,
                                                                           "foreignColumnNames": other_column_name,
                                                                           "knownPkFkRelationship": known_pk_fk_relationship}])
                    else:
                        validated_table_relationships[table_name] = [{"columnName": column_name,
                                                                      "foreignTableName": other_table_name,
                                                                      "foreignColumnNames": other_column_name,
                                                                      "knownPkFkRelationship": known_pk_fk_relationship}]
                else:
                    sample = Sample.Sample(self.host, self.port, column_name, table_name, 0).take_sample()
                    sample_list = ast.literal_eval(sample.content.decode('utf-8'))
                    is_table_pk_fk_for_compare = self.check_fk(self.host, self.port, other_table_name, other_column_name, sample_list)
                    if is_table_pk_fk_for_compare:
                        if table_name in validated_table_relationships:
                            validated_table_relationships[table_name].append([{"columnName": column_name,
                                                                               "foreignTableName": other_table_name,
                                                                               "foreignColumnNames": other_column_name,
                                                                               "knownPkFkRelationship": known_pk_fk_relationship}])
                        else:
                            validated_table_relationships[table_name] = [{"columnName": column_name,
                                                                          "foreignTableName": other_table_name,
                                                                          "foreignColumnNames": other_column_name,
                                                                          "knownPkFkRelationship": known_pk_fk_relationship}]
        logging.info("Finished exhaustively validating pk-fk relationships. Found: " + str(validated_table_relationships))
        return validated_table_relationships