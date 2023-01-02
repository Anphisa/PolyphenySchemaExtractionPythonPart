from Helper.FieldRelationship import FieldRelationship
#from Mapping.pyDynaMap.pyDynaMap import pyDynaMap

class Mapping:
    def __init__(self, source_relations):
        # print(source_relations)
        # A list of involved tables and at least their column names
        # assuming form {tableX: [col1, col2, col3], tableY: [col1, col2]}
        self.source_relations = source_relations

    @staticmethod
    def naiveMapping(matches, threshold):
        # given a list of matches from schema matcher and a threshold, return all matches that are above the threshold
        return {k:v for (k, v) in matches.items() if matches[k] >= threshold}

    @staticmethod
    def naiveMappingDecider(involved_tables, matches):
        # given matches (above threshold) and involved_tables (class variable), naively decide the mapping
        # put matches into mappings, with no relationship type yet
        mappings = []
        map_cols = {}
        table_relationships = {}
        for m in matches:
            # example: (('depts', 'deptno'), ('emps', 'deptno'))
            from_table = m[0][0]
            if from_table not in map_cols:
                map_cols[from_table] = []
            from_column = m[0][1]
            to_table = m[1][0]
            if to_table not in map_cols:
                map_cols[to_table] = []
            to_column = m[1][1]

            # table1, table2, field_name1, field_name2, relationship_type=None, relationship_strength=None
            mappings.append(FieldRelationship(from_table, to_table,
                                              from_column, to_column,
                                              relationship_strength=matches[m]))
            map_cols[from_table].append(from_column)
            map_cols[to_table].append(to_column)
            if from_table not in table_relationships:
                table_relationships[from_table] = [to_table]
            else:
                table_relationships[from_table].append(to_table)
            if to_table not in table_relationships:
                table_relationships[to_table] = [from_table]
            else:
                table_relationships[to_table].append(from_table)

        # decide relationship type (join, view union, union)
        for tableX in involved_tables:
            for tableY in involved_tables:
                if tableX == tableY:
                    continue

                # is there no relationship from one of the tables at all?
                if tableX not in table_relationships and tableY not in table_relationships:
                    continue

                # is there any relationship between the two tables? if not, continue
                if tableY not in table_relationships[tableX] and tableX not in table_relationships[tableY]:
                    continue

                map_cols_tableX = map_cols[tableX]
                map_cols_tableY = map_cols[tableY]
                all_cols_tableX = [c[0] for c in involved_tables[tableX]["columns"].columns.values]
                all_cols_tableY = [c[0] for c in involved_tables[tableY]["columns"].columns.values]

                if map_cols_tableX == all_cols_tableX and map_cols_tableY == all_cols_tableY:
                    # union case
                    match_type = "union"
                    for m in mappings:
                        if m.table1 == tableX and m.table2 == tableY:
                            m.relationship_type = match_type
                        elif m.table1 == tableY and m.table2 == tableX:
                            m.relationship_type = match_type

                else:
                    # view union or join case
                    for m in mappings:
                        if m.table1 == tableX and m.table2 == tableY:
                            if m.relationship_type == (None,) or m.relationship_type is None:
                                # have both: one view union case and one join case
                                # todo: generate schema candidates would have to find both alternatives and make schema candidates out of this??
                                # todo: who has to handle this, if there are two possibilities of mapping columns? who has to find them and how to give them to next step in pipeline?
                                m.relationship_type = "view union"
                                #mappings.append(FieldRelationship(m.table1, m.table2,
                                #                                  m.field_name1, m.field_name2,
                                #                                  relationship_strength=m.relationship_strength,
                                #                                  relationship_type="join"))
                        if m.table1 == tableY and m.table2 == tableX:
                            if m.relationship_type == (None,) or m.relationship_type is None:
                                m.relationship_type = "view union"
        return mappings

    def pyDynaMapMapping(self, matches):
        for rel in source_relations:
            col_names = source_relations[rel].keys()
            if len(col_names) > len(set(col_names)):
                raise RuntimeError("pyDynaMap doesn't accept duplicate column names. "
                                   "Duplicate column names found in source_relations: ", source_relations)
        pass