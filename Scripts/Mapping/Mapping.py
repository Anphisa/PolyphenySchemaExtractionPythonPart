from Helper.FieldRelationship import FieldRelationship

class Mapping:
    def __init__(self, involved_tables):
        print(involved_tables)
        # A list of involved tables and at least their column names
        # todo: perspectively, these invovled tables should be table objects
        if len(involved_tables) > 2:
            raise NotImplementedError("currently only implemented for two tables!")
        # assuming form {tableX: [col1, col2, col3], tableY: [col1, col2]}
        self.involved_tables = involved_tables

    @staticmethod
    def naiveMapping(matches, threshold):
        # given a list of matches from schema matcher and a threshold, return all matches that are above the threshold
        return {k:v for (k, v) in matches.items() if matches[k] >= threshold}

    @staticmethod
    def naiveMappingDecider(involved_tables, matches):
        # given matches (above threshold) and involved_tables (class variable), naively decide the mapping
        # todo: think about case with > 2 involved tables!
        mappings = []
        for tableX in involved_tables:
            for tableY in involved_tables:
                if tableX == tableY:
                    continue
                matchCols = {}
                for m in matches:
                    # (('depts', 'deptno'), ('emps', 'deptno'))
                    from_table = m[0][0]
                    if from_table not in matchCols:
                        matchCols[from_table] = []
                    from_column = m[0][1]
                    to_table = m[1][0]
                    if to_table not in matchCols:
                        matchCols[to_table] = []
                    to_column = m[1][1]

                    #table1, table2, field_name1, field_name2, relationship_type=None, relationship_strength=None
                    mappings.append(FieldRelationship(from_table, to_table,
                                                      from_column, to_column,
                                                      relationship_strength=matches[m]))
                    matchCols[from_table].append(from_column)
                    matchCols[to_table].append(to_column)
                # find relationship type
                if len(matchCols[tableX]) == len(involved_tables[tableX]) and len(matchCols[tableY]) == len(involved_tables[tableY]):
                    # union case
                    match_type = "union"
                    for m in mappings:
                        m.relationship_type = match_type
                else:
                    # view union or join case
                    for m in mappings:
                        if m.relationship_type == (None,) or m.relationship_type is None:
                            m.relationship_type = "view union"
                            mappings.append(FieldRelationship(m.table1, m.table2,
                                                              m.field_name1, m.field_name2,
                                                              relationship_strength=m.relationship_strength,
                                                              relationship_type="join"))
        return mappings