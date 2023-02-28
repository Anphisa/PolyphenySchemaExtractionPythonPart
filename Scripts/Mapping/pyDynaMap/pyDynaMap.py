# Re-implementing the DynaMap algorithm as described in "Schema mapping generation in the wild" (https://doi.org/10.1016/j.is.2021.1019049)
import pandas as pd
import networkx as nx
import math
import copy
import collections
import statistics
from Helper.FieldRelationship import FieldRelationship

class pyDynaMap():
    def __init__(self, source_relations: dict, matches: dict):
        for rel in source_relations:
            col_names = source_relations[rel].keys()
            if len(col_names) > len(set(col_names)):
                raise RuntimeError("pyDynaMap doesn't accept duplicate column names. "
                                   "Duplicate column names found in source_relations: ", source_relations)
        self.source_relations = source_relations
        self.matches = matches
        # In t_rel, we save the column names of the target.
        # We also save the aliases that this column name has in different source tables
        self.t_rel = self.target_relation_from_matches()
        self.sub_solution = {}
        self.pd = {}
        self.fitness_value_by_source = {}
        self.highest_fitness_value = {}
        self.mapping_sources = {}
        self.mapping_path = {}
        for source in self.source_relations:
            self.mapping_path[source] = source
        # loss information for display
        self.instance_loss = {}
        self.field_loss = ""
        # renamed columns for re-tracing steps
        self.renamed_columns = {}
        # explanation tracking for explainability of results
        self.explanations = {}

    def field_loss(self):
        # If a source relation has a field that is not represented in t_rel, it is lost
        for source in self.source_relations:
            source_field_repr = {}
            for field in source:
                field_repr = False
                for t_col in self.t_rel:
                    if field in self.t_rel[t_col]:
                        field_repr = True
                source_field_repr[field] = field_repr
            if any(source_field_repr.values()):
                lost_fields = [f for f in source_field_repr.keys() if source_field_repr[f]]
                self.field_loss().append("Source: ", source, "loses fields: ", ",".join(lost_fields))

    def is_match_exact(self, from_table, from_column, to_table, to_column):
        # Allow non-equal column names and still achieve a mapping
        for m in self.matches:
            m_from_table = m[0][0]
            m_to_table = m[1][0]
            m_from_column = m[0][1]
            m_to_column = m[1][1]
            if from_table == m_from_table and \
                    to_table == m_to_table and \
                    from_column == m_to_column and \
                    to_column == m_to_column:
                return True
        return False

    def matching_keys(self, from_table, from_column, to_table):
        # Return all columns that have a match with this column
        to_columns = []
        for m in self.matches:
            m_from_table = m[0][0]
            m_to_table = m[1][0]
            m_from_column = m[0][1]
            m_to_column = m[1][1]
            if from_table == m_from_table and \
                    to_table == m_to_table and \
                    to_column == m_to_column:
                to_columns.append(m_to_column)
        return to_columns

    def choose_column_name(self, col_name_candidates):
        # given a list of column names, choose the one that we will use in target relation
        return min(col_name_candidates, key=len)

    def target_relation_from_matches(self):
        # we are given column names from matches
        # example: {(('depts', 'deptno'), ('emps', 'deptno')): 1, (('depts', 'name'), ('emps', 'name')): 1, (('emp', 'employeeno'), ('work', 'employeeno')): 1}
        # and we use these to build up a graph of connected columns, so we use each of these connected component only for one target column name
        tuplist = [("}".join(x[0]), "}".join(x[1])) for x in self.matches.keys()]
        # example: [{'depts_deptno', 'emps_deptno'}, {'depts_name', 'emps_name'}, {'emp_employeeno', 'work_employeeno'}]
        G = nx.Graph()
        G.add_edges_from(tuplist)

        target_columns = {}
        for connected_fields in list(nx.connected_components(G)):
            field_names = [f.split("}")[1] for f in connected_fields]
            # Choose the column name we want to keep
            col_name = self.choose_column_name(field_names)

            # save col_name aliases from source tables
            for table_field in connected_fields:
                table = table_field.split("}")[0]
                field = table_field.split("}")[1]
                if col_name in target_columns:
                    target_columns[col_name][table] = field
                else:
                    target_columns[col_name] = {table: field}
        return target_columns

    def inclusion_dependencies(self, atts_s: set, atts_r: set):
        # is atts_r completely included in atts_s?
        # then there is a total inclusion dependency from atts_r to atts_s
        if atts_r.intersection(atts_s) == atts_r:
            return 1.
        # is atts_r not included in atts_s at all?
        # then there is no inclusion dependency from atts_r to atts_s
        if atts_r.intersection(att_s) == set():
            return 0.
        # is atts_r partially included in atts_s?
        # then there is a partial inclusion dependency from atts_r to atts_s
        else:
            return len(atts_r.intersection(atts_s))/len(atts_r)

    def get_match_strength(self, from_table, to_table, from_column, to_column):
        # matches = {(('df1', 'B'), ('df2', 'B')): 1,
                    # (('df1', 'C'), ('df2', 'C')): 1,
                    # (('df1', 'A'), ('df3', 'A')): 1,
                    # (('df1', 'A'), ('df3', 'C')): 1,
                    # (('df2', 'C'), ('df3', 'C')): 1}
        for m in self.matches:
            m_from_table = m[0][0]
            m_to_table = m[1][0]
            m_from_column = m[0][1]
            m_to_column = m[1][1]
            if self.is_match_exact(from_table, from_column, to_table, to_column):
                return self.matches[m]

    def compute_profile_data(self, map):
        # todo: get this from polypheny statistics and/or metanome library
        profile_data = {}
        # we need cardinalities of relations (--> pp statistics)
        # we need the number of distinct values for each attribute (--> pp statistics)
        # we need the number of nulls for each attribute (--> pp statistics)
        # we need candidate keys for every relation (--> metanome?)
        for relation in source_relations:
            profile_data[relation]["candidate_keys"] = 1
        # we need inclusion dependencies for all attribute combinations (--> metanome)
        # todo: we need to propagate inclusion dependencies in case of new mappings
        # we need candidate matches (--> valentine)
        return profile_data

    def generate_mappings(self, i):
        # Algorithm 1: Mapping generation — the recursive method of dynamic programming.
        if i == 0:
            return []
        if i in self.sub_solution:
            return self.sub_solution[i]
        if i == 1:
            self.sub_solution[i] = self.source_relations
            return self.sub_solution[i]
        else:
            iteration_maps = []
            for j in range(1, math.ceil(i/2) + 1):
                b1 = self.generate_mappings(j)
                b2 = self.generate_mappings(i - j)
                new_maps = self.merge_mappings(b1, b2)
                iteration_maps.append(new_maps)
            # flatten list of iteration_maps
            flat_iteration_maps = {k: v for d in iteration_maps for k, v in d.items()}
            self.sub_solution[i] = flat_iteration_maps
            return self.sub_solution[i]
        # todo: later include pruning search space as discussed in section 4.7.3. Pruning techniques

    def merge_mappings(self, batch1, batch2):
        # Algorithm 2: Merge pairwise the mappings from 2 sets of mappings.
        new_maps = {}
        batch1 = copy.deepcopy(batch1)
        batch2 = copy.deepcopy(batch2)
        for map_i_name in batch1:
            for map_j_name in batch2:
                # If map_i == map_j, do not return union with itself (endless recursion, unintended endless self-unions)
                if map_i_name == map_j_name:
                    continue
                # If a relation is already part of a mapping, don't combine it in again
                if self.mapping_subsumed(map_i_name, map_j_name):
                    continue
                map_i = copy.deepcopy(batch1[map_i_name])
                map_j = copy.deepcopy(batch2[map_j_name])
                # put initial sources in highest fitness records as well, so we can return if not merging them at all is the best solution
                if map_i_name not in self.highest_fitness_value:
                    self.highest_fitness_value[map_i_name] = self.fitness(map_i, map_i_name)
                if map_j_name not in self.highest_fitness_value:
                    self.highest_fitness_value[map_j_name] = self.fitness(map_j, map_j_name)
                # continue merging of mappings
                operator_explanation = self.choose_operator(map_i, map_j, map_i_name, map_j_name)
                operator = operator_explanation["operator"]
                explanation = operator_explanation["explanation"]
                if operator:
                    try:
                        new_map = self.new_mapping(*operator)
                    except Exception as e:
                        raise RuntimeError("Mapping ", map_i_name, "and", map_j_name, "failed.", e)
                    # todo: compute metadata and use it for something?
                    #md = self.compute_metadata(new_map)
                    # mapping was accepted
                    # metadata needs to be recorded given initial sources (we need to find mapping with highest fitness given same initial sources)
                    max_fitness_for_sources = self.get_max_fitness([map_i_name, map_j_name])
                    map_name = map_i_name + "_" + map_j_name
                    new_maps[map_name] = new_map
                    self.mapping_sources[map_name] = [map_i_name, map_j_name]
                    if operator[0] == "left join":
                        self.mapping_path[map_name] = [operator[0], map_i_name, map_j_name, operator[-2]]
                    else:
                        self.mapping_path[map_name] = [operator[0], map_i_name, map_j_name, operator[-1]]
                    new_map_fitness = self.fitness(new_map, map_name)
                    if new_map_fitness > max_fitness_for_sources:
                        # mapping was accepted
                        self.highest_fitness_value[map_name] = new_map_fitness
                        self.explanations[map_name] = explanation
                        for t_col in self.t_rel.keys():
                            if map_i_name in self.t_rel[t_col] and map_j_name in self.t_rel[t_col]:
                                if operator[0] == "left join":
                                    # column name comes from map1 (we take df1_aliases in join)
                                    # todo: put in explanation
                                    self.t_rel[t_col][map_name] = self.t_rel[t_col][map_i_name]
                                elif operator[0] == "union":
                                    # We replace target names with t_col names
                                    # todo: put in explanation
                                    self.t_rel[t_col][map_name] = t_col
                                else:
                                    self.t_rel[t_col][map_name] = self.choose_column_name([self.t_rel[t_col][map_i_name],
                                                                                           self.t_rel[t_col][map_j_name]])
                            elif map_i_name in self.t_rel[t_col]:
                                self.t_rel[t_col][map_name] = self.t_rel[t_col][map_i_name]
                            elif map_j_name in self.t_rel[t_col]:
                                self.t_rel[t_col][map_name] = self.t_rel[t_col][map_j_name]
                    else:
                        # mapping was not accepted
                        del new_maps[map_name]
                        del self.mapping_sources[map_name]
                        del self.mapping_path[map_name]
                    # if self.is_fittest(new_map, [map_i_name, map_j_name]):
                    #     map_name = map_i_name + "_" + map_j_name
                    #     new_maps[map_name] = new_map
                    #     self.mapping_sources[map_name] = [map_i_name, map_j_name]
                    #     self.mapping_path[map_name] = [operator[0], map_i_name, map_j_name, operator[-1]]
                    #     self.highest_fitness_value[map_name] = self.fitness(new_map, map_name)
                    #     for t_col in self.t_rel.keys():
                    #         if map_i_name in self.t_rel[t_col] and map_j_name in self.t_rel[t_col]:
                    #             self.t_rel[t_col][map_name] = self.choose_column_name([self.t_rel[t_col][map_i_name],
                    #                                                                    self.t_rel[t_col][map_j_name]])
                    #         elif map_i_name in self.t_rel[t_col]:
                    #             self.t_rel[t_col][map_name] = self.t_rel[t_col][map_i_name]
                    #         elif map_j_name in self.t_rel[t_col]:
                    #             self.t_rel[t_col][map_name] = self.t_rel[t_col][map_j_name]
                    #         # else:
                    #         #     raise RuntimeError("Both parents of map ", map_name, " are not present in t_rel: ", self.t_rel)
        return new_maps

    def find_parentage(self, ancestor_names):
        # find all ancestor map names, so parents, grandparents, etc.
        # this should help combat the issue of endlessly re-combining in the same dfs over and over again
        i = 0
        while i < len(ancestor_names):
            ancestor = ancestor_names[i]
            if ancestor in self.mapping_sources:
                sources = self.mapping_sources[ancestor]
                for s in sources:
                    ancestor_names.append(s)
            i += 1
        return ancestor_names

    def mapping_subsumed(self, map1_name, map2_name):
        # Don't combine immediate ancestors back into a mapping.
        # If we wanted to forbid all re-combinations, we would have to make a set operations in the if-condition to
        # check that there are no common ancestors at all.
        map1_ancestors = self.find_parentage([map1_name])
        map2_ancestors = self.find_parentage([map2_name])
        if map1_name in map2_ancestors or map2_name in map1_ancestors:
            return True
        else:
            return False

    # def compute_metadata(self, new_map):
    #     # compute fitness data for new mapping
    #     fitness = self.fitness(new_map)
    #     # compute profile data for new mapping
    #     profile_data = self.compute_profile_data(new_map)
    #     # if operator == 'join':
    #     #     if self.lossy_merge_conditions_satisfied(map1, map2):
    #     #         metadata = self.compute_metadata_lossy(map1, map2)
    #     #     else:
    #     #         metadata = self.compute_metadata_lossless(map1, map2)
    #     # elif operator == 'full outer join':
    #     #     metadata = self.compute_metadata_lossless(map1, map2)
    #     # elif operator == 'union':
    #     #     metadata = self.compute_metadata_lossless(map1, map2)
    #     # return metadata

    def lossy_merge_conditions_satisfied(self, map1, map2):
        # When one of the relations may lose attribute values.
        return any([key not in map2.keys() for key in map1.keys()])

    def compute_metadata_lossless(self, map1, map2):
        pass

    def compute_metadata_lossy(self, map1, map2):
        pass

    def aliases_for_t_rel_columns(self, map_name, t_rel_column_names: list):
        # Given a map and column names as they are in t_rel, find local aliases in map
        map_col_names = []
        for column_name in t_rel_column_names:
            for t_col in self.t_rel.keys():
                if column_name != t_col:
                    continue
                if map_name in self.t_rel[t_col]:
                    map_col_names.append(self.t_rel[t_col][map_name])
        return map_col_names

    def t_rel_column_names(self, map, map_name, map_column_names):
        # Given a map and its column names, return column names (field names) as they are represented in t_rel
        target_col_names = []
        for column_name in map_column_names:
            for t_col in self.t_rel.keys():
                if map_name in self.t_rel[t_col] and self.t_rel[t_col][map_name] == column_name:
                    target_col_names.append(t_col)
        return target_col_names

    def choose_operator(self, map1, map2, map1_name, map2_name):
        # Algorithm 3: Choose suitable merge operator.
        # t_rel is the target relation and it's a class variable (paper: global variable)
        map1_ma = self.find_matches_attr(map1, map1_name)
        map2_ma = self.find_matches_attr(map2, map2_name)
        operator = None
        explanation = ""
        attributes = None
        if not self.diff_matches(set(map1_ma.keys()), set(map2_ma.keys())):
            operator_explanation = self.choose_operator_diff(map1, map2, map1_name, map2_name)
            return {"operator": operator_explanation["operator"], "explanation": operator_explanation["explanation"]}
        else:
            # todo: this is an extension of original dynamap (-> thesis)
            # they share the same attributes with target relation. should those be joined or unioned?
            # naive idea: if there's inclusion, make it a join. Otherwise make it a union.
            included = {}
            for att1 in map1.keys():
                # We first check if the attribute is in target and we can go along that route
                att1_in_target = self.t_rel_column_names(map1, map1_name, [att1])
                if len(att1_in_target) == 1:
                    att1_in_target = att1_in_target[0]
                    if map2_name in self.t_rel[att1_in_target]:
                        att1_in_map2 = self.t_rel[att1_in_target][map2_name]
                        included[(att1, att1_in_map2)] = self.is_attribute_subsumed(map1, map2, map1_name, map2_name, att1_in_target,
                                                                              att1_in_target)
                else:
                    # attribute is not in target. We may still want to join mappings
                    # todo: think if it makes sense. intuitively yes, as it reduces size, but maybe it's not leading to our goal (t_rel cols)
                    # we go naively via column name
                    if att1 in map2:
                        included[(att1, att1)] = self.is_attribute_subsumed(map1, map2, map1_name, map2_name, att1, att1, False)
            if any(included.values()):
                # There are inclusions!
                # We still have to be careful: If both maps include columns that are in target, but *not* in inclusions, they will be renamed in the join
                # and we can no longer provide a unique alias. E.g: join emps & depts on employeeno, what happens to name? It may be in target, but
                # we now have to name it name_depts and name_emps or so and can't put a unique alias in self.t_rel.
                # So we check for that. Once that's clear, we can proceed with a clean left join.
                included_cols_map1 = [c[0] for c in included.keys() if included[c]]
                dont_left_join = False
                for col1 in map1:
                    if col1 not in included_cols_map1:
                        for t_col in self.t_rel:
                            if map1_name in self.t_rel[t_col] and self.t_rel[t_col][map1_name] == col1:
                                # this is a col that is in target, but not in inclusion
                                dont_left_join = True
                included_cols_map2 = [c[1] for c in included.keys() if included[c]]
                for col2 in map2:
                    if col2 not in included_cols_map2:
                        for t_col in self.t_rel:
                            if map2_name in self.t_rel[t_col] and self.t_rel[t_col][map2_name] == col2:
                                # this is a col that is in target, but not in inclusion
                                dont_left_join = True
                if not dont_left_join:
                    included_cols = [c for c in included.keys()]
                    operator = ("left join", map1, map2, map1_name, map2_name, included_cols, False)
                    explanation = map1_name + " and " + map2_name + " don't match the same target attributes. " \
                                  "There's full inclusion in attributes: " + str(included_cols) + ". -> left join."
                else:
                    operator = ("union", map1, map2, map1_name, map2_name,
                                list(set(map1_ma.keys()).intersection(set(map2_ma.keys()))))
                    explanation = map1_name + " and " + map2_name + " don't match the same target attributes. " \
                                  "There's full inclusion between matched target attributes: " + \
                                  str(list(set(map1_ma.keys()).intersection(set(map2_ma.keys())))) + ", but there are other target attributes" \
                                  "represented in either. Can't safely join. -> union."
            else:
                operator = ("union", map1, map2, map1_name, map2_name, list(set(map1_ma.keys()).intersection(set(map2_ma.keys()))))
                explanation = map1_name + " and " + map2_name + " don't match the same target attributes. " \
                              "There's no full inclusion between matched target attributes: " + \
                              str(list(set(map1_ma.keys()).intersection(set(map2_ma.keys())))) + ". -> union."
        return {"operator": operator, "explanation": explanation}

    def choose_operator_diff(self, map1, map2, map1_name, map2_name):
        # Algorithm 4: Generate operator when two mappings match different target attributes.
        # t_rel, pd (profile data) are class variables
        op = None
        explanation = ""
        subsumed_map = self.is_subsumed(map1, map2, map1_name, map2_name)
        if subsumed_map:
            return {"operator": op, "explanation": explanation}
        map1_keys = self.find_keys(map1)
        map2_keys = self.find_keys(map2)
        ind_op = self.max_ind(map1, map2, map1_keys, map2_keys, map1_name, map2_name)
        if ind_op:
            # There is an overlap between candidate keys in map1 and map2
            return {"operator": ind_op["op"], "explanation": ind_op["explanation"]}
        else:
            # There is no overlap between candidate keys in map1 and map2, try to find overlap between candidate keys in map1 with attributes in map2
            # and overlap between candidate keys in map2 with attributes in map1
            map1_ind_op = self.max_ind(map1, map2, map1_keys, list(map2.keys()), map1_name, map2_name)
            map2_ind_op = self.max_ind(map2, map1, map2_keys, list(map1.keys()), map1_name, map2_name)
            if map1_ind_op and map2_ind_op:
                if map1_ind_op["inclusion_ratio"] >= map2_ind_op["inclusion_ratio"]:
                    return {"operator": map1_ind_op["op"], "explanation": map1_ind_op["explanation"]}
                else:
                    return {"operator": map2_ind_op["op"], "explanation": map2_ind_op["explanation"]}
            elif map1_ind_op:
                return {"operator": map1_ind_op["op"], "explanation": map1_ind_op["explanation"]}
            elif map2_ind_op:
                return {"operator": map2_ind_op["op"], "explanation": map2_ind_op["explanation"]}
            else:
                # There is no overlap between candidate keys in map1 with attributes in map2 or the other way round.
                # Next, if a foreign key relationship cannot be inferred, then, on
                # lines 26–27, FindMatchedKeys retrieves the candidate keys from
                # both mappings that match target attributes and checks if they
                # match the same target attributes (line 28). If they do, then the
                # two mappings are merged using full outer join, where the join
                # condition is on the attributes that meet the requirements (line
                # 29). The intuition behind this last step is that even if there is
                # no overlap between the attribute values of the two mappings,
                # it could be that there is instance complementarity between the
                # two mappings, in which case performing a full outer join vertically
                # aligns the key attributes that match the same target attributes
                map1_mk = self.find_matched_keys(map1, map1_name)
                map2_mk = self.find_matched_keys(map2, map2_name)
                same_matches = self.same_matches(map1_mk, map2_mk)
                if same_matches:
                    op = ("outer join", map1, map2, map1_name, map2_name, list(same_matches))
                    explanation = map1_name + " and " + map2_name + " match the same target attributes. " \
                                  "There's no full inclusion between matched target attributes. Align attributes " \
                                  "that match the same target attributes: " + \
                                  str(list(same_matches)) + ". -> outer join."
        return {"operator": op, "explanation": explanation}

    def new_mapping(self, operator, map1, map2, map1_name, map2_name, attributes, target_col_translation=True):
        # apply operator to two mappings
        if operator == "union":
            return self.op_union(map1, map2, map1_name, map2_name, attributes)
        elif operator == "join":
            return self.op_inner_join(map1, map2, map1_name, map2_name, attributes)
        elif operator == "left join":
            return self.op_left_join(map1, map2, map1_name, map2_name, attributes, target_col_translation)
        elif operator == "outer join":
            return self.op_outer_join(map1, map2, map1_name, map2_name, attributes)
        else:
            raise RuntimeError("unknown operator name")

    def op_union(self, map1, map2, map1_name, map2_name, attributes):
        # attributes holds the names of columns in t_rel
        df1 = pd.DataFrame.from_dict(map1)
        df2 = pd.DataFrame.from_dict(map2)
        # rename columns in dataframes to names in t_rel.
        replace_df1 = {}
        for col in df1.columns.to_list():
            if col not in self.t_rel:
                for t_col in self.t_rel:
                    if map1_name in self.t_rel[t_col] and self.t_rel[t_col][map1_name] == col:
                        replace_df1[col] = t_col
        df1.rename(columns=replace_df1, inplace=True)
        replace_df2 = {}
        for col in df2.columns.to_list():
            if col not in self.t_rel:
                for t_col in self.t_rel:
                    if map2_name in self.t_rel[t_col] and self.t_rel[t_col][map2_name] == col:
                        replace_df2[col] = t_col
        df2.rename(columns=replace_df2, inplace=True)
        # tracking column renaming
        self.renamed_columns[map1_name] = replace_df1
        self.renamed_columns[map2_name] = replace_df2
        # using pandas union
        try:
            unioned_dfs = pd.concat([df1, df2], axis=0, ignore_index=True)
        except Exception as e:
            raise RuntimeError("op_union failed on", map1_name, "and", map2_name, "with exception", e)
        # getting unioned dfs back to dict format we use here
        union_map = unioned_dfs.to_dict(orient='list')
        return union_map

    def op_inner_join(self, map1, map2, map1_name, map2_name, attributes: list):
        df1 = pd.DataFrame.from_dict(map1)
        df2 = pd.DataFrame.from_dict(map2)
        # attributes gives us column names as they are in t_rel, so we have to find out what their aliases are in df1 and df2
        df1_aliases = self.aliases_for_t_rel_columns(map1_name, attributes)
        df2_aliases = self.aliases_for_t_rel_columns(map2_name, attributes)
        # using pandas merge to join on non-index columns
        try:
            joined_dfs = df1.join(df2.set_index(df2_aliases), on=df1_aliases, how='inner')
        except Exception as e:
            raise RuntimeError("op_inner_join failed on", map1_name, "and", map2_name, "with exception", e)
        # tracking column renaming
        for i, col in enumerate(df1_aliases):
            if map2_name in self.renamed_columns:
                self.renamed_columns[map2_name][df2_aliases[i]] = df1_aliases[i]
            else:
                self.renamed_columns[map2_name] = {df2_aliases[i]: df1_aliases[i]}
        if 'index_y' in joined_dfs:
            joined_dfs.drop('index_y', axis=1, inplace=True)
        # getting unioned dfs back to dict format we use here
        join_map = joined_dfs.to_dict(orient='list')
        return join_map

    def op_left_join(self, map1, map2, map1_name, map2_name, attributes: list, target_col_translation=True):
        df1 = pd.DataFrame.from_dict(map1)
        df2 = pd.DataFrame.from_dict(map2)
        # attributes gives us column names as they are in t_rel, so we have to find out what their aliases are in df1 and df2
        if target_col_translation:
            df1_aliases = self.aliases_for_t_rel_columns(map1_name, attributes)
            df2_aliases = self.aliases_for_t_rel_columns(map2_name, attributes)
        else:
            df1_aliases = [att[0] for att in attributes]
            df2_aliases = [att[1] for att in attributes]
        # using pandas merge to join on non-index columns
        try:
            #joined_dfs = df1.join(df2.set_index(df2_aliases), on=df1_aliases, how='left')
            joined_dfs = df1.join(df2.set_index(df2_aliases), on=df1_aliases, how='left')
            #joined_dfs = df1.merge(df2, left_on=df1_aliases, right_on=df2_aliases, how="left")
            # suffixes=("_"+map1_name,"_"+map2_name)
            # Statistics for instance loss
            outer_joined_dfs = df1.join(df2.set_index(df2_aliases), on=df1_aliases, how='outer')
            lost_rows = len(outer_joined_dfs) - len(joined_dfs)
            if lost_rows > 0:
                self.instance_loss[map1_name + "_" + map2_name] = lost_rows
        except Exception as e:
            raise RuntimeError("op_left_join failed on", map1_name, "and", map2_name, "with exception", e)
        # tracking column renaming
        for i, col in enumerate(df1_aliases):
            if map2_name in self.renamed_columns:
                self.renamed_columns[map2_name][df2_aliases[i]] = df1_aliases[i]
            else:
                self.renamed_columns[map2_name] = {df2_aliases[i]: df1_aliases[i]}
        # getting unioned dfs back to dict format we use here
        join_map = joined_dfs.to_dict(orient='list')
        return join_map

    def op_outer_join(self, map1, map2, map1_name, map2_name, attributes):
        df1 = pd.DataFrame.from_dict(map1)
        df2 = pd.DataFrame.from_dict(map2)
        # attributes gives us column names as they are in t_rel, so we have to find out what their aliases are in df1 and df2
        df1_aliases = self.aliases_for_t_rel_columns(map1_name, attributes)
        df2_aliases = self.aliases_for_t_rel_columns(map2_name, attributes)
        # using pandas merge to join on non-index columns
        try:
            joined_dfs = df1.merge(df2, left_on=df1_aliases, right_on=df2_aliases, suffixes=('', '_y'), how="outer")
        except Exception as e:
            raise RuntimeError("op_outer_join failed on", map1_name, "and", map2_name, "with exception", e)
        # tracking column renaming
        for i, col in enumerate(df1_aliases):
            if map2_name in self.renamed_columns:
                self.renamed_columns[map2_name][df2_aliases[i]] = df1_aliases[i]
            else:
                self.renamed_columns[map2_name] = {df2_aliases[i]: df1_aliases[i]}
        if 'index_y' in joined_dfs:
            joined_dfs.drop('index_y', axis=1, inplace=True)
        # getting unioned dfs back to dict format we use here
        join_map = joined_dfs.to_dict(orient='list')
        return join_map
        # df1 = pd.DataFrame.from_dict(map1)
        # df2 = pd.DataFrame.from_dict(map2)
        # # using pandas merge to join on non-index columns
        # outer_joined_dfs = pd.merge(left=df2, right=df1, on=attributes, how='outer')
        # # getting unioned dfs back to dict format we use here
        # outer_join_map = outer_joined_dfs.to_dict(orient='list')
        # return outer_join_map

    def is_fittest(self, map, source_names):
        # return fittest map from self.metadata (? algorithm 2)
        # isFittest checks if the intermediate mapping has the
        # highest fitness of any mapping involving the same initial sources,
        # if so, it is retained (alg 2 lines 9–10).
        # Compute the fitness of the mapping
        fitness_0 = self.fitness(map, source_names[0])
        fitness_1 = self.fitness(map, source_names[1])
        fitness = max(fitness_0, fitness_1)
        # Check if the mapping has the highest fitness of any mapping involving the same initial sources
        # If yes, update highest known fitness value
        if fitness > self.get_max_fitness(source_names):
            # Update highest fitness value by source, so we can see which source combinations have the highest fitness value
            source_1_name = source_names[0]
            source_2_name = source_names[1]
            if source_1_name in self.fitness_value_by_source:
                self.fitness_value_by_source[source_1_name][source_2_name] = fitness
            else:
                self.fitness_value_by_source[source_1_name] = {source_2_name: fitness}
            if source_2_name in self.fitness_value_by_source:
                self.fitness_value_by_source[source_2_name][source_1_name] = fitness
            else:
                self.fitness_value_by_source[source_2_name] = {source_1_name: fitness}
            return True
        return False

    def get_max_fitness(self, source_names):
        # Return the highest fitness of any mapping involving the same sources
        max_fitness = 0
        source_1_name = source_names[0]
        source_2_name = source_names[1]
        if source_1_name in self.fitness_value_by_source and source_2_name in self.fitness_value_by_source[source_1_name]:
            max_fitness = max(max_fitness, self.fitness_value_by_source[source_1_name][source_2_name])
        if source_2_name in self.fitness_value_by_source and source_1_name in self.fitness_value_by_source[source_2_name]:
            max_fitness = max(max_fitness, self.fitness_value_by_source[source_2_name][source_1_name])
        return max_fitness

    def find_matches_attr(self, map, map_name):
        # For each mapping, get attributes that are in target
        # i.e. column names that appear in t_rel
        # format: col_name in target : col_name in map
        cols = {}
        for t_col in self.t_rel.keys():
            if map_name in self.t_rel[t_col]:
                cols[t_col] = self.t_rel[t_col][map_name]
        return cols

    def diff_matches(self, set1, set2):
        # DiffMatches checks whether the matches are for different target attributes. If they are, they become candidates for joining,
        # to be decided by ChooseOperatorDiff (line 7). If the matches are for the same target attributes, then the two mappings are unioned (line 9).
        # are the matched attributes from map1 with t_rel (set1) the same as matched attributes from map2 with t_rel (set2)?
        # This should work because we use the keys that we set in self.t_rel for populating the sets in self.find_matches_attr()
        return len(set1.symmetric_difference(set2)) == 0

    def is_attribute_subsumed(self, map1, map2, map1_name, map2_name, map1_attribute, map2_attribute, look_via_t_rel=True):
        # Helper function: Given attributes in two maps, is attribute1 from map1 subsumed in attribute2 in map2?
        if look_via_t_rel:
            # we give attribute names as they are in t_rel
            for t_col in self.t_rel.keys():
                if t_col in [map1_attribute, map2_attribute]:
                    if map1_name in self.t_rel[t_col] and map2_name in self.t_rel[t_col]:
                        # atts point to same target column
                        # todo: replace this with call to profile data (propagation method)
                            att1 = self.aliases_for_t_rel_columns(map1_name, [map1_attribute])[0]
                            att2 = self.aliases_for_t_rel_columns(map2_name, [map2_attribute])[0]
        else:
            att1 = map1_attribute
            att2 = map2_attribute
        att1_in_map1 = set(map1[att1])
        att2_in_map2 = set(map2[att2])
        if att1_in_map1.intersection(att2_in_map2) == att1_in_map1:
            # this att is subsumed in map2
            return True
        else:
            return False

    def is_subsumed(self, map1, map2, map1_name, map2_name):
        # In lines 4–7, IsSubsumed determines whether, on attributes
        # that match the target, the profiling data has inclusion dependencies between an attribute in one mapping and a corresponding
        # attribute in the other mapping. If so, the subsumed mapping is
        # discarded from the set of kept mappings (for further reference,
        # in Section 4.3, we refer to the kept mappings as memorized subsolutions) and null is returned.
        map1_ma = self.find_matches_attr(map1, map1_name)
        map2_ma = self.find_matches_attr(map2, map2_name)

        map1_subsumptions = {}
        for att in map1_ma:
            if att in map2_ma:
                map1_subsumptions[att] = self.is_attribute_subsumed(map1, map2, map1_name, map2_name, att, att)
            else:
                # This is an attribute that appears in target, but it is not subsumed! I.e. mapping1 carries more value than mapping2
                # because it knows more target attributes, even though one may be subsumed.
                map1_subsumptions[att] = False
        # todo: all or any? text in paper sounds like "any", but then the two cases of subsumption map1 by map2 and the other way round wouldn't be symmetrical anymore
        if map1_subsumptions and all(map1_subsumptions.values()):
            # map1 is subsumed by map2
            self.remove_mapping(map1_name)
            return True

        map2_subsumptions = {}
        for att in map2_ma:
            if att in map1_ma:
                map2_subsumptions[att] = self.is_attribute_subsumed(map2, map1, map2_name, map1_name, att, att)
            else:
                map2_subsumptions[att] = False
        if map2_subsumptions and all(map2_subsumptions.values()):
            # map2 is subsumed by map1
            self.remove_mapping(map2_name)
            return True
        return False

    def remove_mapping(self, map_name):
        for solution in self.sub_solution:
            if map_name in self.sub_solution[solution]:
                sub_sol = copy.deepcopy(self.sub_solution)
                del sub_sol[solution][map_name]
                self.sub_solution = sub_sol

    def get_mapping_rows(self, map_name):
        for solution in self.sub_solution:
            if map_name in self.sub_solution[solution]:
                return self.sub_solution[solution][map_name]

    def find_keys(self, map):
        # todo: replace with call to profiling function
        # we want candidate keys here, i.e. attributes that are unique
        keys = set()
        for attr, values in map.items():
            if len(values) == len(set(values)):
                keys.add(attr)
        return keys

    def max_ind(self, map1, map2, map1_keys, map2_keys, map1_name, map2_name):
        # todo: replace this with call to profiling function
        # todo: in this function and "find_keys", we could replace these functions with pk/fk inferrence from paper fernandez2018
        map1_map2_inclusions = {}
        # e.g. "df1_df2": "employeeno"
        for key1 in map1_keys:
            # e.g. "df3": "empno"
            for key2 in map2_keys:
                for t_col in self.t_rel.keys():
                    if map1_name in self.t_rel[t_col] and map2_name in self.t_rel[t_col]:
                        # keys both point to the same target column!
                        inclusion = sum(el in map1[key1] for el in map2[key2])
                        inclusion_ratio = inclusion/len(map1[key1])
                        map1_map2_inclusions[t_col] = inclusion_ratio
        map2_map1_inclusions = {}
        for key2 in map2_keys:
            for key1 in map1_keys:
                for t_col in self.t_rel.keys():
                    if map1_name in self.t_rel[t_col] and map2_name in self.t_rel[t_col]:
                        # keys both point to the same target column!
                        inclusion = sum(el in map2[key2] for el in map1[key1])
                        inclusion_ratio = inclusion / len(map2[key2])
                        map2_map1_inclusions[t_col] = inclusion_ratio

        if not map1_map2_inclusions or not map2_map1_inclusions:
            return None

        if max(map1_map2_inclusions.values()) > max(map2_map1_inclusions.values()):
            max_inclusion = {key: inclusion for key, inclusion in map1_map2_inclusions.items() if
                             inclusion == max(map1_map2_inclusions.values())}
        else:
            max_inclusion = {key: inclusion for key, inclusion in map2_map1_inclusions.items() if
                             inclusion == max(map2_map1_inclusions.values())}

        if max(max_inclusion.values()) == 1:
            # if θ = 1.0, then the inclusion dependency is total and the
            # chosen operator is join because a foreign key relationship
            # is inferred between two mappings on their candidate key
            # attributes (lines 12–13);
            # if inclusion_ratio == 1 --> all elements from map1[key1] are in map2[key2], so we assume key1 is a pk for map1 and a fk for map2 (? foreign key relationship)
            op = ("join", map1, map2, map1_name, map2_name, list(max_inclusion.keys()))
            explanation = map1_name + " and " + map2_name + " match the same target attributes. " \
                          "There's full inclusion between matched target attributes: " + \
                          str(list(max_inclusion.keys())) + ". -> (inner) join."
            return {"op": op,
                    "inclusion_ratio": 1,
                    "explanation": explanation}
        elif 0 < max(max_inclusion.values()) < 1:
            # if θ ∈ (0, 1.0), then the inclusion dependency is partial
            # and the operator is a full outer join because a foreign key
            # relationship cannot be inferred so the algorithm joins the
            # tuples that can be joined and keeps the remaining data (lines
            # 14–15).
            op = ("outer join", map1, map2, map1_name, map2_name, list(max_inclusion.keys()))
            explanation = map1_name + " and " + map2_name + " match the same target attributes. " \
                          "There's partial inclusion between matched target attributes: " + \
                          str(list(max_inclusion.keys())) + ". -> outer join to keep data that could not be joined."
            return {"op": op,
                    "inclusion_ratio": max(max_inclusion.values()),
                    "explanation": explanation}
        else:
            return None

    def find_matched_keys(self, map, map_name):
        matched_keys = set()
        for att in list(map.keys()):
            for t_col in self.t_rel.keys():
                if map_name in self.t_rel[t_col] and att == self.t_rel[t_col][map_name]:
                    matched_keys.add(t_col)
        return matched_keys

    def same_matches(self, map1_mk, map2_mk):
        return map1_mk.intersection(map2_mk)

    def fitness(self, map, map_name):
        # Algorithm 5: Fitness function
        # todo: get cardinalities and null counts from polypheny statistics module
        # atts = self.remove_outliers(map)
        null_counts = []
        attr_counts = []
        # target_counts = 0
        for att in map:
            # null counts
            values = map[att]
            null_count = len([v for v in values if pd.isna(v)])
            null_counts.append(null_count)
            # attr counts
            attr_count = len(values)
            attr_counts.append(attr_count)
            # # target counts
            # att_in_trel = self.t_rel_column_names(map, map_name, [att])
            # if len(att_in_trel) == 1:
            #     target_counts += 1
        max_attr_nulls = max(null_counts)
        # target_percentage = target_counts/len(self.t_rel)
        # attr_null_percentages = result = [n/a for n, a in zip(null_counts, attr_counts)]
        # attr_avg_null_percentage = sum(attr_null_percentages)/len(attr_null_percentages)
        # Reward merges, but prefer
        merge_counts = self.count_merge_types(self.mapping_path.get(map_name), {})
        all_merges = sum(merge_counts.values())
        union_merges = merge_counts.get("union", 0)
        # The number of largely complete tuples in the mapping is estimated to be
        # the cardinality of the mapping minus the number of nulls in the attribute with the most nulls.
        # Original fitness logic as I understood it from the dynaMap paper:
        # reward many rows with few null values
        return self.map_cardinality(map) - max_attr_nulls + all_merges - union_merges
        # todo: this is also an extension of original dynamap
        # New idea for fitness logic:
        # Reward having many cols from target + few NULLs in rows
        # and many merges, but few unions
        # return target_percentage - attr_avg_null_percentage

    def map_cardinality(self, map):
        # cardinality of a map = number of tuples = number of rows
        most_tups = 0
        for att in map:
            if len(map[att]) > most_tups:
                most_tups = len(map[att])
        return most_tups

    def remove_outliers(self, map):
        # Specifically, given a mapping with a list of attributes,
        # RemoveOutliers returns the set of attributes that are not outliers
        # with respect to the number of nulls they contain (alg 5, line 2)
        # Outliers are identified using the Median and Interquartile Deviation Method.
        null_counts = []
        att_null_counts = {}
        for att in map:
            values = map[att]
            null_count = len([v for v in values if pd.isna(v)])
            null_counts.append(null_count)
            att_null_counts[att] = null_count
        median_null_count = statistics.median(null_counts)
        if len(null_counts) >= 2:
            iqr = statistics.quantiles(null_counts, n=4)[2] - statistics.quantiles(null_counts, n=4)[1]
            lower_bound = median_null_count - 1.5 * iqr
            upper_bound = median_null_count + 1.5 * iqr
            return [a for a in att_null_counts if lower_bound <= att_null_counts[a] <= upper_bound]
        else:
            return [a for a in att_null_counts]

    def reconstruct_mapping_path(self, mapping_name, mapping_path=[]):
        # Reconstruct the expanded mapping path for a mapping name, i.e. down to base source relations
        if mapping_name in self.source_relations:
            return [mapping_name]
        mapping_path = copy.deepcopy(self.mapping_path.get(mapping_name))
        # ['join', 'df1', 'df2_df3', ['A']]
        for index, j_mapping_name in enumerate(mapping_path[1:-1]):
            if j_mapping_name not in self.source_relations:
                # recursively replace mappings with their mapping paths
                mapping_path[index + 1] = self.reconstruct_mapping_path(j_mapping_name)
        # ['join', 'df1', ['union', 'df2', 'df3', None], 'A']
        return mapping_path

    # def explain_mapping_path(self, mapping_path):
    #     # Given a 4-set of a mapping path, return an explanation of the chosen merge operation
    #     if len(mapping_path) == 4:
    #         rel = mapping_path[0]
    #         from_table = mapping_path[1]
    #         to_table = mapping_path[2]
    #         cols = mapping_path[3]
    #         explanation = from_table + " and " + to_table + ": " + rel + " on " + str(cols)
    #         if rel == "union":
    #             explanation += "()"
    #         elif rel == "left join":
    #             pass
    #         elif rel == "join":
    #             pass
    #         elif rel == "outer join":
    #             pass
    #         return explanation
    #     elif len(mapping_path) == 1:
    #         return
    #     else:
    #         raise RuntimeError("wrong mapping path format", mapping_path)

    def explain_mapping_operations_for_mapping_name(self, mapping_name):
        # Explain a mapping name, i.e. why were the mapping operations leading here chosen?
        explanations = []
        ancestors = self.find_parentage([mapping_name])
        for ancestor in ancestors:
            explanation = self.explanations.get(ancestor, None)
            if explanation:
                explanations.append(explanation)
        return explanations

    def count_merge_types(self, mapping_path, merge_types={}):
        if not type(mapping_path) == list:
            return merge_types
        if len(mapping_path) == 4:
            rel = mapping_path[0]
            from_table = mapping_path[1]
            to_table = mapping_path[2]
            cols = mapping_path[3]
            if rel in merge_types:
                merge_types[rel] += 1
            else:
                merge_types[rel] = 1
        elif len(mapping_path) == 1:
            return
        else:
            raise RuntimeError("wrong mapping path format", mapping_path)
        # recursively get merge types from mappings, starting from the simplest ones
        if (isinstance(from_table, list)):
            self.count_merge_types(from_table, merge_types)
        if (isinstance(to_table, list)):
            self.count_merge_types(to_table, merge_types)
        return merge_types

    def mapping_name_to_field_relationships(self, mapping_path, field_relationships=[]):
        # ['outer join', 'df3', ['outer join', 'df1', 'df2', ['B', 'C']], ['A', 'C']]
        if len(mapping_path) == 4:
            rel = mapping_path[0]
            from_table = mapping_path[1]
            to_table = mapping_path[2]
            cols = mapping_path[3]
        elif len(mapping_path) == 1:
            return
        else:
            raise RuntimeError("wrong mapping path format", mapping_path)
        if not (isinstance(from_table, list) or isinstance(to_table, list)):
            if cols is None:
                from_cols = list(self.get_mapping_rows(from_table).keys())
                to_cols = list(self.get_mapping_rows(to_table).keys())
                for fc in from_cols:
                    for tc in to_cols:
                        rel_strength = self.get_match_strength(from_table, to_table, fc, tc)
                        field_relationships.append(FieldRelationship(from_table, to_table,
                                                                     fc, tc,
                                                                     relationship_type=rel,
                                                                     relationship_strength=rel_strength))
            elif type(cols) == list and type(cols[0]) == tuple:
                from_cols = [c[0] for c in cols]
                to_cols = [c[1] for c in cols]
                for fc in from_cols:
                    for tc in to_cols:
                        rel_strength = self.get_match_strength(from_table, to_table, fc, tc)
                        field_relationships.append(FieldRelationship(from_table, to_table,
                                                                     fc, tc,
                                                                     relationship_type=rel,
                                                                     relationship_strength=rel_strength))
            else:
                for col in cols:
                    rel_strength = self.get_match_strength(from_table, to_table, col, col)
                    field_relationships.append(FieldRelationship(from_table, to_table,
                                                                 col, col,
                                                                 relationship_type=rel,
                                                                 relationship_strength=rel_strength))
        else:
            # recursively get fieldrelationships from mappings, starting from the simplest ones
            if (isinstance(from_table, list)):
                self.mapping_name_to_field_relationships(from_table, field_relationships)
            if (isinstance(to_table, list)):
                self.mapping_name_to_field_relationships(to_table, field_relationships)
        return field_relationships

    def k_best_mappings(self, k):
        # In the current approach, Dynamap outputs the best k mappings where
        # k is an integer. The output mappings merge subsets of i source
        # relations, 1 ≤ i ≤ n, where n is the total number of input source
        # relations, which were obtained during the dynamic programming
        # search, and that are ranked according to their fitness.
        # todo: for thesis writing, what are the tie breaks here?
        num = len(self.highest_fitness_value)
        k = int(k)
        k = k if k <= num else num
        k_highest_fitness_values = dict(collections.Counter(self.highest_fitness_value).most_common(k))
        # tie break 1: prefer mappings to source relations
        not_source_mappings = [m for m in k_highest_fitness_values.keys() if m not in self.source_relations]
        if not_source_mappings:
            k_highest_fitness_values = {k: v for k, v in k_highest_fitness_values.items() if k in not_source_mappings}
        #print("k highest fitness values", k_highest_fitness_values)
        # example: {'df1_df2_df3': 6, 'df2_df1_df3': 6, 'df3_df1_df2': 6, 'df1': 3, 'df2': 3}
        # format that we want for graphviz visualization:
        # [{'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'deptno', 'field_name2': 'deptno', 'relationship_type': 'view union', 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'name', 'field_name2': 'name', 'relationship_type': 'view union', 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'emp', 'table2': 'work', 'field_name1': 'employeeno', 'field_name2': 'employeeno', 'relationship_type': 'view union', 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'deptno', 'field_name2': 'deptno', 'relationship_type': ('join',), 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'name', 'field_name2': 'name', 'relationship_type': ('join',), 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'emp', 'table2': 'work', 'field_name1': 'employeeno', 'field_name2': 'employeeno', 'relationship_type': ('join',), 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'deptno', 'field_name2': 'deptno', 'relationship_type': 'view union', 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'name', 'field_name2': 'name', 'relationship_type': 'view union', 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'emp', 'table2': 'work', 'field_name1': 'employeeno', 'field_name2': 'employeeno', 'relationship_type': 'view union', 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'deptno', 'field_name2': 'deptno', 'relationship_type': ('join',), 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'name', 'field_name2': 'name', 'relationship_type': ('join',), 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'emp', 'table2': 'work', 'field_name1': 'employeeno', 'field_name2': 'employeeno', 'relationship_type': ('join',), 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'deptno', 'field_name2': 'deptno', 'relationship_type': 'view union', 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'name', 'field_name2': 'name', 'relationship_type': 'view union', 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'emp', 'table2': 'work', 'field_name1': 'employeeno', 'field_name2': 'employeeno', 'relationship_type': 'view union', 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'deptno', 'field_name2': 'deptno', 'relationship_type': ('join',), 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'name', 'field_name2': 'name', 'relationship_type': ('join',), 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'emp', 'table2': 'work', 'field_name1': 'employeeno', 'field_name2': 'employeeno', 'relationship_type': ('join',), 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'deptno', 'field_name2': 'deptno', 'relationship_type': 'view union', 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'name', 'field_name2': 'name', 'relationship_type': 'view union', 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'emp', 'table2': 'work', 'field_name1': 'employeeno', 'field_name2': 'employeeno', 'relationship_type': 'view union', 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'deptno', 'field_name2': 'deptno', 'relationship_type': ('join',), 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'name', 'field_name2': 'name', 'relationship_type': ('join',), 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'emp', 'table2': 'work', 'field_name1': 'employeeno', 'field_name2': 'employeeno', 'relationship_type': ('join',), 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'deptno', 'field_name2': 'deptno', 'relationship_type': 'view union', 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'name', 'field_name2': 'name', 'relationship_type': 'view union', 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'emp', 'table2': 'work', 'field_name1': 'employeeno', 'field_name2': 'employeeno', 'relationship_type': 'view union', 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'deptno', 'field_name2': 'deptno', 'relationship_type': ('join',), 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'name', 'field_name2': 'name', 'relationship_type': ('join',), 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'emp', 'table2': 'work', 'field_name1': 'employeeno', 'field_name2': 'employeeno', 'relationship_type': ('join',), 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'deptno', 'field_name2': 'deptno', 'relationship_type': 'view union', 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'name', 'field_name2': 'name', 'relationship_type': 'view union', 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'emp', 'table2': 'work', 'field_name1': 'employeeno', 'field_name2': 'employeeno', 'relationship_type': 'view union', 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'deptno', 'field_name2': 'deptno', 'relationship_type': ('join',), 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'name', 'field_name2': 'name', 'relationship_type': ('join',), 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'emp', 'table2': 'work', 'field_name1': 'employeeno', 'field_name2': 'employeeno', 'relationship_type': ('join',), 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'deptno', 'field_name2': 'deptno', 'relationship_type': 'view union', 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'name', 'field_name2': 'name', 'relationship_type': 'view union', 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'emp', 'table2': 'work', 'field_name1': 'employeeno', 'field_name2': 'employeeno', 'relationship_type': 'view union', 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'deptno', 'field_name2': 'deptno', 'relationship_type': ('join',), 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'name', 'field_name2': 'name', 'relationship_type': ('join',), 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'emp', 'table2': 'work', 'field_name1': 'employeeno', 'field_name2': 'employeeno', 'relationship_type': ('join',), 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'deptno', 'field_name2': 'deptno', 'relationship_type': 'view union', 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'name', 'field_name2': 'name', 'relationship_type': 'view union', 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'emp', 'table2': 'work', 'field_name1': 'employeeno', 'field_name2': 'employeeno', 'relationship_type': 'view union', 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'deptno', 'field_name2': 'deptno', 'relationship_type': ('join',), 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'name', 'field_name2': 'name', 'relationship_type': ('join',), 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'emp', 'table2': 'work', 'field_name1': 'employeeno', 'field_name2': 'employeeno', 'relationship_type': ('join',), 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'deptno', 'field_name2': 'deptno', 'relationship_type': 'view union', 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'name', 'field_name2': 'name', 'relationship_type': 'view union', 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'emp', 'table2': 'work', 'field_name1': 'employeeno', 'field_name2': 'employeeno', 'relationship_type': 'view union', 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'deptno', 'field_name2': 'deptno', 'relationship_type': ('join',), 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'name', 'field_name2': 'name', 'relationship_type': ('join',), 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'emp', 'table2': 'work', 'field_name1': 'employeeno', 'field_name2': 'employeeno', 'relationship_type': ('join',), 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'deptno', 'field_name2': 'deptno', 'relationship_type': 'view union', 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'name', 'field_name2': 'name', 'relationship_type': 'view union', 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'emp', 'table2': 'work', 'field_name1': 'employeeno', 'field_name2': 'employeeno', 'relationship_type': 'view union', 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'deptno', 'field_name2': 'deptno', 'relationship_type': ('join',), 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'name', 'field_name2': 'name', 'relationship_type': ('join',), 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'emp', 'table2': 'work', 'field_name1': 'employeeno', 'field_name2': 'employeeno', 'relationship_type': ('join',), 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'deptno', 'field_name2': 'deptno', 'relationship_type': 'view union', 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'name', 'field_name2': 'name', 'relationship_type': 'view union', 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'emp', 'table2': 'work', 'field_name1': 'employeeno', 'field_name2': 'employeeno', 'relationship_type': 'view union', 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'deptno', 'field_name2': 'deptno', 'relationship_type': ('join',), 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'name', 'field_name2': 'name', 'relationship_type': ('join',), 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'emp', 'table2': 'work', 'field_name1': 'employeeno', 'field_name2': 'employeeno', 'relationship_type': ('join',), 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'deptno', 'field_name2': 'deptno', 'relationship_type': 'view union', 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'name', 'field_name2': 'name', 'relationship_type': 'view union', 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'emp', 'table2': 'work', 'field_name1': 'employeeno', 'field_name2': 'employeeno', 'relationship_type': 'view union', 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'deptno', 'field_name2': 'deptno', 'relationship_type': ('join',), 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'name', 'field_name2': 'name', 'relationship_type': ('join',), 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'emp', 'table2': 'work', 'field_name1': 'employeeno', 'field_name2': 'employeeno', 'relationship_type': ('join',), 'relationship_strength': 1}]
        viz_info = {}
        for mapping_name in k_highest_fitness_values:
            # ['join', 'df1', ['union', 'df2', 'df3', None], 'A']
            mapping_path = self.reconstruct_mapping_path(mapping_name)
            field_relationships = self.mapping_name_to_field_relationships(mapping_path)
            mapping_rows = self.get_mapping_rows(mapping_name)
            max_map_length = 0
            for col in mapping_rows:
                if max_map_length < len(mapping_rows[col]):
                    max_map_length = len(mapping_rows[col])
            viz_info[mapping_name] = {
                "fitness_score": k_highest_fitness_values[mapping_name],
                "mapping_path": mapping_path,
                "field_relationships": field_relationships,
                "mapping_rows": mapping_rows,
                "max_mapping_rows_length": max_map_length
            }
            #print(viz_info)
        return viz_info

    def instance_loss_for_mapping_name(self, map_name):
        # Given a mapping name, find how many rows were (potentially) lost during its creation
        ancestors = self.find_parentage([map_name])
        instance_loss = 0
        for ancestor in ancestors:
            instance_loss += self.instance_loss.get(ancestor, 0)
        instance_loss_string = "Lost " + str(instance_loss) + " rows during mapping of " + map_name + ".\r\n"
        return instance_loss_string

    def renamed_columns_for_mapping(self, map_name):
        # Given a mapping name, how did we rename columns in its ancestors?
        renamed_column_string = ""
        ancestors = self.find_parentage([map_name])
        for ancestor in ancestors:
            ancestor_renaming = self.renamed_columns.get(ancestor, "")
            if ancestor_renaming:
                for col in ancestor_renaming:
                    if col != ancestor_renaming[col]:
                        renamed_column_string += "Renamed column " + col + " in " + ancestor + " to " + ancestor_renaming[col] + "; "
        if not renamed_column_string:
            return
        else:
            return renamed_column_string[:-2] + "."

if __name__ == "__main__":
    dfs = {"df1": {'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]},
           "df2": {'B': [10, 11, 12], 'C': [13, 14, 15]},
           "df3": {'A': [16, 17, 18], 'C': [19, 20, 21]}}
    matches = {(('df1', 'B'), ('df2', 'B')): 1,
               (('df1', 'C'), ('df2', 'C')): 1,
               (('df1', 'A'), ('df3', 'A')): 1,
               (('df2', 'C'), ('df3', 'C')): 1}
    dynamap = pyDynaMap(dfs, matches)
    dynamap.generate_mappings(len(dfs))
    print(dynamap.k_best_mappings(3))
    print(dynamap.highest_fitness_value)
    print(dynamap.sub_solution)
