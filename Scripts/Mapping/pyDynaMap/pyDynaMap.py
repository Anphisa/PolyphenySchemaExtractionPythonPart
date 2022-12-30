# Re-implementing the DynaMap algorithm as described in "Schema mapping generation in the wild" (https://doi.org/10.1016/j.is.2021.1019049)
import pandas as pd
import math
import copy
import statistics

class pyDynaMap():
    def __init__(self, source_relations: dict, matches: dict):
        for rel in source_relations:
            col_names = source_relations[rel].keys()
            if len(col_names) > len(set(col_names)):
                raise RuntimeError("pyDynaMap doesn't accept duplicate column names. "
                                   "Duplicate column names found in source_relations: ", source_relations)
        self.source_relations = source_relations
        self.matches = matches
        self.t_rel = self.target_relation_from_matches()
        self.sub_solution = {}
        # todo: profile data
        self.pd = {}
        self.highest_fitness_value = {}
        self.mapping_sources = {}

    def target_relation_from_matches(self):
        # given column names from matches, use all of them as names in a (simple) target relation
        # example: {(('depts', 'deptno'), ('emps', 'deptno')): 1, (('depts', 'name'), ('emps', 'name')): 1, (('emp', 'employeeno'), ('work', 'employeeno')): 1}
        target_columns = []
        for match in self.matches:
            col_name1 = match[0][1]
            col_name2 = match[1][0]
            if col_name1 == col_name1:
                target_columns.append(col_name1)
            elif col_name1 in col_name2:
                target_columns.append(col_name1)
            elif col_name2 in col_name1:
                target_columns.append(col_name2)
            else:
                # return shorter string
                col_name = col_name1 if len(col_name1) < len(col_name2) else col_name2
                target_columns.append(col_name)
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
        for map_i_name in batch1:
            for map_j_name in batch2:
                # If map_i == map_j, do not return union with itself (probably not intended)
                if map_i_name == map_j_name:
                    continue
                # If a relation is already part of a mapping, don't combine it in again
                if self.mapping_subsumed(map_i_name, map_j_name):
                    continue
                map_i = copy.deepcopy(batch1[map_i_name])
                map_j = copy.deepcopy(batch2[map_j_name])
                operator = self.choose_operator(map_i, map_j, map_i_name, map_j_name)
                if operator is not None:
                    new_map = self.new_mapping(*operator)
                    # todo: compute metadata and use it for something?
                    #md = self.compute_metadata(new_map)
                    # metadata needs to be recorded given initial sources (we need to find mapping with highest fitness given same initial sources)
                    if self.is_fittest(new_map, [map_i_name, map_j_name]):
                        map_name = map_i_name + "_" + map_j_name
                        new_maps[map_name] = new_map
                        self.mapping_sources[map_name] = [map_i_name, map_j_name]
        return new_maps

    def mapping_subsumed(self, map1_name, map2_name):
        if (map1_name in self.mapping_sources and map2_name in self.mapping_sources[map1_name]) or \
            (map2_name in self.mapping_sources and map1_name in self.mapping_sources[map2_name]):
            return True
        return False

    def compute_metadata(self, new_map):
        # compute fitness data for new mapping
        fitness = self.fitness(new_map)
        # compute profile data for new mapping
        profile_data = self.compute_profile_data(new_map)
        pass
        # if operator == 'join':
        #     if self.lossy_merge_conditions_satisfied(map1, map2):
        #         metadata = self.compute_metadata_lossy(map1, map2)
        #     else:
        #         metadata = self.compute_metadata_lossless(map1, map2)
        # elif operator == 'full outer join':
        #     metadata = self.compute_metadata_lossless(map1, map2)
        # elif operator == 'union':
        #     metadata = self.compute_metadata_lossless(map1, map2)
        # return metadata

    def lossy_merge_conditions_satisfied(self, map1, map2):
        # When one of the relations may lose attribute values.
        return any([key not in map2.keys() for key in map1.keys()])

    def compute_metadata_lossless(self, map1, map2):
        pass

    def compute_metadata_lossy(self, map1, map2):
        pass

    def choose_operator(self, map1, map2, map1_name, map2_name):
        # Algorithm 3: Choose suitable merge operator.
        # t_rel is the target relation and it's a class variable (paper: global variable)
        map1_ma = self.find_matches_attr(map1)
        map2_ma = self.find_matches_attr(map2)
        operator = None
        attributes = None
        if self.diff_matches(map1_ma, map2_ma):
            operator = self.choose_operator_diff(map1, map2, map1_name, map2_name)
        else:
            operator = ("union", map1, map2, None)
        return operator

    def choose_operator_diff(self, map1, map2, map1_name, map2_name):
        # Algorithm 4: Generate operator when two mappings match different target attributes.
        # t_rel, pd (profile data) are class variables
        op = None
        subsumed_map = self.is_subsumed(map1, map2, map1_name, map2_name)
        if subsumed_map:
            return op
        map1_keys = self.find_keys(map1)
        map2_keys = self.find_keys(map2)
        ind_op = self.max_ind(map1, map2, map1_keys, map2_keys)
        if ind_op:
            # There is an overlap between candidate keys in map1 and map2
            return ind_op["op"]
        else:
            # There is no overlap between candidate keys in map1 and map2, try to find overlap between candidate keys in map1 with attributes in map2
            # and overlap between candidate keys in map2 with attributes in map1
            map1_ind_op = self.max_ind(map1, map2, map1_keys, list(map2.keys()))
            map2_ind_op = self.max_ind(map2, map1, map2_keys, list(map1.keys()))
            if map1_ind_op and map2_ind_op:
                if map1_ind_op["inclusion_ratio"] >= map2_ind_op["inclusion_ratio"]:
                    return map1_ind_op["op"]
                else:
                    return map2_ind_op["op"]
            elif map1_ind_op:
                return map1_ind_op["op"]
            elif map2_ind_op:
                return map2_ind_op["op"]
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
                map1_mk = self.find_matched_keys(map1)
                map2_mk = self.find_matched_keys(map2)
                if self.same_matches(map1_mk, map2_mk):
                    op = ("outer join", map1, map2, [map1_mk, map2_mk])
        return op

    def new_mapping(self, operator, map1, map2, attributes):
        # apply operator to two mappings
        if operator == "union":
            return self.op_union(map1, map2)
        elif operator == "join":
            return self.op_join(map1, map2, attributes)
        elif operator == "outer join":
            return self.op_outer_join(map1, map2, attributes)
        else:
            raise RuntimeError("unknown operator name")

    def op_union(self, map1, map2):
        df1 = pd.DataFrame.from_dict(map1)
        df2 = pd.DataFrame.from_dict(map2)
        # using pandas union
        unioned_dfs = pd.concat([df1, df2], axis=0, ignore_index=True)
        # getting unioned dfs back to dict format we use here
        union_map = unioned_dfs.to_dict(orient='list')
        return union_map

    def op_join(self, map1, map2, attributes):
        df1 = pd.DataFrame.from_dict(map1)
        df2 = pd.DataFrame.from_dict(map2)
        # using pandas merge to join on non-index columns
        joined_dfs = df1.merge(df2, on=attributes, suffixes=('', '_y')).drop('index_y', axis=1)
        # getting unioned dfs back to dict format we use here
        join_map = joined_dfs.to_dict(orient='list')
        return join_map

    def op_outer_join(self, map1, map2, attributes):
        df1 = pd.DataFrame.from_dict(map1)
        df2 = pd.DataFrame.from_dict(map2)
        # using pandas merge to join on non-index columns
        outer_joined_dfs =pd.merge(left=df2, right=df1, on=attributes, how='outer')
        # getting unioned dfs back to dict format we use here
        outer_join_map = outer_joined_dfs.to_dict(orient='list')
        return outer_join_map

    def is_fittest(self, map, source_names):
        # return fittest map from self.metadata (? algorithm 2)
        # isFittest checks if the intermediate mapping has the
        # highest fitness of any mapping involving the same initial sources,
        # if so, it is retained (alg 2 lines 9–10).
        # Compute the fitness of the mapping
        fitness = self.fitness(map)
        # Check if the mapping has the highest fitness of any mapping involving the same initial sources
        # If yes, update highest known fitness value
        if fitness > self.get_max_fitness(source_names):
            #for s in source_names:
            #    self.highest_fitness_value[s] = fitness
            source_1_name = source_names[0]
            source_2_name = source_names[1]
            if source_1_name in self.highest_fitness_value:
                self.highest_fitness_value[source_1_name][source_2_name] = fitness
            else:
                self.highest_fitness_value[source_1_name] = {source_2_name: fitness}
            if source_2_name in self.highest_fitness_value:
                self.highest_fitness_value[source_2_name][source_1_name] = fitness
            else:
                self.highest_fitness_value[source_2_name] = {source_1_name: fitness}
            return True
        return False

    def get_max_fitness(self, source_names):
        # Return the highest fitness of any mapping involving the same sources
        max_fitness = 0
        source_1_name = source_names[0]
        source_2_name = source_names[1]
        if source_1_name in self.highest_fitness_value and source_2_name in self.highest_fitness_value[source_1_name]:
            max_fitness = max(max_fitness, self.highest_fitness_value[source_1_name][source_2_name])
        if source_2_name in self.highest_fitness_value and source_1_name in self.highest_fitness_value[source_2_name]:
            max_fitness = max(max_fitness, self.highest_fitness_value[source_2_name][source_1_name])
        #for s in source_names:
        #    if s in self.highest_fitness_value:
        #        max_fitness = max(max_fitness, self.highest_fitness_value[s])
        return max_fitness

    def find_matches_attr(self, map):
        # For each mapping, get attributes that are in target
        matches = set()
        for attr, values in map.items():
            # todo: more intelligent ideas could be used here? e.g. jl distance or something?
            if attr in self.t_rel:
                matches.add(attr)
        return matches

    def diff_matches(self, set1, set2):
        # DiffMatches checks whether the matches are for different target attributes. If they are, they become candidates for joining,
        # to be decided by ChooseOperatorDiff (line 7). If the matches are for the same target attributes, then the two mappings are unioned (line 9).
        # are the matched attributes from map1 with t_rel (set1) the same as matched attributes from map2 with t_rel (set2)?
        return len(set1.symmetric_difference(set2)) == 0

    def is_subsumed(self, map1, map2, map1_name, map2_name):
        # In lines 4–7, IsSubsumed determines whether, on attributes
        # that match the target, the profiling data has inclusion dependencies between an attribute in one mapping and a corresponding
        # attribute in the other mapping. If so, the subsumed mapping is
        # discarded from the set of kept mappings (for further reference,
        # in Section 4.3, we refer to the kept mappings as memorized subsolutions) and null is returned.
        map1_ma = self.find_matches_attr(map1)
        map2_ma = self.find_matches_attr(map2)

        map1_subsumptions = {}
        for att in map1_ma:
            if att in map2:
                # todo: replace this with call to profile data (propagation method)
                inclusion = sum(el in map1[att] for el in map2[att])
                if inclusion == len(map1[att]):
                    # this att is subsumed in map2
                    map1_subsumptions[att] = True
                else:
                    map1_subsumptions[att] = False
            else:
                map1_subsumptions[att] = False
        # todo: all or any? text in paper sounds like "any", but then the two cases of subsumption map1 by map2 and the other way round wouldn't be symmetrical anymore
        if all(map1_subsumptions):
            # map1 is subsumed by map2
            self.remove_mapping(map1_name)
            return None

        map2_subsumptions = {}
        for att in map2_ma:
            if att in map1:
                inclusion = sum(el in map2[att] for el in map1[att])
                if inclusion == len(map2[att]):
                    # this att is subsumed in map2
                    map2_subsumptions[att] = True
                else:
                    map2_subsumptions[att] = False
            else:
                map2_subsumptions[att] = False
        if all(map2_subsumptions):
            # map2 is subsumed by map1
            self.remove_mapping(map2_name)
            return None

    def remove_mapping(self, map_name):
        for solution in self.sub_solution:
            if map_name in solution:
                del self.sub_solution[map_name]

    def find_keys(self, map):
        # todo: replace with call to profiling function
        # we want candidate keys here, i.e. attributes that are unique
        keys = set()
        for attr, values in map.items():
            if len(values) == len(set(values)):
                keys.add(attr)
        return keys

    def max_ind(self, map1, map2, map1_keys, map2_keys):
        # todo: replace this with call to profiling function
        # todo: in this function and "find_keys", we could replace these functions with pk/fk inferrence from paper fernandez2018
        map1_map2_inclusions = {}
        for key in map1_keys:
            if key in map2_keys:
                inclusion = sum(el in map1[key] for el in map2[key])
                inclusion_ratio = inclusion/len(map1[key])
                map1_map2_inclusions[key] = inclusion_ratio
        map2_map1_inclusions = {}
        for key in map2_keys:
            if key in map1_keys:
                inclusion = sum(el in map2[key] for el in map1[key])
                inclusion_ratio = inclusion / len(map2[key])
                map2_map1_inclusions[key] = inclusion_ratio

        if max(map1_map2_inclusions.values()) > max(map2_map1_inclusions.values()):
            max_inclusion = [(key, inclusion) for key, inclusion in map1_map2_inclusions.items() if
                             inclusion == max(map1_map2_inclusions.values())]
        else:
            max_inclusion = [(key, inclusion) for key, inclusion in map2_map1_inclusions.items() if
                             inclusion == max(map2_map1_inclusions.values())]

        if max(max_inclusion.values()) == 1:
            # if θ = 1.0, then the inclusion dependency is total and the
            # chosen operator is join because a foreign key relationship
            # is inferred between two mappings on their candidate key
            # attributes (lines 12–13);
            # if inclusion_ratio == 1 --> all elements from map1[key] are in map2[key], so we assume key is a pk for map1 and a fk for map2 (? foreign key relationship)
            op = ("join", map1, map2, list(max_inclusion.keys()))
            return {"op": op, "inclusion_ratio": 1}
        elif 0 <= max(max_inclusion.values()) < 1:
            # if θ ∈ (0, 1.0), then the inclusion dependency is partial
            # and the operator is a full outer join because a foreign key
            # relationship cannot be inferred so the algorithm joins the
            # tuples that can be joined and keeps the remaining data (lines
            # 14–15).
            op = ("outer join", map1, map2, list(max_inclusion.keys()))
            return {"op": op, "inclusion_ratio": max(max_inclusion.values())}
        else:
            return None

    def find_matched_keys(self, map):
        matched_keys = set()
        for att in list(map.keys()):
            if att in self.t_rel:
                matched_keys.add(att)
        return matched_keys

    def same_matches(self, map1_mk, map2_mk):
        return map1_mk == map2_mk

    def fitness(self, map):
        # Algorithm 5: Fitness function
        # todo: get cardinalities and null counts from polypheny statistics module
        atts = self.remove_outliers(map)
        null_counts = []
        for att in map:
            values = map[att]
            null_count = len([v for v in values if pd.isna(v)])
            null_counts.append(null_count)
        max_attr_nulls = max(null_counts)
        # The number of largely complete tuples in the mapping is estimated to be
        # the cardinality of the mapping minus the number of nulls in the attribute with the most nulls.
        return self.map_cardinality(map) - max_attr_nulls

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

        #null_counts = [count(a.values) for a in attributes]
        #median = statistics.median(null_counts)
        #iqr = statistics.quantiles(null_counts, n=4)[2] - statistics.quantiles(null_counts, n=4)[1]
        #lower_bound = median - 1.5 * iqr
        #upper_bound = median + 1.5 * iqr
        #return [a for a in attributes if lower_bound <= count(a.values) <= upper_bound]
        #return [a for a, null_count in zip(attributes, null_counts) if lower_bound <= null_count <= upper_bound]

    def attr_nulls(self, atts):
        # For atts, count nulls (? alg 5 line 3)
        null_count = 0
        for a in atts:
            if not a:
                null_count += 0
        return null_count

    def best_mappings(self):
        # In the current approach, Dynamap outputs the best k mappings where
        # k is an integer. The output mappings merge subsets of i source
        # relations, 1 ≤ i ≤ n, where n is the total number of input source
        # relations, which were obtained during the dynamic programming
        # search, and that are ranked according to their fitness.
        pass

    def main(self):
        self.generate_mappings(len(self.source_relations))

if __name__ == "__main__":
    # Create a dataframe with three columns: 'A', 'B', and 'C'
    df1 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
    # Create a dataframe with two columns: 'B' and 'C'
    df2 = pd.DataFrame({'B': [10, 11, 12], 'C': [13, 14, 15]})
    dfs = {"df1": {"columns": df1},
           "df2": {"columns": df2}}
    dfs = {"df1": {'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]},
           "df2": {'B': [10, 11, 12], 'C': [13, 14, 15]},
           "df3": {'A': [16, 17, 18], 'C': [19, 20, 21]}}
    # todo: check earlier that dfs source relations doesn't have duplicate keys
    matches = {(('df1', 'B'), ('df2', 'B')): 1,
               (('df1', 'C'), ('df2', 'C')): 1,
               (('df1', 'A'), ('df3', 'A')): 1,
               (('df1', 'A'), ('df3', 'C')): 1,
               (('df2', 'C'), ('df3', 'C')): 1}

    dynamap = pyDynaMap(dfs, matches)
    dynamap.generate_mappings(len(dfs))
    for solution in dynamap.sub_solution:
        print(dynamap.sub_solution[solution])
