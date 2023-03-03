from Helper.FieldRelationship import FieldRelationship
from Mapping.pyDynaMap.pyDynaMap import pyDynaMap

class Mapping:
    def __init__(self, source_relations):
        # print(source_relations)
        # A list of involved tables and at least their column names
        # example for source_relations:
        # dfs = {"df1": {'id': [1, 2, 3],
        #                'name': ['A', 'B', 'C']},
        #        "df2": {'id2': [1, 2, 3],
        #                'name': ['A', 'B', 'C']}}
        self.source_relations = source_relations

    @staticmethod
    def matchesAboveThreshold(matches, threshold):
        # given a list of matches from schema matcher and a threshold, return all matches that are above the threshold
        return {k:v for (k, v) in matches.items() if matches[k] >= threshold}

    def pyDynaMapMapping(self, matches, show_n_best_mappings):
        for rel in self.source_relations:
            # todo: test this with duplicate table names
            df_names = self.source_relations[rel].keys()
            if len(df_names) > len(set(df_names)):
                raise RuntimeError("pyDynaMap doesn't accept duplicate table names. "
                                   "Duplicate table names found in source_relations: ", self.source_relations)
        dyna_source_relations = {}
        for rel in self.source_relations:
            dyna_source_relations[rel] = self.source_relations[rel]["columns"].to_dict(orient="list")
        dynamap = pyDynaMap(dyna_source_relations, matches)
        dynamap.generate_mappings(len(dyna_source_relations))
        target_relation = list(dynamap.t_rel.keys())
        k_best_mappings = dynamap.k_best_mappings(show_n_best_mappings)
        # instance loss
        instance_loss_k_best_mappings = {}
        renamed_columns = {}
        for mapping in k_best_mappings:
            instance_loss_k_best_mappings[mapping] = dynamap.instance_loss_for_mapping_name(mapping)
            renamed_columns_mapping = dynamap.renamed_columns_for_mapping(mapping)
            if renamed_columns_mapping:
                renamed_columns[mapping] = renamed_columns_mapping
        # field loss
        field_loss = dynamap.field_loss
        # explanations
        explanations = {}
        for mapping in k_best_mappings:
            explanations[mapping] = dynamap.explain_mapping_operations_for_mapping_name(mapping)
        return({"target_relation": target_relation,
                "k_best_mappings": k_best_mappings,
                "instance_loss": instance_loss_k_best_mappings,
                "field_loss": field_loss,
                "renamed_columns": renamed_columns,
                "explanations": explanations})