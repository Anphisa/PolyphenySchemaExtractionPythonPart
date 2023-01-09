import graphviz
from graphviz import Digraph
from Helper.FieldRelationship import FieldRelationship

class SchemaCandidateVisualization():
    def __init__(self,
                 n_schema_candidate: int,
                 mapping_name: str,
                 mapping_path: list,
                 mapping_result: dict,
                 sample_length: int,
                 field_loss: list, instance_loss: list, structure_loss: list):
        self.n_schema_candidate = str(n_schema_candidate)
        self.mapping_name = mapping_name
        self.mapping_path = mapping_path
        self.mapping_result = mapping_result
        self.sample_length = sample_length
        self.field_loss = field_loss
        self.instance_loss = instance_loss
        self.structure_loss = structure_loss
        self.g = graphviz.Digraph('G', filename="output/schemacandidatetest.svg")

    def escape_special_characters(self, string):
        escaped = string.replace("&", "&amp;")
        return escaped

    def draw(self):
        self.g.attr(labelloc="t")
        self.g.attr(label="Schema candidate " + self.n_schema_candidate)
        self.g.attr(rankdir="TB")

        # table combinations in mapping
        # example mapping: ['union', 'df1', ['outer join', 'df2', 'df3', ['C']], None]
        with self.g.subgraph(name='cluster_mappings') as t:
            t.attr(label='Proposed mapping operations')
            seen_submappings = []
            def draw_mapping_path(mapping_path, t):
                if len(mapping_path) == 4:
                    rel = mapping_path[0]
                    from_table = mapping_path[1]
                    to_table = mapping_path[2]
                    cols = mapping_path[3]
                    if not (isinstance(from_table, list) or isinstance(to_table, list)):
                        #pk_from_table = ", ".join(self.source_relations[from_table]["pk"])
                        #pk_to_table = ", ".join(self.source_relations[to_table]["pk"])
                        t.node(from_table)
                        t.node(to_table)
                        t.node(str(mapping_path), label=rel + "\n" + str(cols))
                        t.edge(from_table, str(mapping_path))
                        t.edge(to_table, str(mapping_path))
                        #seen_submappings.append(str(mapping_path))
                    elif isinstance(from_table, list) and isinstance(to_table, list):
                        draw_mapping_path(from_table, t)
                        draw_mapping_path(to_table, t)
                        t.node(str(mapping_path), label=rel + "\n" + str(cols))
                        t.edge(str(from_table), str(mapping_path))
                        t.edge(str(to_table), str(mapping_path))
                    elif isinstance(from_table, list):
                        draw_mapping_path(from_table, t)
                        t.node(str(mapping_path), label=rel + "\n" + str(cols))
                        t.edge(str(from_table), str(mapping_path))
                        t.edge(str(to_table), str(mapping_path))
                    elif isinstance(to_table, list):
                        draw_mapping_path(to_table, t)
                        t.node(str(mapping_path), label=rel + "\n" + str(cols))
                        t.edge(str(from_table), str(mapping_path))
                        t.edge(str(to_table), str(mapping_path))
                elif len(mapping_path) == 1:
                    # todo: this means that we do not propose any mapping combinations at all. just return source tables
                    return
            draw_mapping_path(self.mapping_path, t)

        # Draw sample from proposed schema, so we can see some content that rows would have
        # todo: this should also display multiple tables if mapping returns multiple tables (this could happen with dynamap, although currently it doesn't)
        with self.g.subgraph(name='cluster_proposed_schema') as t:
            t.attr(label='Proposed schema sample data')
            table_name = self.mapping_name
            col_names = [self.escape_special_characters(c) for c in list(self.mapping_result.keys())]
            with t.subgraph(name='cluster_' + table_name) as tt:
                tt.attr(style='filled', color='lightgrey')
                tt.attr(label=table_name)
                table_content = """<<table border="0" cellborder="1" cellspacing="0"> <tr>  <td>""" + "</td><td>".join(col_names) + " </td>  </tr>"
                for i in range(0, self.sample_length):
                    tables_at_i = []
                    for table in self.mapping_result:
                        tables_at_i.append(self.escape_special_characters(str(self.mapping_result[table][i])))
                    table_content += "<tr> <td>" + "</td><td>".join(tables_at_i) + "</td></tr>"
                print(table_content + "</table>>")
                tt.node("blub", label=table_content + "</table>>", shape="box")

        # field loss, instance loss, structure loss
        with self.g.subgraph(name='cluster_loss') as l:
            l.attr(label='Loss under proposed schema')
            label = "{ " + self.field_loss + " | " + self.instance_loss + " | " + self.structure_loss + " }"
            l.node("loss records", shape="record", label=label)

        return self.g

if __name__ == "__main__":
    s = SchemaCandidateVisualization(0,
                                     'df2_df3_df1',
                                     ['union', 'df1', ['outer join', 'df2', 'df3', ['C']], None],
                                     {'B': [10.0, 11.0, 12.0, 4.0, 5.0, 6.0, None, None, None],
                                      'C': [13, 14, 15, 7, 8, 9, 19, 20, 21],
                                      'A': [None, None, None, 1.0, 2.0, 3.0, 16.0, 17.0, 18.0]},
                                      "Field loss: None.",
                                      "Instance loss: None.",
                                      "Structure loss: None.")
                                     #"Field loss: depts: loss of fields [1, 2, 3]",
                                     #"Instance loss: 95% of bla",
                                     #"Structure loss: None.")
    g = s.draw()
    g.view()