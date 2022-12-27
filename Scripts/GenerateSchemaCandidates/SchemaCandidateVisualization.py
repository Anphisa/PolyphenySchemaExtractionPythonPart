import graphviz
from graphviz import Digraph
from Helper.FieldRelationship import FieldRelationship

class SchemaCandidateVisualization():
    def __init__(self,
                 n_schema_candidate: int,
                 operations: dict,
                 new_tables: dict, table_relationships: list,
                 field_loss: list, instance_loss: list, structure_loss: list):
        self.n_schema_candidate = str(n_schema_candidate)
        self.operations = operations
        self.new_tables = new_tables
        self.table_relationships = table_relationships
        self.field_loss = field_loss
        self.instance_loss = instance_loss
        self.structure_loss = structure_loss
        self.g = graphviz.Digraph('G', filename="output/schemacandidatetest.svg")

    def draw(self):
        self.g.attr(labelloc="t")
        self.g.attr(label="Schema candidate " + self.n_schema_candidate)
        self.g.attr(rankdir="TB")

        rank_hack_nodes = []

        # operations: manipulations that would have to be taken on data
        with self.g.subgraph(name='cluster_operations') as c:
            #c.attr(style='filled', color='lightgrey')
            #c.node_attr.update(style='filled', color='white')
            c.attr(label='Operations')
            for i, op in enumerate(self.operations):
                instruction = op["operation"]
                explanation = op["explanation"]
                with c.subgraph(name="cluster_op_" + str(i)) as cc:
                    label = "{{ Explanation: {} }}".format(explanation)
                    #print(label)
                    cc.node("op_" + str(i), shape="record", label=label)
                    cc.attr(label=instruction)
                    cc.attr(style='filled', color='lightgrey')
                if i == len(self.operations) - 1:
                    rank_hack_nodes.append("op_" + str(i))
            # todo: force top to bottom order of operations
            #if len(self.operations) >= 2:
                # force top to bottom order for operations
             #   c.edge("op_instruction_0", "op_instruction_1", style="invis")

        # new tables (or data structures), i.e. new schema visualized
        first_table_name = ""
        with self.g.subgraph(name='cluster_tables') as t:
            t.attr(label='Proposed schema')
            for j, table in enumerate(self.new_tables):
                if self.new_tables[table]["datamodel"].lower() != "relational":
                    raise NotImplementedError("only implemented for relational data!")
                table_name = table
                pk = ", ".join(self.new_tables[table]["pk"])
                if first_table_name == "":
                    first_table_name = table_name
                    rank_hack_nodes.append(table_name + "_columns:" + table_name + "_" + pk)
                #print("str_columns", [c[0] for c in self.new_tables[table]["columns"].columns.values])
                col_names = [c[0] for c in self.new_tables[table]["columns"].columns.values]
                print("col_names", col_names)
                #str_columns = "\n".join(col_names)
                with t.subgraph(name='cluster_' + table_name) as tt:
                    tt.attr(style='filled', color='lightgrey')
                    tt.attr(label=table_name)
                    #tt.node(table_name + "_" + pk, label="primary key: " + pk + "\n", shape="box")
                    with tt.subgraph(name='cluster_' + table_name + "_columns") as ttt:
                        ttt.attr(label="")
                        inner_label = ""
                        pk_name = ""
                        for c in col_names:
                            if c in self.new_tables[table]["pk"]:
                                c_name = c + " (pk)"
                                pk_name = c
                            else:
                                c_name = c
                            inner_label += "<{}>{} |".format(table_name + "_" + c, c_name)
                        if j == len(self.new_tables) - 1 and pk_name != "":
                            rank_hack_nodes.append(table_name + "_columns:" + table_name + "_" + pk_name)
                        print("table", table_name, "inner label", inner_label)
                        #inner_label = " | ".join(col_names)
                        label = "{ " + inner_label[:-1] + " }"
                        print("table", table_name, "whole label", label)
                        ttt.node(table_name + "_columns", shape="record", label=label)
                        #for i, col in enumerate(col_names):
                            #ttt.node(table_name + "_" + col, label=col)
                            #if i == len(self.new_tables[table]["columns"]) - 1 and j == len(self.new_tables) - 1:
                            #    rank_hack_nodes.append(table_name + "_" + col)
                    #tt.edge(table_name + "_" + pk, table_name + "_columns", style="invis")

            # relationships between data fields
            for relationship in self.table_relationships:
                rel = FieldRelationship(relationship["from_table"],
                                        relationship["to_table"],
                                        relationship["from_column"],
                                        relationship["to_column"],
                                        relationship["relationship"],
                                        relationship["relationship_strength"])
                #t.edge(relationship["from_table"] + "_" + relationship["from_column"],
                #       relationship["to_table"] + "_" + relationship["to_column"],
                #       label=relationship["relationship"])
                t.edge(rel.table1 + "_columns:" + rel.table1 + "_" + rel.field_name1,
                       rel.table2 + "_columns:" + rel.table2 + "_" + rel.field_name2,
                       label=rel.relationship_type)
                #print("edge info: ", (rel.table1 + "_" + rel.field_name1,
                #       rel.table2 + "_" + rel.field_name2,
                #       rel.relationship_type))

        # todo: force top to bottom order of subgraphs. next line doesn't work
        #self.g.edge("op_" + str(len(self.operations)), 'cluster_' + first_table_name)

        # field loss, instance loss, structure loss
        with self.g.subgraph(name='cluster_loss') as l:
            l.attr(label='Loss under proposed schema')
            # inner_label = " | ".join(self.new_tables[table]["columns"])
            # label = "{{ " + inner_label + " }}"
            label = "{ " + self.field_loss + " | " + self.instance_loss + " | " + self.structure_loss + " }"
            # print(label)
            l.node("loss records", shape="record", label=label)
            #l.node("field loss", label=self.field_loss, shape="box")
            #l.node("instance loss", label=self.instance_loss, shape="box")
            #l.node("structure loss", label=self.structure_loss, shape="box")

        #rank_hack_nodes += ["field loss", "instance loss", "structure loss"]
        rank_hack_nodes.append("loss records")
        #print(rank_hack_nodes)
        k = 0
        while k < len(rank_hack_nodes) - 1:
            self.g.edge(rank_hack_nodes[k], rank_hack_nodes[k+1], style="invis")
            k += 1
        return self.g
        #self.g.view()

if __name__ == "__main__":
    s = SchemaCandidateVisualization(0,
                                     [{"operation" : "left join: dept:deptid <-> emps:deptid",
                                       "explanation": "JL col names only matcher: 95% match"}],
                                      {"dept": {"datamodel": "relational",
                                                "pk": "deptid",
                                                "columns": ["deptname", "deptblub"]},
                                       "emps": {"datamodel": "relational",
                                                "pk": "empid",
                                                "columns": ["deptid", "empname"]}},
                                     [{"from_table": "dept",
                                       "to_table": "emps",
                                       "from_column": "deptid",
                                       "to_column": "deptid",
                                       "relationship": "left join"}],
                                     "Field loss: depts: loss of fields [1, 2, 3]",
                                     "Instance loss: 95% of bla",
                                     "Structure loss: None.")
    g = s.draw()
    g.view()