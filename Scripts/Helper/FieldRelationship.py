class FieldRelationship():
    # Helper class: Two fields (e.g. columns) and their relationship
    def __init__(self, table1, table2, field_name1, field_name2, relationship_type=None, relationship_strength=None):
        self.table1 = table1
        self.table2 = table2
        self.field_name1 = field_name1
        self.field_name2 = field_name2
        if type(relationship_type) == tuple:
            self.relationship_type = relationship_type[0]
        else:
            self.relationship_type = relationship_type
        self.relationship_strength = relationship_strength

    def __repr__(self):
        return(str({"object": "FieldRelationship",
                "table1": self.table1,
                "table2": self.table2,
                "field_name1": self.field_name1,
                "field_name2": self.field_name2,
                "relationship_type": self.relationship_type,
                "relationship_strength": self.relationship_strength}))

    def table_check(self, table1, table2):
        if self.table1 == table1 and self.table2 == table2:
            return True
        return False