import unittest
from unittest import TestCase
from pyDynaMap import pyDynaMap

class onecol_samename_samecontent(unittest.TestCase):
    def setUp(self) -> None:
        dfs = {"df1": {'id': [1, 2, 3]},
               "df2": {'id': [1, 2, 3]}}
        matches = {(('df1', 'id'), ('df2', 'id')): 1}
        self.dynamap = pyDynaMap(dfs, matches)
        self.dynamap.generate_mappings(len(dfs))
    def test_onecol_samename_samecontent(self):
        self.assertIn("df1_df2", self.dynamap.k_best_mappings(3))
        self.assertEqual(self.dynamap.k_best_mappings(3)["df1_df2"]["mapping_path"][0], "left join")

class onecol_samename_overlappingcontent_1(unittest.TestCase):
    def setUp(self) -> None:
        dfs = {"df1": {'id': [1, 2, 3]},
               "df2": {'id': [1, 2]}}
        matches = {(('df1', 'id'), ('df2', 'id')): 1}
        self.dynamap = pyDynaMap(dfs, matches)
        self.dynamap.generate_mappings(len(dfs))
    def test_onecol_samename_overlappingcontent_1(self):
        self.assertIn("df2_df1", self.dynamap.k_best_mappings(3))
        self.assertEqual(self.dynamap.k_best_mappings(3)["df2_df1"]["mapping_path"][0], "left join")



class TestUnequalColumnNamesInMatches(unittest.TestCase):
    def setUp(self) -> None:
        dfs = {"df1": {'name': [1, 2, 3]},
               "df2": {'firstname': [10, 11, 12]},
               "df3": {'name2': [16, 17, 18]}}
        matches = {(('df1', 'name'), ('df2', 'firstname')): 1,
                   (('df1', 'name'), ('df3', 'name2')): 1,
                   (('df2', 'firstname'), ('df3', 'name2')): 1}
        self.dynamap = pyDynaMap(dfs, matches)



class TestpyDynaMap(TestCase):
    def test_choose_column_name(self):
        self.fail()

    def test_target_relation_from_matches(self):
        self.fail()
