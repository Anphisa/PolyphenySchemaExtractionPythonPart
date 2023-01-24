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

class onecol_samename_overlappingcontent_2(unittest.TestCase):
    def setUp(self) -> None:
        dfs = {"df1": {'id': [1, 2]},
               "df2": {'id': [1, 2, 3]}}
        matches = {(('df1', 'id'), ('df2', 'id')): 1}
        self.dynamap = pyDynaMap(dfs, matches)
        self.dynamap.generate_mappings(len(dfs))
    def test_onecol_samename_overlappingcontent_2(self):
        self.assertIn("df1_df2", self.dynamap.k_best_mappings(3))
        self.assertEqual(self.dynamap.k_best_mappings(3)["df1_df2"]["mapping_path"][0], "left join")

class onecol_samename_differentcontent(unittest.TestCase):
    def setUp(self) -> None:
        dfs = {"df1": {'id': [1, 2, 3]},
               "df2": {'id': [4, 5, 6]}}
        matches = {(('df1', 'id'), ('df2', 'id')): 1}
        self.dynamap = pyDynaMap(dfs, matches)
        self.dynamap.generate_mappings(len(dfs))
    def test_onecol_samename_differentcontent(self):
        self.assertIn("df1_df2", self.dynamap.k_best_mappings(3))
        self.assertEqual(self.dynamap.k_best_mappings(3)["df1_df2"]["mapping_path"][0], "union")

class onecol_differentname_samecontent(unittest.TestCase):
    def setUp(self) -> None:
        dfs = {"df1": {'id': [1, 2, 3]},
               "df2": {'id2': [1, 2, 3]}}
        matches = {(('df1', 'id'), ('df2', 'id2')): 1}
        self.dynamap = pyDynaMap(dfs, matches)
        self.dynamap.generate_mappings(len(dfs))
    def test_onecol_differentname_samecontent(self):
        self.assertIn("df1_df2", self.dynamap.k_best_mappings(3))
        self.assertEqual(self.dynamap.k_best_mappings(3)["df1_df2"]["mapping_path"][0], "left join")

class onecol_differentname_overlappingcontent_1(unittest.TestCase):
    def setUp(self) -> None:
        dfs = {"df1": {'id': [1, 2]},
               "df2": {'id2': [1, 2, 3]}}
        matches = {(('df1', 'id'), ('df2', 'id2')): 1}
        self.dynamap = pyDynaMap(dfs, matches)
        self.dynamap.generate_mappings(len(dfs))
    def test_onecol_samename_overlappingcontent_1(self):
        self.assertIn("df1_df2", self.dynamap.k_best_mappings(3))
        self.assertEqual(self.dynamap.k_best_mappings(3)["df1_df2"]["mapping_path"][0], "left join")

class onecol_differentname_overlappingcontent_2(unittest.TestCase):
    def setUp(self) -> None:
        dfs = {"df1": {'id': [1, 2, 3]},
               "df2": {'id2': [1, 2]}}
        matches = {(('df1', 'id'), ('df2', 'id2')): 1}
        self.dynamap = pyDynaMap(dfs, matches)
        self.dynamap.generate_mappings(len(dfs))
    def test_onecol_differentname_overlappingcontent_2(self):
        self.assertIn("df2_df1", self.dynamap.k_best_mappings(3))
        self.assertEqual(self.dynamap.k_best_mappings(3)["df2_df1"]["mapping_path"][0], "left join")

class twocol_samename_samecontent(unittest.TestCase):
    def setUp(self) -> None:
        dfs = {"df1": {'id': [1, 2, 3],
                       'name': ['A', 'B', 'C']},
               "df2": {'id': [1, 2, 3],
                       'name': ['A', 'B', 'C']}}
        matches = {(('df1', 'id'), ('df2', 'id')): 1,
                   (('df1', 'name'), ('df2', 'name')): 1}
        self.dynamap = pyDynaMap(dfs, matches)
        self.dynamap.generate_mappings(len(dfs))
    def test_twocol_samename_samecontent(self):
        self.assertIn("df1_df2", self.dynamap.k_best_mappings(3))
        self.assertEqual(self.dynamap.k_best_mappings(3)["df1_df2"]["mapping_path"][0], "left join")

class twocol_samename_onesamecontent(unittest.TestCase):
    def setUp(self) -> None:
        dfs = {"df1": {'id': [1, 2, 3],
                       'name': ['A', 'B', 'C']},
               "df2": {'id': [1, 2, 3],
                       'name': ['D', 'E', 'F']}}
        matches = {(('df1', 'id'), ('df2', 'id')): 1,
                   (('df1', 'name'), ('df2', 'name')): 1}
        self.dynamap = pyDynaMap(dfs, matches)
        self.dynamap.generate_mappings(len(dfs))
    def test_twocol_samename_onesamecontent(self):
        self.assertIn("df1_df2", self.dynamap.k_best_mappings(3))
        self.assertEqual(self.dynamap.k_best_mappings(3)["df1_df2"]["mapping_path"][0], "union")

class twocol_samename_onesamecontent_2(unittest.TestCase):
    def setUp(self) -> None:
        dfs = {"df1": {'id': [1, 2, 3],
                       'name': ['A', 'B', 'C']},
               "df2": {'id': [4, 5, 6],
                       'name': ['A', 'B', 'C']}}
        matches = {(('df1', 'id'), ('df2', 'id')): 1,
                   (('df1', 'name'), ('df2', 'name')): 1}
        self.dynamap = pyDynaMap(dfs, matches)
        self.dynamap.generate_mappings(len(dfs))
    def test_twocol_samename_onesamecontent_2(self):
        self.assertIn("df1_df2", self.dynamap.k_best_mappings(3))
        self.assertEqual(self.dynamap.k_best_mappings(3)["df1_df2"]["mapping_path"][0], "union")


class twocol_samename_overlapping_content_1(unittest.TestCase):
    def setUp(self) -> None:
        dfs = {"df1": {'id': [1, 2, 3],
                       'name': ['A', 'B', 'C']},
               "df2": {'id': [3, 4, 5],
                       'name': ['A', 'B', 'C']}}
        matches = {(('df1', 'id'), ('df2', 'id')): 1,
                   (('df1', 'name'), ('df2', 'name')): 1}
        self.dynamap = pyDynaMap(dfs, matches)
        self.dynamap.generate_mappings(len(dfs))
    def test_twocol_samename_overlapping_content_1(self):
        self.assertIn("df1_df2", self.dynamap.k_best_mappings(3))
        self.assertEqual(self.dynamap.k_best_mappings(3)["df1_df2"]["mapping_path"][0], "union")
        # not a (left) join. while name is included, id is not. id is in t_rel though, so we can't join, we would have a
        # double ID column and could not uniquely say what the local alias for the id column is in mapping result.
        self.assertEqual(self.dynamap.k_best_mappings(3)["df2_df1"]["mapping_path"][0], "union")

class twocol_samename_overlapping_content_2(unittest.TestCase):
    def setUp(self) -> None:
        dfs = {"df1": {'id': [1, 2, 3],
                       'name': ['A', 'B', 'C']},
               "df2": {'id': [1, 2, 3],
                       'name': ['A', 'B', 'D']}}
        matches = {(('df1', 'id'), ('df2', 'id')): 1,
                   (('df1', 'name'), ('df2', 'name')): 1}
        self.dynamap = pyDynaMap(dfs, matches)
        self.dynamap.generate_mappings(len(dfs))
    def test_twocol_samename_overlapping_content_2(self):
        self.assertIn("df1_df2", self.dynamap.k_best_mappings(3))
        self.assertEqual(self.dynamap.k_best_mappings(3)["df1_df2"]["mapping_path"][0], "union")
        # not a (left) join. while name is included, id is not. id is in t_rel though, so we can't join, we would have a
        # double ID column and could not uniquely say what the local alias for the id column is in mapping result.
        self.assertEqual(self.dynamap.k_best_mappings(3)["df2_df1"]["mapping_path"][0], "union")

class twocol_samename_overlapping_content_3(unittest.TestCase):
    def setUp(self) -> None:
        dfs = {"df1": {'id': [1, 2, 3],
                       'name': ['A', 'B', 'C']},
               "df2": {'id': [10, 2, 3],
                       'name': ['A', 'B', 'D']}}
        matches = {(('df1', 'id'), ('df2', 'id')): 1,
                   (('df1', 'name'), ('df2', 'name')): 1}
        self.dynamap = pyDynaMap(dfs, matches)
        self.dynamap.generate_mappings(len(dfs))
    def test_twocol_samename_overlapping_content_3(self):
        self.assertIn("df1_df2", self.dynamap.k_best_mappings(3))
        self.assertEqual(self.dynamap.k_best_mappings(3)["df1_df2"]["mapping_path"][0], "union")
        # not a (left) join. while name is included, id is not. id is in t_rel though, so we can't join, we would have a
        # double ID column and could not uniquely say what the local alias for the id column is in mapping result.
        self.assertEqual(self.dynamap.k_best_mappings(3)["df2_df1"]["mapping_path"][0], "union")

class twocol_onesamename_samecontent(unittest.TestCase):
    def setUp(self) -> None:
        dfs = {"df1": {'id': [1, 2, 3],
                       'name': ['A', 'B', 'C']},
               "df2": {'id2': [1, 2, 3],
                       'name': ['A', 'B', 'C']}}
        matches = {(('df1', 'id'), ('df2', 'id2')): 1,
                   (('df1', 'name'), ('df2', 'name')): 1}
        self.dynamap = pyDynaMap(dfs, matches)
        self.dynamap.generate_mappings(len(dfs))
    def test_twocol_onesamename_samecontent(self):
        self.assertIn("df1_df2", self.dynamap.k_best_mappings(3))
        self.assertEqual(self.dynamap.k_best_mappings(3)["df1_df2"]["mapping_path"][0], "left join")

class twocol_differentnames_samecontent(unittest.TestCase):
    def setUp(self) -> None:
        dfs = {"df1": {'id': [1, 2, 3],
                       'name': ['A', 'B', 'C']},
               "df2": {'id2': [1, 2, 3],
                       'name2': ['A', 'B', 'C']}}
        matches = {(('df1', 'id'), ('df2', 'id2')): 1,
                   (('df1', 'name'), ('df2', 'name2')): 1}
        self.dynamap = pyDynaMap(dfs, matches)
        self.dynamap.generate_mappings(len(dfs))
    def test_twocol_differentnames_samecontent(self):
        self.assertIn("df1_df2", self.dynamap.k_best_mappings(3))
        self.assertEqual(self.dynamap.k_best_mappings(3)["df1_df2"]["mapping_path"][0], "left join")

class subsumption(unittest.TestCase):
    def setUp(self) -> None:
        dfs = {"df1": {'id': [1, 2, 3],
                       "name": [None, None, "B"]},
               "df2": {'id2': [1, 2, 3]},
               "df3": {"name": ["A", "B", "C"]}}
        matches = {(('df1', 'id'), ('df2', 'id2')): 1,
                   (('df1', 'name'), ('df3', 'name')): 1}
        self.dynamap = pyDynaMap(dfs, matches)
        self.dynamap.generate_mappings(len(dfs))
    def test_subsumption(self):
        # df2 is subsumed by df1. It should be deleted from self.subsolutions[0]
        self.assertNotIn("df1_df2", self.dynamap.k_best_mappings(10))

class complicated_mapping_case_1(unittest.TestCase):
    def setUp(self) -> None:
        dfs = {"df1": {'name': ["Leon", "Albert", "Pupsbanane"]},
               "df2": {'firstname': ["Leon", "Albert", "Pupsbanane"]},
               "df3": {'name2': ["Batista", "Einstein", "Banane"]}}
        matches = {(('df1', 'name'), ('df2', 'firstname')): 1,
                   (('df1', 'name'), ('df3', 'name2')): 1,
                   (('df2', 'firstname'), ('df3', 'name2')): 1}
        self.dynamap = pyDynaMap(dfs, matches)
        self.dynamap.generate_mappings(len(dfs))
    def test_complicated_mapping_case_1(self):
        self.assertIn("df1_df3_df2", self.dynamap.k_best_mappings(10))
        self.assertIn("df1_df2_df3", self.dynamap.k_best_mappings(10))
