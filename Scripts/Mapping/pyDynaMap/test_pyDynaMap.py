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

class public_polypheny_data(unittest.TestCase):
    def setUp(self) -> None:
        dfs = {"depts": {'deptno': ['10', '20', '30'], 'name': ['Sales', 'Marketing', 'HR']},
               "emps": {'empid': ['100', '110', '150', '200'], 'deptno': ['10', '10', '20', '30'],
                        'name': ['Bill', 'Theodore', 'Sebastian', 'Eric'], 'salary': ['10000', '11500', '7000', '8000'],
                        'commission': ['1000', '250', '400', '500']},
               "emp": {'employeeno': ['1', '2', '4', '5', '7'], 'age': ['41', '49', '37', '33', '27'],
                       'gender': ['Female', 'Male', 'Male', 'Female', 'Male'],
                       'maritalstatus': ['Single', 'Married', 'Single', 'Married', 'Married'],
                       'worklifebalance': ['Bad', 'Better', 'Better', 'Better', 'Better'],
                       'education': ['College', 'Below College', 'College', 'Master', 'Below College'],
                       'monthlyincome': ['5993', '5130', '2090', '2909', '3468'],
                       'relationshipjoy': ['Low', 'Very High', 'Medium', 'High', 'Very High'],
                       'workingyears': ['8', '10', '7', '8', '6'], 'yearsatcompany': ['6', '10', '0', '8', '2']},
               "work": {'employeeno': ['1', '2', '4', '5', '7'],
                        'educationfield': ['Life Sciences', 'Life Sciences', 'Other', 'Life Sciences', 'Medical'],
                        'jobinvolvement': ['High', 'Medium', 'Medium', 'High', 'High'],
                        'joblevel': ['2', '2', '1', '1', '1'],
                        'jobrole': ['Sales Executive', 'Research Scientist', 'Laboratory Technician',
                                    'Research Scientist',
                                    'Laboratory Technician'],
                        'businesstravel': ['Travel_Rarely', 'Travel_Frequently', 'Travel_Rarely', 'Travel_Frequently',
                                           'Travel_Rarely'],
                        'department': ['Sales', 'Research & Development', 'Research & Development',
                                       'Research & Development', 'Research & Development'],
                        'attrition': ['Yes', 'No', 'Yes', 'No', 'No'],
                        'dailyrate': ['1102', '279', '1373', '1392', '591']}
               }
        matches = {(('depts', 'deptno'), ('emps', 'deptno')): 1,
                   (('depts', 'name'), ('emps', 'name')): 1,
                   (('emp', 'employeeno'), ('work', 'employeeno')): 1}
        self.dynamap = pyDynaMap(dfs, matches)
        self.dynamap.generate_mappings(len(dfs))
    def test_complicated_mapping_case_1(self):
        self.assertIn("emp_work", self.dynamap.k_best_mappings(10))
        self.assertIn("work_emp", self.dynamap.k_best_mappings(10))
        self.assertEqual(self.dynamap.k_best_mappings(3)["emp_work"]["mapping_path"][0], "left join")


