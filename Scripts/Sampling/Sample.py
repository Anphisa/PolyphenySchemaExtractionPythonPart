# Given column name and table name, take a sample
import requests
import logging
import ast
import random

class Sample():
    def __init__(self, host, port, column: str, namespace: str, table: str, sample_size: int, num_rows=0, random=False):
        """Take a sample of values in a column.

        :param host: host of Polypheny Server
        :param port: port of Polypheny Server
        :param column: Column we want to sample
        :param namespace: Namespace that we can find the table in
        :param table: Table we want to sample from
        :param sample_size: size of sample. 0 if unlimited!
        :param num_rows: Number of rows in table (important for randomization, if random == True)
        :param random: Should sample be randomized?
        """
        self.URL = host + ":" + port + "/query"
        self.column_name = column
        self.namespace = namespace
        self.table_name = table
        self.sample_size = sample_size
        self.random = random
        self.num_rows = num_rows
        self.sample = None
        self.flat_sample = None

    def set_num_rows(self):
        # todo: what happens here for graph/document data model?
        query = "SELECT COUNT(*) FROM " + self.namespace + ".\"" + self.table_name + "\""
        DATA = {'querylanguage': 'SQL',
                'query': query}
        logging.info("Sampling values from column: " + query)
        result = requests.post(url=self.URL, data=DATA)
        num_rows = result.content.decode('utf-8')[2:-2]
        num_rows = int(num_rows) - 1
        self.num_rows = num_rows

    def take_sample_relational(self, column_name, namespace, table_name, random_sample, sample_size):
        query = "SELECT " + column_name + " FROM " + namespace + ".\"" + table_name + "\""
        if random_sample:
            offset = random.randint(0, self.num_rows)
            query += " LIMIT 1 OFFSET " + str(offset) + " ROWS"
        elif sample_size != 0:
            query += " LIMIT " + str(self.sample_size)
        DATA = {'querylanguage': 'SQL',
                'query': query}
        logging.info("Sampling values from column: " + query)
        result = requests.post(url=self.URL, data=DATA)
        return result

    def take_samples_relational(self):
        results = []
        if self.random:
            if self.sample_size != 0:
                for n in range(self.sample_size):
                    result = self.take_sample_relational(self.column_name, self.namespace, self.table_name, self.random, 0)
                    results.append(result)
            else:
                logging.warn("Can't sample randomly if sample size is 0. Taking un-random sample.")
                result = self.take_sample_relational(self.column_name, self.namespace, self.table_name, False, self.sample_size)
                results.append(result)
        else:
            result = self.take_sample_relational(self.column_name, self.namespace, self.table_name, False, self.sample_size)
        results.append(result)
        return results

    def take_sample_graph(self):
        #SELECT * FROM g."label"
        logging.error("Graph sampling doesn't work yet!!")
        query = "SELECT * FROM " + self.namespace + ".\"" + self.column_name + "\""
        if self.sample_size != 0:
            query += " LIMIT " + str(self.sample_size)
        DATA = {'querylanguage': 'SQL',
                'query': query}
        logging.info("Sampling values from column: " + query)
        result = requests.post(url=self.URL, data=DATA)
        self.http_sample_result = result

    def take_sample_document(self):
        # SELECT * FROM d."tag"
        logging.error("Document sampling doesn't work yet!!")
        query = "SELECT * FROM " + self.namespace + ".\"" + self.column_name + "\""
        if self.sample_size != 0:
            query += " LIMIT " + str(self.sample_size)
        DATA = {'querylanguage': 'SQL',
                'query': query}
        logging.info("Sampling values from column: " + query)
        result = requests.post(url=self.URL, data=DATA)
        self.http_sample_result = result

    def take_sample(self, namespace_type):
        if namespace_type == 'RELATIONAL':
            return self.take_samples_relational()
        elif namespace_type == 'GRAPH':
            return self.take_sample_graph()
        elif namespace_type == 'DOCUMENT':
            return self.take_sample_document()

    def extract_sample(self, results: list):
        sample_lists = []
        for result in results:
            sample_string = result.content.decode('utf-8')
            sample_list = [string.strip()[1:] for string in sample_string.strip()[1:-2].split("],")]
            for sample in sample_list:
                sample_lists.append(sample)
        return sample_lists

