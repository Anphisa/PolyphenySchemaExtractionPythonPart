# Given column name and table name, take a sample
import requests
import logging
import ast

class Sample():
    def __init__(self, host, port, column: str, namespace: str, table: str, sample_size: int, random=False):
        """Take a sample of values in a column.

        :param host: host of Polypheny Server
        :param port: port of Polypheny Server
        :param column: Column we want to sample
        :param namespace: Namespace that we can find the table in
        :param table: Table we want to sample from
        :param sample_size: size of sample. 0 if unlimited!
        :param random: Should sample be randomized?
        """
        self.URL = host + ":" + port + "/query"
        self.column_name = column
        self.namespace = namespace
        self.table_name = table
        self.sample_size = sample_size
        # TODO: Implement random sampling
        self.random = random
        self.sample = None
        self.flat_sample = None

    def take_sample_relational(self):
        query = "SELECT " + self.column_name + " FROM " + self.namespace + "." + self.table_name
        if self.sample_size != 0:
            query += " LIMIT " + str(self.sample_size)
        DATA = {'querylanguage': 'SQL',
                'query': query}
        logging.info("Sampling values from column: " + query)
        result = requests.post(url=self.URL, data=DATA)
        self.http_sample_result = result

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

    def take_sample(self):
        if self.namespace_type == 'RELATIONAL':
            return self.take_sample_relational()
        elif self.namespace_type == 'GRAPH':
            return self.take_sample_graph()
        elif self.namespace_type == 'DOCUMENT':
            return self.take_sample_document()

    def extract_sample(self):
        if self.http_sample_result:
            sample_string = self.http_sample_result.content.decode('utf-8')
            # TODO: Results in all string values. Is that a problem? Perhaps
            sample_list = [s.replace('[','').replace(']', '').strip() for s in sample_string.split(",")]
            self.sample = sample_list
        else:
            try:
                self.http_sample_result = self.take_set_sample()
                self.extract_sample()
            except:
                raise RuntimeError("No sample taken before running extract_sample. Resampling failed.")
