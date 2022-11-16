# Given column name and table name, take a sample
import requests
import logging
import ast

class Sample():
    def __init__(self, host, port, column: str, table: str, sample_size: int, random=False):
        """Take a sample of values in a column.

        :param host: host of Polypheny Server
        :param port: port of Polypheny Server
        :param column: Column we want to sample
        :param table: Table we want to sample from
        :param sample_size: size of sample. 0 if unlimited!
        :param random: Should sample be randomized?
        """
        self.URL = host + ":" + port + "/query"
        self.column_name = column
        self.table_name = table
        self.sample_size = sample_size
        self.random = random

    def take_set_sample(self):
        query = "SELECT " + self.column_name + " FROM " + self.table_name
        if self.sample_size != 0:
            query += " LIMIT " + str(self.sample_size)
        DATA = {'querylanguage': 'SQL',
                'query': query}
        logging.info("Sampling values from column: " + query)
        result = requests.post(url=self.URL, data=DATA)
        self.http_sample_result = result
        return self

    def extract_sample(self):
        if self.http_sample_result:
            sample_list = ast.literal_eval(self.http_sample_result.content.decode('utf-8'))
            return sample_list
        else:
            try:
                self.http_sample_result = self.take_set_sample()
                self.extract_sample()
            except:
                raise RuntimeError("No sample taken before running extract_sample. Resampling failed.")