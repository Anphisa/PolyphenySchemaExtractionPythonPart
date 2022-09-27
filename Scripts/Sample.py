# Given column name and table name, take a sample
import requests
import logging

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

    def take_sample(self):
        query = "SELECT " + self.column_name + " FROM " + self.table_name
        if self.sample_size != 0:
            query += " LIMIT " + str(self.sample_size)
        DATA = {'querylanguage': 'SQL',
                'query': query}
        result = requests.post(url=self.URL, data=DATA)
        logging.info("Sampling values from column: " + query)
        return result