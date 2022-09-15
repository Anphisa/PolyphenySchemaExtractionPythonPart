#!/usr/bin/env python
import sys
import json
import logging

def main():
    logging.basicConfig(filename='PolyphenySchemaIntegrationPython.log',
                        level=logging.INFO,
                        format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    logging.info('Started schema integration Python part (main.py)')

    # Connect to Polypheny
    # TODO: Use server instead of polypheny module
    connection = polypheny.connect('localhost', 20591, user='pa', password='')
    host = "http://127.0.0.1"
    port = "20598"
    URL = host + ":" + port
    #URL = "http://127.0.0.1:20598/query"

    # Get a Polypheny cursor
    cursor = connection.cursor()

    # Define a pipeline of steps to be performed (later to be optimized/determined dynamically due to user inputs, etc.)
    pipeline = ['pkfkfinder']

    # Get data about namespace (tables, columns, etc)
    # todo...

    with open('../Sources/pdb-schema-extraction267500031893894154input') as f:
        # returns JSON object as a dictionary
        data = json.load(f)

        datamodel = data["datamodel"]
        print("datamodel:", datamodel)
        if datamodel != "RELATIONAL":
            logging.error('Datamodel is not relational. Not implemented!')
            raise ValueError('Datamodel is not relational. Not implemented!')
            sys.exit('Fatal schema extractor error')

        for i in data["tables"]:
            print("table name:", i["name"] + ".", "number of columns:", len(i["columnNames"]))

            # Execute a Polypheny query
            cursor.execute("SELECT * FROM " + i["name"])
            result = cursor.fetchone()
            print("Result Set: ", result)

    # Close the Polypheny connection
    logging.info('Closing Polypheny connection. Exiting schema integration Python part.')
    connection.close()