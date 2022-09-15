#!/usr/bin/env python
import sys
import polypheny
import json
import logging

# Connect to Polypheny
# TODO: Generalize connection details?
connection = polypheny.connect('localhost', 20591, user='pa', password='')
# TODO: Use server instead of polypheny module

# Get a Polypheny cursor
cursor = connection.cursor()

with open('../Sources/pdb-schema-extraction267500031893894154input') as f:
    # returns JSON object as a dictionary
    data = json.load(f)

    datamodel = data["datamodel"]
    print("datamodel:", datamodel)
    if datamodel != "RELATIONAL":
        raise ValueError('Datamodel is not relational. Not implemented!')
        sys.exit('Fatal schema extractor error')

    for i in data["tables"]:
        print("table name:", i["name"] + ".", "number of columns:", len(i["columnNames"]))

        # Execute a Polypheny query
        cursor.execute("SELECT * FROM " + i["name"])
        result = cursor.fetchone()
        print("Result Set: ", result)

# Close the Polypheny connection
connection.close()