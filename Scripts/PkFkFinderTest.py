#!/usr/bin/env python
import sys
import polypheny
import json
import logging
from PkFkFinder import PkFkFinder

# Connect to Polypheny
connection = polypheny.connect('localhost', 20591, user='pa', password='')

# Get a Polypheny cursor
cursor = connection.cursor()

with open('../Sources/pdb-schema-extraction6698677105460595130input') as f:
    # returns JSON object as a dictionary
    data = json.load(f)

    datamodel = data["datamodel"]
    print("datamodel:", datamodel)
    if datamodel != "RELATIONAL":
        raise ValueError('Datamodel is not relational. Not implemented!')
        sys.exit('Fatal schema extractor error')
    else:
        pk_fk_finder = PkFkFinder(cursor, data["tables"], 20)
        pk_fk_relationships = pk_fk_finder.above_similarity_threshold_pk_comparison()
        print(pk_fk_relationships)

# Close the Polypheny connection
connection.close()