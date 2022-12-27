#!/usr/bin/env python
import sys
import json
import logging
import requests
import asyncio
import websockets
import datetime
import ast
import collections
import time
import pandas as pd
from valentine.algorithms import Coma, JaccardLevenMatcherColNamesOnly
from valentine import valentine_match
from Mapping.Mapping import Mapping
from GenerateSchemaCandidates.SchemaCandidateVisualization import SchemaCandidateVisualization

from PkFkFinder import PkFkFinder

# TODO: Message queue not as global variable
ws_message_queue = collections.deque(maxlen=100)

async def pk_fk_relationship_finder(data):
    datamodel = data["datamodel"]
    print("datamodel:", datamodel)
    if datamodel != "RELATIONAL":
        logging.error('Datamodel is not relational. Not implemented!')
        raise ValueError('Datamodel is not relational. Not implemented!')
    else:
        pk_fk_finder = PkFkFinder("http://127.0.0.1", "20598", data["tables"], 20)
        sampled_pk_fk_relationships = pk_fk_finder.above_similarity_threshold_pk_comparison()
        validated_pk_fk_relationships = pk_fk_finder.validate_pk_fk_relationships(sampled_pk_fk_relationships)
        print("Validated pk_fk relationships", validated_pk_fk_relationships)

def build_dataframes(message):
    message = ast.literal_eval(message["message"])
    tables_list = message["tables"]
    tables_df = {}
    for table in tables_list:
        table_name = table["tableName"]
        column_names = table["columnNames"]
        primary_key = table["primaryKey"]
        foreign_keys = table["foreignKeys"]
        data_model = message["datamodel"]
        df = pd.DataFrame(columns=[column_names])
        if table_name not in tables_df:
            tables_df[table_name] = {}
        tables_df[table_name]["columns"] = df
        tables_df[table_name]["pk"] = primary_key
        tables_df[table_name]["fk"] = foreign_keys
        tables_df[table_name]["datamodel"] = data_model
    return tables_df

def dataframe_valentine_compare(dataframes, valentine_method):
    # given a dict of dataframes (generated by build_dataframes), make a pairwise comparison using valentine
    all_matches = {}
    checked_combinations = []
    for df1 in dataframes:
        for df2 in dataframes:
            if df1 != df2 and (df1, df2) not in checked_combinations and (df2, df1) not in checked_combinations:
                matches = valentine_match(dataframes[df1]["columns"], dataframes[df2]["columns"], valentine_method, df1, df2)
                # add matches to all matches
                all_matches = all_matches | matches
                checked_combinations.append((df1, df2))
    return all_matches

async def consumer(message):
    print("consumer: ", message)
    d_message = ast.literal_eval(message)
    if d_message["topic"] == "namespaceInfo":
        loop = asyncio.get_event_loop()
        dfs = await loop.run_in_executor(None, build_dataframes, d_message)
        # matching
        # todo: threshold for jl colnames only
        matches = await loop.run_in_executor(None, dataframe_valentine_compare, dfs, JaccardLevenMatcherColNamesOnly())
        #print("matches:", await matches)
        # mapping
        matches_above_thresh = await loop.run_in_executor(None, Mapping.naiveMapping, matches, 0.5)
        #print("matches above thresh:", await matches_above_thresh)
        mapping_result = await loop.run_in_executor(None, Mapping.naiveMappingDecider, dfs, matches_above_thresh)
        # [{'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'deptno', 'field_name2': 'deptno', 'relationship_type': 'view union', 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'name', 'field_name2': 'name', 'relationship_type': 'view union', 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'emp', 'table2': 'work', 'field_name1': 'employeeno', 'field_name2': 'employeeno', 'relationship_type': 'view union', 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'deptno', 'field_name2': 'deptno', 'relationship_type': ('join',), 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'name', 'field_name2': 'name', 'relationship_type': ('join',), 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'emp', 'table2': 'work', 'field_name1': 'employeeno', 'field_name2': 'employeeno', 'relationship_type': ('join',), 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'deptno', 'field_name2': 'deptno', 'relationship_type': 'view union', 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'name', 'field_name2': 'name', 'relationship_type': 'view union', 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'emp', 'table2': 'work', 'field_name1': 'employeeno', 'field_name2': 'employeeno', 'relationship_type': 'view union', 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'deptno', 'field_name2': 'deptno', 'relationship_type': ('join',), 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'name', 'field_name2': 'name', 'relationship_type': ('join',), 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'emp', 'table2': 'work', 'field_name1': 'employeeno', 'field_name2': 'employeeno', 'relationship_type': ('join',), 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'deptno', 'field_name2': 'deptno', 'relationship_type': 'view union', 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'name', 'field_name2': 'name', 'relationship_type': 'view union', 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'emp', 'table2': 'work', 'field_name1': 'employeeno', 'field_name2': 'employeeno', 'relationship_type': 'view union', 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'deptno', 'field_name2': 'deptno', 'relationship_type': ('join',), 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'name', 'field_name2': 'name', 'relationship_type': ('join',), 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'emp', 'table2': 'work', 'field_name1': 'employeeno', 'field_name2': 'employeeno', 'relationship_type': ('join',), 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'deptno', 'field_name2': 'deptno', 'relationship_type': 'view union', 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'name', 'field_name2': 'name', 'relationship_type': 'view union', 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'emp', 'table2': 'work', 'field_name1': 'employeeno', 'field_name2': 'employeeno', 'relationship_type': 'view union', 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'deptno', 'field_name2': 'deptno', 'relationship_type': ('join',), 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'name', 'field_name2': 'name', 'relationship_type': ('join',), 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'emp', 'table2': 'work', 'field_name1': 'employeeno', 'field_name2': 'employeeno', 'relationship_type': ('join',), 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'deptno', 'field_name2': 'deptno', 'relationship_type': 'view union', 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'name', 'field_name2': 'name', 'relationship_type': 'view union', 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'emp', 'table2': 'work', 'field_name1': 'employeeno', 'field_name2': 'employeeno', 'relationship_type': 'view union', 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'deptno', 'field_name2': 'deptno', 'relationship_type': ('join',), 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'name', 'field_name2': 'name', 'relationship_type': ('join',), 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'emp', 'table2': 'work', 'field_name1': 'employeeno', 'field_name2': 'employeeno', 'relationship_type': ('join',), 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'deptno', 'field_name2': 'deptno', 'relationship_type': 'view union', 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'name', 'field_name2': 'name', 'relationship_type': 'view union', 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'emp', 'table2': 'work', 'field_name1': 'employeeno', 'field_name2': 'employeeno', 'relationship_type': 'view union', 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'deptno', 'field_name2': 'deptno', 'relationship_type': ('join',), 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'name', 'field_name2': 'name', 'relationship_type': ('join',), 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'emp', 'table2': 'work', 'field_name1': 'employeeno', 'field_name2': 'employeeno', 'relationship_type': ('join',), 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'deptno', 'field_name2': 'deptno', 'relationship_type': 'view union', 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'name', 'field_name2': 'name', 'relationship_type': 'view union', 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'emp', 'table2': 'work', 'field_name1': 'employeeno', 'field_name2': 'employeeno', 'relationship_type': 'view union', 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'deptno', 'field_name2': 'deptno', 'relationship_type': ('join',), 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'name', 'field_name2': 'name', 'relationship_type': ('join',), 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'emp', 'table2': 'work', 'field_name1': 'employeeno', 'field_name2': 'employeeno', 'relationship_type': ('join',), 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'deptno', 'field_name2': 'deptno', 'relationship_type': 'view union', 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'name', 'field_name2': 'name', 'relationship_type': 'view union', 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'emp', 'table2': 'work', 'field_name1': 'employeeno', 'field_name2': 'employeeno', 'relationship_type': 'view union', 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'deptno', 'field_name2': 'deptno', 'relationship_type': ('join',), 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'name', 'field_name2': 'name', 'relationship_type': ('join',), 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'emp', 'table2': 'work', 'field_name1': 'employeeno', 'field_name2': 'employeeno', 'relationship_type': ('join',), 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'deptno', 'field_name2': 'deptno', 'relationship_type': 'view union', 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'name', 'field_name2': 'name', 'relationship_type': 'view union', 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'emp', 'table2': 'work', 'field_name1': 'employeeno', 'field_name2': 'employeeno', 'relationship_type': 'view union', 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'deptno', 'field_name2': 'deptno', 'relationship_type': ('join',), 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'name', 'field_name2': 'name', 'relationship_type': ('join',), 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'emp', 'table2': 'work', 'field_name1': 'employeeno', 'field_name2': 'employeeno', 'relationship_type': ('join',), 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'deptno', 'field_name2': 'deptno', 'relationship_type': 'view union', 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'name', 'field_name2': 'name', 'relationship_type': 'view union', 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'emp', 'table2': 'work', 'field_name1': 'employeeno', 'field_name2': 'employeeno', 'relationship_type': 'view union', 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'deptno', 'field_name2': 'deptno', 'relationship_type': ('join',), 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'name', 'field_name2': 'name', 'relationship_type': ('join',), 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'emp', 'table2': 'work', 'field_name1': 'employeeno', 'field_name2': 'employeeno', 'relationship_type': ('join',), 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'deptno', 'field_name2': 'deptno', 'relationship_type': 'view union', 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'name', 'field_name2': 'name', 'relationship_type': 'view union', 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'emp', 'table2': 'work', 'field_name1': 'employeeno', 'field_name2': 'employeeno', 'relationship_type': 'view union', 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'deptno', 'field_name2': 'deptno', 'relationship_type': ('join',), 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'name', 'field_name2': 'name', 'relationship_type': ('join',), 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'emp', 'table2': 'work', 'field_name1': 'employeeno', 'field_name2': 'employeeno', 'relationship_type': ('join',), 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'deptno', 'field_name2': 'deptno', 'relationship_type': 'view union', 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'name', 'field_name2': 'name', 'relationship_type': 'view union', 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'emp', 'table2': 'work', 'field_name1': 'employeeno', 'field_name2': 'employeeno', 'relationship_type': 'view union', 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'deptno', 'field_name2': 'deptno', 'relationship_type': ('join',), 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'depts', 'table2': 'emps', 'field_name1': 'name', 'field_name2': 'name', 'relationship_type': ('join',), 'relationship_strength': 1}, {'object': 'FieldRelationship', 'table1': 'emp', 'table2': 'work', 'field_name1': 'employeeno', 'field_name2': 'employeeno', 'relationship_type': ('join',), 'relationship_strength': 1}]
        schema_candidate_graph = SchemaCandidateVisualization(0,
                                                              [{"operation" : "left join: dept:deptid <-> emps:deptid",
                                                                "explanation": "JL col names only matcher: 95% match"}],
                                                              {t: {"datamodel": value["datamodel"],
                                                                   "pk": value["pk"],
                                                                   "columns": value["columns"]} for (t, value) in dfs.items()},
                                                              [{"from_table": m.table1,
                                                                "to_table": m.table2,
                                                                "from_column": m.field_name1,
                                                                "to_column": m.field_name2,
                                                                "relationship": m.relationship_type,
                                                                "relationship_strength": m.relationship_strength} for m in mapping_result],
                                                              "Field loss: depts: loss of fields [1, 2, 3]",
                                                              "Instance loss: 95% of bla",
                                                              "Structure loss: None.")
        g = schema_candidate_graph.draw()
        print("graphviz dot code:", g.source)
        g.view()
        print("mapping_results:", mapping_result)
        URL = "http://127.0.0.1:20598/result"
        PARAMS = {'results': str(mapping_result)}
        print("sending PARAMS", PARAMS)
        r = requests.post(url=URL, data=PARAMS)
        print("response", r.status_code)
        #ws_message_queue.appendleft(str(await matches))
        #await pk_fk_relationship_finder(ast.literal_eval(d_message["message"]))
    if d_message["topic"] == "speedThoroughness":
        speedThoroughness = int(d_message["message"])
        print("set speed/thoroughness to ", speedThoroughness)
        # TODO: Use this parameter to set Valentine methods or sth like that. Make & use global config

async def consumer_handler(websocket):
    while True:
        async for message in websocket:
            await consumer(message)    #global glob_message

async def producer_handler(websocket):
    while True:
        if len(ws_message_queue) > 0:
            message = ws_message_queue.pop()
            print("producer: ", message)
            await websocket.send(message)
        await asyncio.sleep(2.0)

async def status_producer():
    now = datetime.datetime.utcnow().isoformat() + 'Z'
    return (now)

async def status_producer_handler(websocket):
    while True:
        message = await status_producer()
        print("status_producer: ", message)
        await websocket.send(message)
        await asyncio.sleep(5.0)

async def handler(websocket):
    consumer_task = asyncio.create_task(consumer_handler(websocket))
    producer_task = asyncio.create_task(producer_handler(websocket))
    status_producer_task = asyncio.create_task(status_producer_handler(websocket))
    done, pending = await asyncio.wait(
        [consumer_task, producer_task, status_producer_task],
        return_when=asyncio.FIRST_COMPLETED,
    )
    for task in pending:
        task.cancel()

async def main():
    # Connect to Polypheny: websocket connection
    # Here we get user input (i.e. call to run Python script)
    try:
        async with websockets.connect("ws://localhost:20598/register") as websocket:
            await handler(websocket)
            await asyncio.Future()  # run forever
    except ConnectionRefusedError as e:
        logging.error("Polypheny websocket server not running or responding")
        print("Polypheny websocket server not running or responding", e)

    # TODO: Define a pipeline of steps to be performed (later to be optimized/determined dynamically due to user inputs, etc.)

if __name__ == "__main__":
    logging.getLogger("asyncio").setLevel(logging.INFO)
    logging.Formatter('%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    # TODO: Set logging file name (https://stackoverflow.com/questions/13839554/how-to-change-filehandle-with-python-logging-on-the-fly-with-different-classes-a)
    logging.info('Started schema integration Python part (main.py)')

    asyncio.run(main())

    logging.info('Closing Polypheny connection. Exiting schema integration Python part.')