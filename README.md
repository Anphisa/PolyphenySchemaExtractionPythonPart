# PolyphenySchemaExtractionPythonPart
Python part of schema extraction for Polypheny.
This part communicates with Polypheny via HTTP and websockets requests. Polypheny sends requests to activate Python scripts, Python scripts query Polypheny via POST requests, Polypheny answers. Python sends results via POST request.

Capabilities:
- Schema matching - Match schemas with Valentine (https://github.com/delftdata/valentine) schema matchers
- Schema mapping - Map schema candidates with DynaMap (https://dl.acm.org/doi/abs/10.1145/3335783.3335785) schema mapper (open source re-implementation)
- Schema visualization - Visualize schema candidates locally with graphviz

Setup:
- Run Polypheny on localhost
- Start Python server
- Load schema data into Polypheny
- Navigate to "schema integration" pane in Polypheny, enter namespace(s), hit "Run"

Apache 2 license
