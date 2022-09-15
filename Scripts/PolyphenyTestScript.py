#!/usr/bin/env python
import polypheny

# Connect to Polypheny
connection = polypheny.connect('localhost', 20591, user='pa', password='')

# Get a cursor
cursor = connection.cursor()

# Execute a query
cursor.execute("SELECT * FROM emps")
result = cursor.fetchall()
print("Result Set: ", result)

# Close the connection
connection.close()