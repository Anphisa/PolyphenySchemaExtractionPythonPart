import requests

# api-endpoint
URL = "http://127.0.0.1:20598/query"

# defining a params dict for the parameters to be sent to the API
PARAMS = {'querylanguage':'SQL', 'query':'blub'}

# sending get request and saving the response as response object
r = requests.get(url = URL, params = PARAMS)
print(r)
print(r.content)