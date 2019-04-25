import requests

# URL
url = 'http://localhost:5000/api'

r = requests.post(url,json={'text':'this is a really terrible review about your stupid company. I hate it',})

print(r.json())