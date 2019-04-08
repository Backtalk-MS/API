import requests

# URL
url = 'http://localhost:5000/api'

r = requests.post(url,json={'text':'Input test text here',})

print(r.json())