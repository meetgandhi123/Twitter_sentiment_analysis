import requests

url = 'http://localhost:5000/results'
r = requests.post(url,json={'Tweet':'@this is a awesome place.'})
print(r.json())