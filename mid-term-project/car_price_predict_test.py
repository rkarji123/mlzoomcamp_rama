import requests

url = "http://localhost:9696/car_price_predict"

car = {
    "Id":291998,
    "year":2014,
    "brand": "Volkswagen",
    "full_model_name": "Volkswagen Vento Highline 1.5(D)",
    "model_name":"Vento",
    "distance_travelled(kms)":100000,
    "fuel_type": "Petrol",
    "city":"Vijayawada",
    "brand_rank":3,
    "car_age": 12
}

response = requests.post(url, json=car).json()
print(response)
