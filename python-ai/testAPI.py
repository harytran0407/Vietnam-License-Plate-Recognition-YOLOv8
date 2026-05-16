import requests

url = "http://localhost:5000/api/parking/checkin"

data = {
    "id":"1234",
    "licensePlate": "51A12345",
    "cameraId": "CAM-01",
    "checkinTime": "2026-09-01T10:00:00"
}

r = requests.post(url, json=data)

print(r.status_code)
print(r.text)