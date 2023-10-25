
import requests as requests


ip = "127.0.0.1"
url = "http://"+ip+":8080/DispensarioJavaWebPostgreSQL/rest/rest_sendCommand"
data = {
    "SerialPort": "COM10",
    "BoardNumber": 1,
    "BoardPortNumber": 1,
    "Command": "c"
}
response = requests.post(url, json=data)
response.raise_for_status()  # raises exception when not a 2xx response

#print(response.status_code)
print(response.json())


