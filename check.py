import requests

res = requests.post(
    "http://localhost:8000/learn",
    json={"prompt": "Explain LangChain basics for educational purposes"}
)

print("Status:", res.status_code)
print("Response:", res.json())
