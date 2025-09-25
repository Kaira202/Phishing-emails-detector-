import requests

# Change to your Flask server URL
url = "http://127.0.0.1:5000/predict"

# Example 1: single 'text' field
payload_text = {
    "text": "Your account has been suspended. Click here to verify immediately."
}

# Example 2: subject + body
payload_subject_body = {
    "subject": "Your account has been suspended",
    "body": "Click here to verify your account immediately."
}

payload = payload_subject_body  # choose one

response = requests.post(url, json=payload)

print("Status code:", response.status_code)
print("Raw text:", response.text)

try:
    print("JSON:", response.json())
except Exception as e:
    print("Error parsing JSON:", e)
    