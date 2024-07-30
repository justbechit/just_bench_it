import time
import requests

GITHUB_REPO_OWNER = "justbechit"
GITHUB_REPO_NAME = "rl_ladder"
GITHUB_CLIENT_ID = "Ov23li6k9dJ9Ws9bsWy0"
GITHUB_CLIENT_SECRET = "5c8ab1c55c159ecb683d7fbbfe1efe657dc1d536"

def get_device_code():
    url = "https://github.com/login/device/code"
    headers = {"Accept": "application/json"}
    data = {
        "client_id": GITHUB_CLIENT_ID,
        "scope": "repo"
    }
    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to obtain device code: {response.content}")
        return None

def poll_for_token(device_code, interval):
    url = "https://github.com/login/oauth/access_token"
    headers = {"Accept": "application/json"}
    data = {
        "client_id": GITHUB_CLIENT_ID,
        "device_code": device_code,
        "grant_type": "urn:ietf:params:oauth:grant-type:device_code"
    }
    while True:
        response = requests.post(url, headers=headers, data=data)
        if response.status_code == 200:
            response_data = response.json()
            if "access_token" in response_data:
                return response_data.get("access_token")
            elif "error" in response_data and response_data["error"] == "authorization_pending":
                print("Authorization pending. Waiting for user to authorize...")
                time.sleep(interval)
            else:
                print(f"Error in response: {response_data}")
                break
        else:
            print(f"Failed to poll for token: {response.content}")
            break
    return None

def get_github_token():
    device_data = get_device_code()
    if not device_data:
        return None

    print(f"Please go to {device_data['verification_uri']} and enter the code: {device_data['user_code']}")

    return poll_for_token(device_data["device_code"], device_data["interval"])

# Usage example
token = get_github_token()
if token:
    print(f"Successfully obtained GitHub token: {token}")
else:
    print("Failed to obtain GitHub token.")

