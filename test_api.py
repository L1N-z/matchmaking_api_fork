"""
Test script to simulate n8n sending data to the matchmaking API
"""

import requests
import json

# Your sample data (trimmed profiles for brevity)
sample_data = [
  {
    "guests": [
      {
        "person_code": "L6D7X",
        "first_name": "Alex",
        "last_name": "Pinel Neparidze",
        "linkedin_url": "https://linkedin.com/in/London",
        "email": "",
        "phone": None,
        "refined_profile": "Fernando R. is a Contract Senior Software Engineer at Leonardo Helicopters...",
        "embedding": None,
        "last_updated": "2026-01-22T19:56:42.423346+00:00",
        "created_at": "2026-01-22T19:56:42.423346+00:00",
        "profile_pic_url": "https://media.licdn.com/dms/image/example.jpg",
        "luma_profile_url": "https://luma.com/user/usr-PfWkpBeRpFjYo6h"
      },
      {
        "person_code": "2Lplx",
        "first_name": "Aalok",
        "last_name": "Rai",
        "linkedin_url": "https://linkedin.com/in/aalok-rai",
        "email": "",
        "phone": None,
        "refined_profile": "Aalok Rai is the Founder and CEO of upLYFT...",
        "embedding": None,
        "last_updated": "2026-01-22T19:55:05.107589+00:00",
        "created_at": "2026-01-22T19:55:05.107589+00:00",
        "profile_pic_url": "https://media.licdn.com/dms/image/example2.jpg",
        "luma_profile_url": "https://luma.com/user/usr-I6KqNjeH6Bf90DK"
      },
      {
        "person_code": "COqzt",
        "first_name": "Adit",
        "last_name": "Shah",
        "linkedin_url": "https://linkedin.com/in/adit-shah-aero",
        "email": "adit.shah@elantar.com",
        "phone": None,
        "refined_profile": "Adit Shah is a Growth Readiness Advisor...",
        "embedding": None,
        "last_updated": "2026-01-22T19:55:28.853122+00:00",
        "created_at": "2026-01-22T19:55:28.853122+00:00",
        "profile_pic_url": "https://media.licdn.com/dms/image/example3.jpg",
        "luma_profile_url": "https://luma.com/user/usr-iVOs9zjTfkfWrtj"
      }
    ]
  }
]

# Test the API
api_url = "http://localhost:8000/matchmake"

print("🧪 Testing matchmaking API...")
print(f"📤 Sending data to: {api_url}")
print(f"📊 Guests in payload: {len(sample_data[0]['guests'])}\n")

try:
    response = requests.post(api_url, json=sample_data)
    
    if response.status_code == 200:
        print("✅ SUCCESS!")
        print(f"\n📥 Response:")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"❌ ERROR: Status {response.status_code}")
        print(response.text)
        
except requests.exceptions.ConnectionError:
    print("❌ ERROR: Could not connect to API")
    print("   Make sure the API is running: python matchmaking_api.py")
except Exception as e:
    print(f"❌ ERROR: {e}")