"""
LumaConnect Matchmaking API (AI Powered)
Receives guest data from n8n, performs Vector Matchmaking, and generates DeepSeek Icebreakers.
"""

import os
import asyncio
import numpy as np
import random
from typing import List, Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv
from openai import OpenAI

NUM_MATCHES = 20

# 1. Load Environment Variables
load_dotenv()

# 2. Initialize AI Clients
try:
    OPENAI_CLIENT = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    DEEPSEEK_CLIENT = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"), 
        base_url="https://api.deepseek.com"
    )
    print("✅ AI Clients Initialized")
except Exception as e:
    print(f"⚠️ Warning: AI Clients failed to initialize: {e}")

app = FastAPI(title="LumaConnect Matchmaking API")

# Enable CORS for n8n requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- DATA MODELS (Retained exactly as you requested) ---
class Guest(BaseModel):
    person_code: str
    first_name: str
    last_name: str
    linkedin_url: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    refined_profile: str
    embedding: Optional[List[float]] = None
    last_updated: Optional[str] = None
    created_at: Optional[str] = None
    profile_pic_url: Optional[str] = None
    luma_profile_url: Optional[str] = None


# --- HELPER FUNCTIONS ---

def get_embedding(text: str) -> List[float]:
    """Generates OpenAI embedding if missing."""
    try:
        text = text.replace("\n", " ")
        return OPENAI_CLIENT.embeddings.create(
            input=[text], 
            model="text-embedding-3-small"
        ).data[0].embedding
    except Exception as e:
        print(f"❌ Embedding Error: {e}")
        return [0.0] * 1536  # Return empty vector on failure to prevent crash

async def generate_icebreaker(user_a_name: str, user_a_profile: str, user_b_name: str, user_b_profile: str):
    """
    Uses DeepSeek to identify common ground and suggest a specific question.
    """
    system_instruction = """
    You are a direct, no-nonsense matchmaker. 
    Your goal is to identify the single strongest connection between two people and suggest a specific question to ask.
    
    Strict Output Format:
    "You both [shared point]. Ask [Person B Name] [specific relevant question]."

    Rules:
    1. Do NOT write a script for Person A to say (no "Hi I saw...").
    2. Do NOT imply Person A has read the profile.
    3. Identify the overlap (Same University, Same Hobby, Same Tech Stack, or Similar Role).
    4. If no overlap, pick the most unique thing about Person B to ask about.
    5. Keep it under 25 words. Simple and direct.
    """

    user_prompt = f"""
    Person A Name: {user_a_name}
    Person A Profile: {user_a_profile}
    
    Person B Name: {user_b_name}
    Person B Profile: {user_b_profile}
    """

    try:
        response = DEEPSEEK_CLIENT.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=100, 
            temperature=0.6 # Lower temperature for more factual logic
        )
        return response.choices[0].message.content.strip().replace('"', '')
    except Exception as e:
        print(f"❌ DeepSeek API Error: {e}")
        return f"Ask {user_b_name} about their work in the industry—you have similar backgrounds."

# --- MAIN ENDPOINT ---

@app.post("/matchmake")
async def matchmake_guests(guests: List[Guest]):
    """
    Receive guest data from n8n and perform matchmaking analysis
    """
    
    print("=" * 80)
    print("RECEIVED MATCHMAKING REQUEST")
    print("=" * 80)
    
    all_guests = guests
    
    print(f"\n📊 DATA DIMENSIONS:")
    print(f"   Total guests: {len(all_guests)}")
    
    # 1. DATA INSPECTION (Kept your original logging)
    print(f"\n👥 GUEST SAMPLE (first 3):")
    for i, guest in enumerate(all_guests[:3]):
        print(f"   Guest {i+1}: {guest.first_name} {guest.last_name} ({guest.person_code})")
        print(f"      Embedding: {'✓ Present' if guest.embedding else '✗ Missing (Will Generate)'}")

    # 2. ENSURE EMBEDDINGS
    print(f"\n⚙️  Checking/Generating Embeddings...")
    embeddings_generated = 0
    for guest in all_guests:
        if not guest.embedding:
            print(f"   Generating embedding for {guest.first_name}...")
            guest.embedding = get_embedding(guest.refined_profile)
            embeddings_generated += 1
    
    print(f"   ✅ All embeddings ready (Generated {embeddings_generated} new)")

    # 3. VECTOR MATCHMAKING MATRIX
    print(f"\n🧮 Calculating Similarity Matrix...")
    
    # Create Matrix [N x 1536]
    matrix = np.array([g.embedding for g in all_guests])
    
    # Normalize (for Cosine Similarity)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # Avoid division by zero
    norms[norms == 0] = 1 
    normalized_matrix = matrix / norms
    
    # Compute Dot Product [N x N]
    similarity_matrix = np.dot(normalized_matrix, normalized_matrix.T)
    
    results = {}
    
    print(f"🧠 Generating DeepSeek Icebreakers (Async)...")
    
    # 4. BUILD RESULTS & ICEBREAKERS
    for i, current_guest in enumerate(all_guests):
        
        # Get scores for this guest
        scores = similarity_matrix[i]
        
        # Sort indices by score (descending)
        top_indices = np.argsort(scores)[::-1]
        
        # Filter: Exclude self
        top_matches_indices = [idx for idx in top_indices if all_guests[idx].person_code != current_guest.person_code][:NUM_MATCHES]
        
        matches_data = []
        icebreaker_coroutines = []
        
        for rank, match_idx in enumerate(top_matches_indices):
            target = all_guests[match_idx]
            score_val = float(round(scores[match_idx], 4))
            
            # Prepare match object
            matches_data.append({
                "rank": rank + 1,
                "person_code": target.person_code,
                "name": f"{target.first_name} {target.last_name}",
                "profile_pic_url": target.profile_pic_url,
                "luma_profile_url": target.luma_profile_url,
                "score": score_val
            })
            
            # Queue the AI Task
            icebreaker_coroutines.append(
                generate_icebreaker(
                    current_guest.first_name, 
                    current_guest.refined_profile,
                    target.first_name,
                    target.refined_profile
                )
            )
        
        # Execute DeepSeek calls in parallel for this guest
        if icebreaker_coroutines:
            icebreakers = await asyncio.gather(*icebreaker_coroutines)
            
            # Attach icebreakers to matches
            for j, text in enumerate(icebreakers):
                matches_data[j]["conversation_starter"] = text
        
        # Add to final results using person_code as key
        results[current_guest.person_code] = {
            "person": {
                "person_code": current_guest.person_code,
                "first_name": current_guest.first_name,
                "last_name": current_guest.last_name,
                "profile_pic_url": current_guest.profile_pic_url,
                "luma_profile_url": current_guest.luma_profile_url
            },
            "matches": matches_data
        }

    print(f"✅ Generated AI rankings for {len(all_guests)} guests")
    
    return {
        "status": "success",
        "algorithm": "semantic-vector-cosine",
        "guests_processed": len(all_guests),
        "results": results
    }


@app.post("/matchmake_user")
async def matchmake_single_user(guests: List[Guest], target_person_code: str):
    """
    Perform matchmaking for a specific user identified by person_code.
    Returns matches only for that one user.

    Args:
        guests: List of all guests (including the target user)
        target_person_code: The person_code of the user to generate matches for
    """

    print("=" * 80)
    print(f"RECEIVED SINGLE USER MATCHMAKING REQUEST FOR: {target_person_code}")
    print("=" * 80)

    all_guests = guests

    # Find the target user
    target_user = None
    for guest in all_guests:
        if guest.person_code == target_person_code:
            target_user = guest
            break

    if not target_user:
        return {
            "status": "error",
            "message": f"User with person_code '{target_person_code}' not found"
        }

    print(f"\n📊 DATA DIMENSIONS:")
    print(f"   Total guests: {len(all_guests)}")
    print(f"   Target user: {target_user.first_name} {target_user.last_name}")

    # Ensure embeddings for all guests
    print(f"\n⚙️  Checking/Generating Embeddings...")
    embeddings_generated = 0
    for guest in all_guests:
        if not guest.embedding:
            print(f"   Generating embedding for {guest.first_name}...")
            guest.embedding = get_embedding(guest.refined_profile)
            embeddings_generated += 1

    print(f"   ✅ All embeddings ready (Generated {embeddings_generated} new)")

    # Create embedding matrix
    print(f"\n🧮 Calculating Similarity Scores...")
    matrix = np.array([g.embedding for g in all_guests])

    # Normalize for cosine similarity
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalized_matrix = matrix / norms

    # Compute similarity scores
    similarity_matrix = np.dot(normalized_matrix, normalized_matrix.T)

    # Find index of target user
    target_index = all_guests.index(target_user)

    # Get scores for target user
    scores = similarity_matrix[target_index]

    # Sort indices by score (descending)
    top_indices = np.argsort(scores)[::-1]

    # Filter: Exclude self
    top_matches_indices = [idx for idx in top_indices if all_guests[idx].person_code != target_person_code][:NUM_MATCHES]

    print(f"🧠 Generating DeepSeek Icebreakers for {len(top_matches_indices)} matches...")

    matches_data = []
    icebreaker_coroutines = []

    for rank, match_idx in enumerate(top_matches_indices):
        target_match = all_guests[match_idx]
        score_val = float(round(scores[match_idx], 4))

        # Prepare match object
        matches_data.append({
            "rank": rank + 1,
            "person_code": target_match.person_code,
            "name": f"{target_match.first_name} {target_match.last_name}",
            "profile_pic_url": target_match.profile_pic_url,
            "luma_profile_url": target_match.luma_profile_url,
            "score": score_val
        })

        # Queue the AI Task
        icebreaker_coroutines.append(
            generate_icebreaker(
                target_user.first_name,
                target_user.refined_profile,
                target_match.first_name,
                target_match.refined_profile
            )
        )

    # Execute DeepSeek calls in parallel
    if icebreaker_coroutines:
        icebreakers = await asyncio.gather(*icebreaker_coroutines)

        # Attach icebreakers to matches
        for j, text in enumerate(icebreakers):
            matches_data[j]["conversation_starter"] = text

    print(f"✅ Generated {len(matches_data)} matches for {target_user.first_name}")

    return {
        "status": "success",
        "algorithm": "semantic-vector-cosine",
        "target_user": {
            "person_code": target_user.person_code,
            "first_name": target_user.first_name,
            "last_name": target_user.last_name,
            "profile_pic_url": target_user.profile_pic_url,
            "luma_profile_url": target_user.luma_profile_url
        },
        "matches": matches_data
    }


@app.get("/")
async def root():
    return {"status": "running", "service": "LumaConnect AI Matchmaker"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    print("🚀 Starting LumaConnect Matchmaking API...")
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")