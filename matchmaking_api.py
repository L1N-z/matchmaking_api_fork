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
    Uses DeepSeek to generate a punchy, natural 1-sentence icebreaker.
    """
    system_instruction = """
    You are an elite networking wingman. 
    Your goal is to write ONE single, natural conversation starter sentence.
    - Do NOT use labels like 'Common Interest:' or 'Conversation Starter:'.
    - Do NOT write an explanation. Just write the sentence someone would say.
    - Mention the specific shared topic naturally in the sentence.
    - Keep it under 30 words.
    """

    user_prompt = f"""
    Person A ({user_a_name}) Profile: {user_a_profile}
    Person B ({user_b_name}) Profile: {user_b_profile}
    
    Write the conversation opener for Person A to say to Person B.
    """

    try:
        response = DEEPSEEK_CLIENT.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=150, 
            temperature=0.7
        )
        return response.choices[0].message.content.strip().replace('"', '')
    except Exception as e:
        print(f"❌ DeepSeek API Error: {e}")
        return f"Hi {user_b_name}, I noticed we work in similar fields and I'd love to connect!"


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
        top_matches_indices = [idx for idx in top_indices if all_guests[idx].person_code != current_guest.person_code][:5]
        
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


@app.get("/")
async def root():
    return {"status": "running", "service": "LumaConnect AI Matchmaker"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    print("🚀 Starting LumaConnect Matchmaking API...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")