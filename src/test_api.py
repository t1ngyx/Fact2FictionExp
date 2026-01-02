import asyncio
import aiohttp
import yaml
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from attack.attack_utils import query_gpt_single, GPT_URL, GPT_API_KEY

async def test():
    print(f"Testing API connection...")
    print(f"URL: {GPT_URL}")
    print(f"Key: {GPT_API_KEY[:5]}...{GPT_API_KEY[-5:]}")
    
    # Load the actual prompt
    try:
        with open("infact/prompts/pose_questions.md", "r", encoding="utf-8") as f:
            prompt_template = f.read()
        
        # Simulate a claim
        claim = "The earth is flat and rests on a giant turtle."
        prompt = prompt_template.replace("[CLAIM]", claim).replace("[N_QUESTIONS]", "5")
        
        print("\nTesting with actual prompt content...")
        async with aiohttp.ClientSession() as session:
            response = await query_gpt_single(
                session, 
                prompt, 
                model_name="gemini-2.5-flash",
                max_retries=3
            )
            
            if response:
                print(f"\nSuccess! Response length: {len(response)}")
                print(f"Response preview: {response[:200]}...")
            else:
                print("\nFailed to get response with actual prompt.")
                
    except Exception as e:
        print(f"Error during test: {e}")

if __name__ == "__main__":
    asyncio.run(test())
