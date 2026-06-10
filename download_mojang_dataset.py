import os
import re
import json
import urllib.request
import urllib.error
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

def clean_username_to_prompt(username):
    """
    Cleans a Minecraft username into a natural-sounding English prompt.
    E.g. "CreeperBoy99" -> "creeper boy 99"
    "cool_blue_ninja" -> "cool blue ninja"
    """
    # Split camelCase
    cleaned = re.sub(r'(?<!^)(?=[A-Z])', ' ', username)
    # Replace underscores, hyphens, dots with spaces
    cleaned = re.sub(r'[_.\-]', ' ', cleaned)
    # Split letters and digits (e.g. "Gamer99" -> "Gamer 99")
    cleaned = re.sub(r'([a-zA-Z])([0-9])', r'\1 \2', cleaned)
    cleaned = re.sub(r'([0-9])([a-zA-Z])', r'\1 \2', cleaned)
    # Remove multiple spaces and lowercase
    cleaned = re.sub(r'\s+', ' ', cleaned).strip().lower()
    return cleaned

def fetch_potential_usernames():
    print("Fetching wordlist of common usernames from public repository...")
    url = "https://raw.githubusercontent.com/jeanphorn/wordlist/master/usernames.txt"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            names = response.read().decode('utf-8').splitlines()
        print(f"Successfully fetched {len(names)} raw names from wordlist.")
        
        # Filter for valid Minecraft username constraints:
        # 3 to 16 characters, only letters, numbers, and underscores
        valid_pattern = re.compile(r'^[a-zA-Z0-9_]{3,16}$')
        valid_names = [name.strip() for name in names if valid_pattern.match(name.strip())]
        print(f"Filtered to {len(valid_names)} potential valid Minecraft usernames.")
        return valid_names
    except Exception as e:
        print(f"Error fetching usernames list: {e}")
        return []

def resolve_potential_usernames_to_profiles(potential_names, target_count=1000):
    """
    Resolves potential names to real Minecraft player profiles (UUID and name)
    in batches of 100 using official Mojang bulk API.
    """
    print(f"Resolving potential usernames to real profiles (target: {target_count})...")
    resolved_profiles = []
    
    # Shuffle the potential names to get a diverse, random set of players
    shuffled_names = list(set(potential_names))
    random.seed(42)  # For reproducible dataset selection
    random.shuffle(shuffled_names)
    
    batch_size = 10
    for i in range(0, len(shuffled_names), batch_size):
        if len(resolved_profiles) >= target_count:
            break
            
        batch = shuffled_names[i:i+batch_size]
        
        # POST request to Mojang Bulk API
        url = "https://api.mojang.com/profiles/minecraft"
        data = json.dumps(batch).encode('utf-8')
        
        req = urllib.request.Request(
            url, 
            data=data, 
            headers={
                'Content-Type': 'application/json',
                'User-Agent': 'Mozilla/5.0'
            },
            method='POST'
        )
        
        try:
            with urllib.request.urlopen(req) as response:
                profiles = json.loads(response.read().decode('utf-8'))
                
            for profile in profiles:
                resolved_profiles.append({
                    "uuid": profile["id"],
                    "username": profile["name"]
                })
                
            if len(profiles) > 0:
                print(f"Progress: Found {len(resolved_profiles)} genuine active player accounts...")
            
            # Brief sleep to avoid hitting Mojang rate limit
            time.sleep(0.2)
            
        except urllib.error.HTTPError as e:
            print(f"HTTP Error during bulk resolve: {e.code} - {e.reason}")
            if e.code == 429:
                print("Rate limit reached. Sleeping for 10 seconds...")
                time.sleep(10.0)
            else:
                time.sleep(2.0)
        except Exception as e:
            print(f"Error during bulk resolve: {e}")
            time.sleep(1.0)
            
    return resolved_profiles[:target_count]

def download_skin(profile, output_dir):
    uuid = profile["uuid"]
    username = profile["username"]
    
    # Avoid invalid characters in filenames on Windows
    safe_username = re.sub(r'[\\/*?:"<>|]', '', username)
    if not safe_username:
        return False
        
    png_path = os.path.join(output_dir, f"{safe_username}.png")
    txt_path = os.path.join(output_dir, f"{safe_username}.txt")
    
    # Check if we already downloaded this skin to avoid redownloads
    if os.path.exists(png_path) and os.path.exists(txt_path):
        return True
        
    # Download 2D skin file from Crafatar (very fast, rate-limit free)
    skin_url = f"https://crafatar.com/skins/{uuid}"
    
    try:
        req = urllib.request.Request(skin_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            img_data = response.read()
            
        # Verify it's a valid PNG
        if not img_data.startswith(b'\x89PNG\r\n\x1a\n'):
            return False
            
        # Save PNG
        with open(png_path, "wb") as f:
            f.write(img_data)
            
        # Save TXT prompt
        prompt = clean_username_to_prompt(username)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(prompt)
            
        return True
    except Exception:
        # Fallback to mc-heads
        try:
            skin_url = f"https://mc-heads.net/skin/{uuid}"
            req = urllib.request.Request(skin_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=10) as response:
                img_data = response.read()
            if img_data.startswith(b'\x89PNG\r\n\x1a\n'):
                with open(png_path, "wb") as f:
                    f.write(img_data)
                prompt = clean_username_to_prompt(username)
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(prompt)
                return True
        except Exception:
            pass
    return False

def main():
    target_count = 1000
    output_dir = r"c:\Users\USER\Desktop\ai\dataset"
    os.makedirs(output_dir, exist_ok=True)
    
    potential_names = fetch_potential_usernames()
    if not potential_names:
        print("Failed to load potential usernames. Aborting.")
        return
        
    resolved_profiles = resolve_potential_usernames_to_profiles(potential_names, target_count=target_count)
    if not resolved_profiles:
        print("Failed to resolve any usernames to profiles. Aborting.")
        return
        
    print(f"\nStarting bulk download of {len(resolved_profiles)} skins using ThreadPoolExecutor...")
    success_count = 0
    
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(download_skin, profile, output_dir): profile for profile in resolved_profiles}
        for idx, future in enumerate(as_completed(futures), 1):
            profile = futures[future]
            try:
                success = future.result()
                if success:
                    success_count += 1
                if idx % 100 == 0 or idx == len(resolved_profiles):
                    print(f"Downloaded {idx}/{len(resolved_profiles)} skins... (Success rate: {success_count}/{idx})")
            except Exception as e:
                print(f"Thread error downloading skin for {profile['username']}: {e}")
                
    print(f"\n🎉 Finished! Successfully downloaded and labeled {success_count} high-quality skins in '{output_dir}'!")

if __name__ == "__main__":
    main()
