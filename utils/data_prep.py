import os
import random
import numpy as np
from PIL import Image, ImageDraw

def create_noise_texture(width, height, base_color, noise_level=15):
    """Creates a PIL Image of specified size with texture noise over base_color."""
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    r, g, b = base_color[0], base_color[1], base_color[2]
    
    for x in range(width):
        for y in range(height):
            # Apply slight noise to make it look like a textured pixel art skin
            noise = random.randint(-noise_level, noise_level)
            nr = max(0, min(255, r + noise))
            ng = max(0, min(255, g + noise))
            nb = max(0, min(255, b + noise))
            
            draw.point((x, y), fill=(nr, ng, nb, 255))
            
    return img

def generate_procedural_skin(output_path, theme, primary, secondary):
    """
    Generates a 64x64 Minecraft 1.8 skin procedurally.
    Coordinates match standard Minecraft skin texture layouts.
    """
    # Create main 64x64 RGBA canvas
    skin = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    
    # 1. HEAD (Base: 0..31, 0..15)
    # Head parts: top/bottom (8x8), sides (8x8). Let's make a textured head box
    head_tex = create_noise_texture(32, 16, primary, noise_level=20)
    skin.paste(head_tex, (0, 0))
    
    # Add face details (eyes, mouth) directly onto the front face (x: 8..15, y: 8..15)
    draw = ImageDraw.Draw(skin)
    # Eyes (glowing secondary color or white/black)
    eye_color = secondary
    # Right eye
    draw.point((9, 12), fill=eye_color)
    draw.point((10, 12), fill=(255, 255, 255, 255)) # sclera
    # Left eye
    draw.point((13, 12), fill=eye_color)
    draw.point((14, 12), fill=(255, 255, 255, 255)) # sclera
    # Mouth/mask
    draw.rectangle([10, 14, 13, 14], fill=(max(0, primary[0]-40), max(0, primary[1]-40), max(0, primary[2]-40), 255))
    
    # 2. HEAD OUTER LAYER / HAT (32..63, 0..15)
    # Draw some cool crown, headphones or horns with secondary color
    hat_draw = ImageDraw.Draw(skin)
    # Draw headphones or crown on outer layer (front is 40..47, 8..15)
    if "king" in theme or "royal" in theme:
        # Crown on front and sides of head
        hat_draw.rectangle([32, 4, 63, 7], fill=(255, 215, 0, 255)) # Golden crown band
        # Gems on crown (front 40..47)
        hat_draw.point((41, 5), fill=(255, 0, 0, 255)) # Ruby
        hat_draw.point((44, 5), fill=(0, 0, 255, 255)) # Sapphire
        hat_draw.point((46, 5), fill=(255, 0, 0, 255))
    elif "ninja" in theme or "assassin" in theme:
        # Mask overlay on face
        hat_draw.rectangle([40, 11, 47, 15], fill=secondary) # Ninja face mask
    elif "demon" in theme:
        # Horns on sides
        hat_draw.rectangle([40, 2, 41, 5], fill=secondary)
        hat_draw.rectangle([46, 2, 47, 5], fill=secondary)
        
    # 3. TORSO / BODY (16..39, 16..31)
    torso_tex = create_noise_texture(24, 16, primary, noise_level=15)
    skin.paste(torso_tex, (16, 16))
    # Chest detail / logo on front chest (20..27, 20..31)
    # Let's draw a symbol in secondary color (like an 'H' or 'X' or creeper face)
    torso_draw = ImageDraw.Draw(skin)
    if "creeper" in theme:
        # Creeper mouth
        torso_draw.rectangle([21, 23, 26, 27], fill=(0, 0, 0, 255))
        torso_draw.point((21, 28), fill=(0, 0, 0, 255))
        torso_draw.point((26, 28), fill=(0, 0, 0, 255))
    else:
        # Cool neon lines or cross
        torso_draw.line([22, 22, 25, 29], fill=secondary, width=1)
        torso_draw.line([25, 22, 22, 29], fill=secondary, width=1)

    # 4. JACKET / TORSO OUTER (16..39, 32..47)
    # A cool coat or belt overlay
    jacket_draw = ImageDraw.Draw(skin)
    jacket_draw.rectangle([16, 44, 39, 45], fill=secondary) # Belt around waist
    
    # 5. RIGHT LEG (0..15, 16..31)
    r_leg_tex = create_noise_texture(16, 16, primary, noise_level=15)
    skin.paste(r_leg_tex, (0, 16))
    # Boots
    leg_draw = ImageDraw.Draw(skin)
    leg_draw.rectangle([0, 28, 15, 31], fill=secondary)
    
    # 6. LEFT LEG (16..31, 48..63)
    l_leg_tex = create_noise_texture(16, 16, primary, noise_level=15)
    skin.paste(l_leg_tex, (16, 48))
    # Boots
    leg_draw.rectangle([16, 60, 31, 63], fill=secondary)
    
    # 7. RIGHT ARM (40..55, 16..31)
    r_arm_tex = create_noise_texture(16, 16, primary, noise_level=15)
    skin.paste(r_arm_tex, (40, 16))
    # Gloves/shoulder pads
    arm_draw = ImageDraw.Draw(skin)
    arm_draw.rectangle([40, 16, 55, 19], fill=secondary) # shoulder pad
    arm_draw.rectangle([40, 28, 55, 31], fill=secondary) # glove
    
    # 8. LEFT ARM (32..47, 48..63)
    l_arm_tex = create_noise_texture(16, 16, primary, noise_level=15)
    skin.paste(l_arm_tex, (32, 48))
    # Gloves/shoulder pads
    arm_draw.rectangle([32, 48, 47, 51], fill=secondary) # shoulder pad
    arm_draw.rectangle([32, 60, 47, 63], fill=secondary) # glove

    # Save to disk
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    skin.save(output_path, "PNG")

def generate_test_dataset(data_dir="dataset", count_per_theme=10):
    """
    Generates a full synthetic dataset of Minecraft skins with different themes, colors, and descriptions.
    """
    themes = [
        {"name": "fire demon", "primary": (40, 10, 10), "secondary": (255, 69, 0), "desc": "demon with lava red glowing eyes and horns"},
        {"name": "ice mage", "primary": (240, 248, 255), "secondary": (0, 191, 255), "desc": "frost wizard in frozen white robe and blue glowing eyes"},
        {"name": "forest elf", "primary": (34, 139, 34), "secondary": (139, 69, 19), "desc": "woodland ranger in dark green tunic and brown boots"},
        {"name": "ender knight", "primary": (20, 20, 30), "secondary": (186, 85, 211), "desc": "shadow warrior in dark void armor and glowing purple eyes"},
        {"name": "golden king", "primary": (255, 215, 0), "secondary": (128, 0, 128), "desc": "royal king with golden crown, purple cape and gems"},
        {"name": "cyber ninja", "primary": (17, 17, 17), "secondary": (0, 255, 127), "desc": "cyberpunk assassin with neon green mask and glowing armor lines"},
        {"name": "slime creeper", "primary": (50, 205, 50), "secondary": (0, 100, 0), "desc": "creeper boy in lime green hoodie and dark pants"},
        {"name": "royal guard", "primary": (70, 130, 180), "secondary": (218, 165, 32), "desc": "knight in blue steel armor with golden details"},
        {"name": "shadow assassin", "primary": (30, 30, 30), "secondary": (220, 20, 60), "desc": "dark ninja with blood red glowing eyes and scarf"},
        {"name": "space astronaut", "primary": (255, 255, 255), "secondary": (30, 144, 255), "desc": "astronaut with a white spacesuit, glass visor and blue patches"}
    ]
    
    print(f"Generating synthetic training dataset in '{data_dir}'...")
    total_generated = 0
    
    for theme in themes:
        # We generate several variations for each theme
        for i in range(count_per_theme):
            # Introduce color variation
            variation_factor = 0.8 + random.random() * 0.4 # [0.8, 1.2]
            p_color = tuple(max(0, min(255, int(c * variation_factor))) for c in theme["primary"])
            s_color = tuple(max(0, min(255, int(c * variation_factor))) for c in theme["secondary"])
            
            filename = f"{theme['name'].replace(' ', '_')}_v{i+1}.png"
            file_path = os.path.join(data_dir, filename)
            
            # Generate procedural skin
            generate_procedural_skin(file_path, theme["name"], p_color, s_color)
            
            # Generate a rich description for prompt pairing (both in Russian and English)
            # This teaches the model both languages simultaneously thanks to the multilingual encoder!
            descriptions = [
                f"{theme['desc']} variation {i+1}",
                f"cool {theme['name']} skin, style {i+1}",
                f"крутой скин {theme['name']} версия {i+1}"
            ]
            # Pick a random style description to make the dataset diverse
            prompt = random.choice(descriptions)
            
            # Save corresponding txt prompt file
            txt_path = os.path.join(data_dir, f"{theme['name'].replace(' ', '_')}_v{i+1}.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(prompt)
                
            total_generated += 1
            
    print(f"Dataset generated successfully! Total items: {total_generated} in '{data_dir}'.")
    return total_generated

if __name__ == "__main__":
    generate_test_dataset()
