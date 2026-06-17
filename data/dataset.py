import os
import json
import random
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import re
import numpy as np

from utils.visualization import get_minecraft_mask

class MinecraftSkinDataset(Dataset):
    def __init__(self, data_dir, text_encoder=None, transform=None):
        """
        Minecraft Skin Dataset.
        Args:
            data_dir (str): Path to the directory containing skins (*.png) and prompts (*.txt).
            text_encoder (TextEncoder, optional): Encoder to pre-calculate text embeddings.
            transform (callable, optional): PyTorch transform for images.
        """
        self.data_dir = data_dir
        self.text_encoder = text_encoder
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),  # Convert PIL Image to tensor [0.0, 1.0]
            # We scale RGB to [-1, 1], but keeping Alpha in [0, 1] is usually better,
            # or we can scale all channels. Let's normalize RGB to [-1, 1] and Alpha to [-1, 1] 
            # for uniform GAN training, or keep Alpha separate. Let's normalize all 4 channels to [-1, 1].
            transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))
        ])
        
        self.samples = []
        self._load_dataset()

        # Pre-compute text embeddings ONCE (huge speedup vs. encoding in __getitem__)
        self.embeddings = None
        if self.text_encoder is not None and len(self.samples) > 0:
            prompts = [s["prompt"] for s in self.samples]
            print(f"Pre-computing text embeddings for {len(prompts)} prompts...")
            self.embeddings = self.text_encoder.encode(prompts)
            print("Text embeddings cached.")

    def _clean_filename_to_prompt(self, filename):
        """Converts a filename like 'knight_blue_armor_v2.png' into 'knight blue armor v2'"""
        name_without_ext = os.path.splitext(filename)[0]
        # Replace underscores, dashes, dots with spaces
        prompt = re.sub(r'[_.\-]', ' ', name_without_ext)
        # Remove multiple spaces
        prompt = re.sub(r'\s+', ' ', prompt).strip()
        return prompt

    def _load_dataset(self):
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)
            return

        # Check for metadata.json first
        metadata = {}
        metadata_path = os.path.join(self.data_dir, "metadata.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load metadata.json: {e}")

        for file in os.listdir(self.data_dir):
            if file.lower().endswith(".png"):
                img_path = os.path.join(self.data_dir, file)
                base_name = os.path.splitext(file)[0]
                
                # Check options for prompts:
                # 1. metadata.json
                # 2. skin_name.txt file
                # 3. Cleaned filename
                prompt = ""
                if file in metadata:
                    prompt = metadata[file]
                elif base_name in metadata:
                    prompt = metadata[base_name]
                else:
                    txt_path = os.path.join(self.data_dir, f"{base_name}.txt")
                    if os.path.exists(txt_path):
                        try:
                            with open(txt_path, 'r', encoding='utf-8') as f:
                                prompt = f.read().strip()
                        except Exception as e:
                            print(f"Warning: Failed to read {txt_path}: {e}")
                
                # Fallback to filename
                if not prompt:
                    prompt = self._clean_filename_to_prompt(file)
                
                self.samples.append({
                    "image_path": img_path,
                    "prompt": prompt
                })

    @staticmethod
    def _color_jitter_rgba(tensor, hue_range=0.05, brightness_range=0.15):
        """
        Apply random hue shift and brightness jitter to the RGB channels
        of an RGBA tensor while keeping alpha untouched.
        Operates on a (4, H, W) tensor in [-1, 1].
        """
        rgb = tensor[:3]  # (3, H, W)
        alpha = tensor[3:4]  # (1, H, W)

        # Rescale RGB from [-1, 1] to [0, 1] for torchvision functional ops
        rgb01 = (rgb + 1.0) / 2.0

        # Random hue shift
        hue_factor = random.uniform(-hue_range, hue_range)
        rgb01 = TF.adjust_hue(rgb01, hue_factor)

        # Random brightness
        brightness_factor = 1.0 + random.uniform(-brightness_range, brightness_range)
        rgb01 = TF.adjust_brightness(rgb01, brightness_factor)

        rgb01 = rgb01.clamp(0.0, 1.0)

        # Scale back to [-1, 1]
        rgb = rgb01 * 2.0 - 1.0

        return torch.cat([rgb, alpha], dim=0)

    @staticmethod
    def _skin_aware_hflip(tensor):
        """
        Horizontally flip a 64×64 Minecraft skin tensor while swapping
        left↔right body-part regions so the anatomy stays correct.
        """
        out = tensor.clone()
        # Flip every pixel column-wise
        out = torch.flip(out, dims=[2])  # flip W axis

        # Swap left↔right body-part blocks so limbs stay on the correct side.
        # Right Leg (rows 16-32, cols 0-16) ↔ Left Leg (rows 48-64, cols 16-32)
        rl = out[:, 16:32, 48:64].clone()
        ll = out[:, 48:64, 32:48].clone()
        out[:, 16:32, 48:64] = ll
        out[:, 48:64, 32:48] = rl

        # Right Arm (rows 16-32, cols 40-56) ↔ Left Arm (rows 48-64, cols 32-48)
        ra = out[:, 16:32, 8:24].clone()
        la = out[:, 48:64, 16:32].clone()
        out[:, 16:32, 8:24] = la
        out[:, 48:64, 16:32] = ra

        return out

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = sample["image_path"]
        prompt = sample["prompt"]
        
        try:
            # Load as RGBA
            img = Image.open(img_path).convert("RGBA")
            
            # If the image is 64x32 (old Minecraft skin format), convert to 64x64
            if img.size == (64, 32):
                new_img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
                new_img.paste(img, (0, 0))
                # For Minecraft, old 1.7 skins can be converted to 1.8 by copying parts, 
                # but simply padding it to 64x64 keeps the texture layout intact and valid.
                img = new_img
            elif img.size != (64, 64):
                # Resize other dimensions to 64x64
                img = img.resize((64, 64), Image.Resampling.NEAREST)

            # Clean up noise the model should NOT learn:
            # 1. Zero out RGB wherever the pixel is fully transparent (RGB there is arbitrary garbage)
            # 2. Force unused (non-skin) regions to fully transparent black
            np_img = np.array(img)
            transparent = np_img[:, :, 3] == 0
            np_img[transparent] = [0, 0, 0, 0]
            np_img[~get_minecraft_mask()] = [0, 0, 0, 0]
            img = Image.fromarray(np_img, mode="RGBA")

            img_tensor = self.transform(img)

            # Color augmentation: hue/brightness jitter on RGB (50% chance)
            if random.random() < 0.5:
                img_tensor = self._color_jitter_rgba(img_tensor)

            # Skin-aware horizontal flip augmentation (50% chance)
            if random.random() < 0.5:
                img_tensor = self._skin_aware_hflip(img_tensor)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank transparent image tensor in case of error
            img_tensor = torch.zeros((4, 64, 64))
        
        if self.embeddings is not None:
            # Use pre-computed (cached) text embedding
            text_embedding = self.embeddings[idx]
            return img_tensor, text_embedding, prompt
            
        return img_tensor, prompt
