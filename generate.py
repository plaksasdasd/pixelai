import os
import argparse
import torch
from models.generator import Generator
from models.text_encoder import TextEncoder
from utils.visualization import save_skin, tensor_to_pil

def _resolve_weights(explicit_path):
    """
    Resolve generator weights path. Prefers EMA checkpoints for smoother
    inference, falling back to raw weights for backward compatibility.
    """
    candidates = [
        "checkpoints/generator_ema_latest.pth",
        "checkpoints/generator_ema_final.pth",
        "checkpoints/generator_latest.pth",
        "checkpoints/generator_final.pth",
        "generator_latest.pth",
        "generator_final.pth",
    ]
    if explicit_path and os.path.exists(explicit_path):
        return explicit_path
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"Could not find model weights at '{explicit_path}'. "
        "Make sure you train the model first or place the weights file correctly."
    )


def generate_skin(prompt, weights_path=None, output_path="output_skin.png", latent_dim=128, device=None):
    """
    Generates a Minecraft skin from a text prompt.
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    weights_path = _resolve_weights(weights_path)

    # Initialize Text Encoder
    print("Loading Text Encoder...")
    text_encoder = TextEncoder(device=device)
    text_embed_dim = text_encoder.embedding_dim
    
    # Initialize Generator and load weights
    print(f"Loading Generator from: {weights_path}...")
    netG = Generator(latent_dim=latent_dim, text_embed_dim=text_embed_dim).to(device)
    
    # Load state dict with map_location
    state_dict = torch.load(weights_path, map_location=device)
    netG.load_state_dict(state_dict)
    netG.eval()
    
    # Encode prompt
    print(f"Encoding prompt: '{prompt}'...")
    text_embed = text_encoder.encode(prompt).unsqueeze(0).to(device) # Shape: (1, text_embed_dim)
    
    # Generate latent noise
    z = torch.randn(1, latent_dim, device=device)
    
    # Generate image
    print("Generating skin...")
    with torch.no_grad():
        fake_skin = netG(z, text_embed)[0] # Take first from batch (shape: 4, 64, 64)
    
    # Save image
    save_skin(fake_skin, output_path)
    print(f"Skin successfully generated and saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Minecraft Skin from Text Prompt")
    parser.add_argument("--prompt", type=str, required=True, help="Description of the skin (e.g. 'ninja with red eyes')")
    parser.add_argument("--weights", type=str, default=None, help="Path to generator weights (.pth). Auto-resolves EMA weights if omitted.")
    parser.add_argument("--output", type=str, default="generated_skin.png", help="Path to save the output PNG file")
    parser.add_argument("--latent_dim", type=int, default=128, help="Size of noise vector z")
    args = parser.parse_args()
    
    generate_skin(
        prompt=args.prompt,
        weights_path=args.weights,
        output_path=args.output,
        latent_dim=args.latent_dim
    )
