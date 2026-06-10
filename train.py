import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import MinecraftSkinDataset
from models.text_encoder import TextEncoder
from models.generator import Generator
from models.discriminator import Discriminator
from utils.visualization import save_skin

def compute_gradient_penalty(critic, real_images, fake_images, text_embeddings, device):
    """
    Computes the gradient penalty for WGAN-GP.
    """
    batch_size = real_images.size(0)
    
    # Random weight term epsilon for interpolation between real and fake images
    alpha = torch.rand((batch_size, 1, 1, 1), device=device)
    
    # Get random interpolation between real and fake images
    interpolated = (alpha * real_images + ((1 - alpha) * fake_images)).requires_grad_(True)
    
    # Calculate critic scores on interpolated images
    critic_interpolates = critic(interpolated, text_embeddings)
    
    # Get gradients of critic scores w.r.t interpolated images
    gradients = torch.autograd.grad(
        outputs=critic_interpolates,
        inputs=interpolated,
        grad_outputs=torch.ones_like(critic_interpolates, device=device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    # Flatten the gradients to compute L2 norm
    gradients = gradients.view(batch_size, -1)
    
    # Compute L2 norm of gradients
    gradient_norm = gradients.norm(2, dim=1)
    
    # Compute gradient penalty: penalty = lambda * (norm - 1)^2
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1 or classname.find('InstanceNorm') != -1:
        if m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)

def train(data_dir, checkpoint_dir="checkpoints", samples_dir="samples", epochs=100, batch_size=16, 
          g_lr=1e-4, d_lr=2e-4, n_critic=5, gp_lambda=10.0, latent_dim=128, device=None, callbacks=None,
          resume=False, epoch_offset=0):
    """
    Main training function for Minecraft Skin cWGAN-GP.
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)
    
    # Initialize Models
    text_encoder = TextEncoder(device=device)
    text_embed_dim = text_encoder.embedding_dim
    
    netG = Generator(latent_dim=latent_dim, text_embed_dim=text_embed_dim).to(device)
    netD = Discriminator(text_embed_dim=text_embed_dim).to(device)
    
    if resume:
        generator_path = os.path.join(checkpoint_dir, "generator_latest.pth")
        discriminator_path = os.path.join(checkpoint_dir, "discriminator_latest.pth")
        if not os.path.exists(generator_path):
            generator_path = os.path.join(checkpoint_dir, "generator_final.pth")
        if not os.path.exists(discriminator_path):
            discriminator_path = os.path.join(checkpoint_dir, "discriminator_final.pth")
        if os.path.exists(generator_path):
            netG.load_state_dict(torch.load(generator_path, map_location=device))
            print(f"Resumed generator from: {generator_path}")
        else:
            print("Warning: generator checkpoint not found. Starting generator from scratch.")
        if os.path.exists(discriminator_path):
            netD.load_state_dict(torch.load(discriminator_path, map_location=device))
            print(f"Resumed discriminator from: {discriminator_path}")
        else:
            print("Warning: discriminator checkpoint not found. Starting discriminator from scratch.")
    else:
        # Apply standard GAN weight initialization to "start from scratch" cleanly
        netG.apply(init_weights)
        netD.apply(init_weights)
    
    # Optimizers (WGAN-GP uses Adam with beta1=0.0, beta2=0.9 or 0.99 for stability)
    optG = optim.Adam(netG.parameters(), lr=g_lr, betas=(0.0, 0.9))
    optD = optim.Adam(netD.parameters(), lr=d_lr, betas=(0.0, 0.9))
    
    # Load dataset
    print("Initializing dataset...")
    dataset = MinecraftSkinDataset(data_dir=data_dir, text_encoder=text_encoder)
    
    if len(dataset) == 0:
        print(f"Error: No images found in '{data_dir}'. Add some .png skins to start training!")
        return None, None
        
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    print(f"Dataset loaded with {len(dataset)} images. Number of batches per epoch: {len(dataloader)}")
    
    # Fixed noise & text for tracking training progress
    # Let's generate samples using some cool standard Minecraft concepts!
    fixed_noise = torch.randn(4, latent_dim, device=device)
    sample_prompts = [
        "creeper boy in blue hoodie",
        "red knight in heavy iron armor",
        "cyberpunk neon green ninja",
        "golden king with a royal crown"
    ]
    # Keep standard Russian fallback if needed, sentence-transformer matches them to the same embedding space!
    fixed_embeddings = text_encoder.encode(sample_prompts).to(device)
    
    # Loss histories
    g_losses = []
    d_losses = []
    
    print("Starting training...")
    for local_epoch in range(1, epochs + 1):
        epoch = epoch_offset + local_epoch
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        epoch_real_score = 0.0
        epoch_fake_score = 0.0
        epoch_gp = 0.0
        epoch_g_updates = 0
        
        # Use tqdm for neat terminal visualization
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{epoch_offset + epochs}")
        for i, (real_imgs, text_embeds, prompts) in enumerate(progress_bar):
            real_imgs = real_imgs.to(device)
            text_embeds = text_embeds.to(device)
            
            b_size = real_imgs.size(0)
            
            # ----------------------------------------------------
            # 1. Train Discriminator (Critic) n_critic times
            # ----------------------------------------------------
            netD.zero_grad()
            
            # Generate fake images
            noise = torch.randn(b_size, latent_dim, device=device)
            fake_imgs = netG(noise, text_embeds)
            
            # Critic scores
            real_validity = netD(real_imgs, text_embeds)
            fake_validity = netD(fake_imgs.detach(), text_embeds)
            real_score = torch.mean(real_validity)
            fake_score = torch.mean(fake_validity)
            
            # Gradient penalty
            gp = compute_gradient_penalty(netD, real_imgs, fake_imgs.detach(), text_embeds, device)
            
            # Critic loss (WGAN critic maximizes: real_score - fake_score, so we minimize negative of that)
            # Small drift penalty keeps critic outputs from drifting to huge magnitudes
            drift = 1e-3 * torch.mean(real_validity ** 2)
            d_loss = fake_score - real_score + gp_lambda * gp + drift
            
            d_loss.backward()
            optD.step()
            
            epoch_d_loss += d_loss.item()
            epoch_real_score += real_score.item()
            epoch_fake_score += fake_score.item()
            epoch_gp += gp.item()
            
            # ----------------------------------------------------
            # 2. Train Generator every n_critic steps
            # ----------------------------------------------------
            if (i + 1) % n_critic == 0 or (i + 1) == len(dataloader):
                netG.zero_grad()
                
                # Sample noise again or reuse
                noise = torch.randn(b_size, latent_dim, device=device)
                gen_imgs = netG(noise, text_embeds)
                
                # Generator loss: maximize fake scores (minimize negative fake score)
                g_loss = -torch.mean(netD(gen_imgs, text_embeds))
                
                g_loss.backward()
                optG.step()
                
                epoch_g_loss += g_loss.item()
                epoch_g_updates += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'D_loss': f"{d_loss.item():.4f}", 
                'G_loss': f"{epoch_g_loss / max(1, epoch_g_updates):.4f}",
                'D_real': f"{real_score.item():.4f}",
                'D_fake': f"{fake_score.item():.4f}",
                'GP': f"{gp.item():.4f}"
            })
            
        # Average losses for the epoch
        avg_d_loss = epoch_d_loss / len(dataloader)
        avg_g_loss = epoch_g_loss / max(1, epoch_g_updates)
        avg_real_score = epoch_real_score / len(dataloader)
        avg_fake_score = epoch_fake_score / len(dataloader)
        avg_gp = epoch_gp / len(dataloader)
        g_losses.append(avg_g_loss)
        d_losses.append(avg_d_loss)
        
        # Periodic output
        if local_epoch % 10 == 0 or local_epoch == 1 or local_epoch == epochs:
            # Generate and save fixed evaluation samples
            netG.eval()
            with torch.no_grad():
                gen_samples = netG(fixed_noise, fixed_embeddings)
                for idx, sample in enumerate(gen_samples):
                    safe_prompt = sample_prompts[idx].replace(" ", "_")
                    filename = f"epoch_{epoch}_{safe_prompt}.png"
                    save_skin(sample, os.path.join(samples_dir, filename))
            netG.train()
            
            # Save checkpoints
            torch.save(netG.state_dict(), os.path.join(checkpoint_dir, "generator_latest.pth"))
            torch.save(netD.state_dict(), os.path.join(checkpoint_dir, "discriminator_latest.pth"))
            
            # Checkpoint at specific milestones
            if epoch % 50 == 0:
                torch.save(netG.state_dict(), os.path.join(checkpoint_dir, f"generator_epoch_{epoch}.pth"))
        
        # Invoke callback if supplied (useful for updating Streamlit UI charts)
        if callbacks is not None:
            diagnostics = {
                "critic_real": avg_real_score,
                "critic_fake": avg_fake_score,
                "gradient_penalty": avg_gp,
                "generator_updates": epoch_g_updates
            }
            keep_running = callbacks(epoch, avg_d_loss, avg_g_loss, diagnostics)
            if keep_running is False:
                print("Training stopped early by callback request.")
                break
            
    # Save final models
    torch.save(netG.state_dict(), os.path.join(checkpoint_dir, "generator_final.pth"))
    torch.save(netD.state_dict(), os.path.join(checkpoint_dir, "discriminator_final.pth"))
    print("Training finished! Models saved successfully.")
    
    return netG, netD

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Minecraft Skin Generator (cWGAN-GP)")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to folder with .png skins and prompts")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save weights")
    parser.add_argument("--samples_dir", type=str, default="samples", help="Directory to save training sample images")
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--g_lr", type=float, default=1e-4, help="Generator learning rate")
    parser.add_argument("--d_lr", type=float, default=2e-4, help="Discriminator learning rate")
    parser.add_argument("--n_critic", type=int, default=5, help="Number of critic updates per generator update")
    parser.add_argument("--gp_lambda", type=float, default=10.0, help="Gradient penalty weight")
    parser.add_argument("--latent_dim", type=int, default=128, help="Size of noise vector z")
    parser.add_argument("--resume", action="store_true", help="Resume from latest generator/discriminator checkpoints")
    parser.add_argument("--epoch_offset", type=int, default=0, help="Epoch number offset for resumed training")
    args = parser.parse_args()
    
    train(
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        samples_dir=args.samples_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        g_lr=args.g_lr,
        d_lr=args.d_lr,
        n_critic=args.n_critic,
        gp_lambda=args.gp_lambda,
        latent_dim=args.latent_dim,
        resume=args.resume,
        epoch_offset=args.epoch_offset
    )
