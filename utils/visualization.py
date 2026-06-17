import os
import io
import base64
import torch
import numpy as np
from PIL import Image

def get_minecraft_mask():
    """
    Returns a numpy boolean mask of shape (64, 64) where True means the pixel 
    is ALLOWED to be opaque (part of the skin), and False means it MUST be transparent.
    This fixes issues with launchers rejecting AI skins due to noisy backgrounds.
    """
    mask = np.zeros((64, 64), dtype=bool)
    
    # Head (top/bottom, right/left, front/back) [0-16, 0-32]
    mask[0:16, 0:32] = True
    # Hat/Helmet overlay [0-16, 32:64]
    mask[0:16, 32:64] = True
    
    # Right Leg (16-32, 0-16)
    mask[16:32, 0:16] = True
    # Torso (16-32, 16-40)
    mask[16:32, 16:40] = True
    # Right Arm (16-32, 40-56)
    mask[16:32, 40:56] = True
    
    # Left Leg (48-64, 16-32)
    mask[48:64, 16:32] = True
    # Left Arm (48-64, 32-48)
    mask[48:64, 32:48] = True
    
    # Right Leg Overlay (32-48, 0-16)
    mask[32:48, 0:16] = True
    # Torso Overlay (32-48, 16-40)
    mask[32:48, 16:40] = True
    # Right Arm Overlay (32-48, 40-56)
    mask[32:48, 40:56] = True
    
    # Left Leg Overlay (48-64, 0-16)
    mask[48:64, 0:16] = True
    # Left Arm Overlay (48-64, 48-64)
    mask[48:64, 48:64] = True
    
    return mask

def get_minecraft_mask_tensor(device="cpu"):
    """
    Returns a float tensor of shape (1, 1, 64, 64) with 1.0 for allowed skin
    pixels and 0.0 for background. Suitable for element-wise multiplication
    with image tensors in [-1, 1] range (background becomes 0, which maps
    to mid-grey; the critic sees identical background in real and fake).
    """
    np_mask = get_minecraft_mask().astype(np.float32)  # (64, 64)
    t = torch.from_numpy(np_mask).unsqueeze(0).unsqueeze(0)  # (1,1,64,64)
    return t.to(device)


def apply_skin_mask(images, mask_tensor):
    """
    Zero out non-skin pixels in a batch of images.
    Args:
        images: (B, C, 64, 64) tensor in [-1, 1]
        mask_tensor: (1, 1, 64, 64) float mask (1=skin, 0=background)
    Returns:
        Masked images with background pixels set to 0.
    """
    return images * mask_tensor


def tensor_to_pil(tensor):
    """
    Converts a PyTorch tensor of shape (4, 64, 64) with values in [-1, 1]
    to a PIL RGBA Image.
    """
    # Clone and detach
    t = tensor.detach().cpu().clone()
    
    # Scale from [-1, 1] to [0, 1]
    t = (t + 1.0) / 2.0
    t = torch.clamp(t, 0.0, 1.0)
    
    # Convert to numpy array (H, W, C)
    np_img = t.numpy()
    np_img = np.transpose(np_img, (1, 2, 0)) # [64, 64, 4]
    
    # Convert to uint8 [0, 255]
    np_img = (np_img * 255.0).astype(np.uint8)
    
    # Binarize alpha: Minecraft skin pixels must be fully opaque or fully transparent.
    # Semi-transparent pixels render as visual noise in launchers/viewers.
    alpha = np_img[:, :, 3]
    np_img[:, :, 3] = np.where(alpha >= 128, 255, 0).astype(np.uint8)
    np_img[np_img[:, :, 3] == 0] = [0, 0, 0, 0]
    
    # Apply hard mask to background to ensure compatibility with Minecraft launchers
    mask = get_minecraft_mask()
    np_img[~mask] = [0, 0, 0, 0] # Set background pixels strictly to completely transparent
    
    return Image.fromarray(np_img, mode="RGBA")

def pil_to_base64(pil_img):
    """Converts PIL Image to base64 string for embedding in HTML/Markdown."""
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def save_skin(tensor, filepath):
    """Saves a tensor skin to the given filepath as PNG."""
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)
    img = tensor_to_pil(tensor)
    img.save(filepath, "PNG")

def generate_3d_viewer_html(pil_img_or_base64, width=300, height=400, auto_rotate=True):
    """
    Generates HTML string containing a 3D skin viewer powered by skinview3d.
    The skin is embedded directly as a base64 Data URL.
    """
    if isinstance(pil_img_or_base64, Image.Image):
        base64_str = pil_to_base64(pil_img_or_base64)
    else:
        base64_str = pil_img_or_base64
        
    skin_data_url = f"data:image/png;base64,{base64_str}"
    
    auto_rotate_js = "skinViewer.autoRotate = true; skinViewer.autoRotateSpeed = 1.0;" if auto_rotate else ""
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Minecraft 3D Skin Viewer</title>
        <!-- Load skinview3d from UNPKG CDN -->
        <script src="https://unpkg.com/skinview3d@3.0.0-alpha.1/dist/skinview3d.bundle.js"></script>
        <style>
            body {{
                margin: 0;
                padding: 0;
                background-color: transparent;
                overflow: hidden;
                display: flex;
                justify-content: center;
                align-items: center;
            }}
            #skin-container {{
                width: {width}px;
                height: {height}px;
            }}
            canvas {{
                outline: none;
                display: block;
            }}
        </style>
    </head>
    <body>
        <div id="skin-container"></div>
        <script>
            const skinViewer = new skinview3d.SkinViewer({{
                container: document.getElementById("skin-container"),
                width: {width},
                height: {height},
                skin: "{skin_data_url}"
            }});
            
            // Apply standard options
            skinViewer.camera.position.x = -15;
            skinViewer.camera.position.y = 15;
            skinViewer.camera.position.z = 24;
            skinViewer.zoom = 0.95;
            
            // Add a walking animation
            skinViewer.animation = new skinview3d.WalkingAnimation();
            skinViewer.animation.speed = 0.8;
            
            {auto_rotate_js}
        </script>
    </body>
    </html>
    """
    return html_content
