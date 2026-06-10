import torch
import torch.nn as nn

class GenResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return self.relu(out + residual)

class Generator(nn.Module):
    def __init__(self, latent_dim=128, text_embed_dim=384, out_channels=4):
        """
        Conditional Generator for generating 64x64 Minecraft skins.
        Args:
            latent_dim (int): Dimension of noise vector z.
            text_embed_dim (int): Dimension of text embedding.
            out_channels (int): 4 channels (RGBA) for Minecraft skins.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.text_embed_dim = text_embed_dim
        
        # Combined conditioning input dimension
        self.input_dim = latent_dim + text_embed_dim
        
        # Project combined vector to a 4x4 feature map
        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 512 * 4 * 4, bias=False),
            nn.BatchNorm1d(512 * 4 * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Conv/ResNet Blocks with Upsampling: 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64
        # We use nearest-neighbor upsampling followed by convolutions to prevent checkerboard artifacts.
        self.layer1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'), # 4x4 -> 8x8
            GenResBlock(512, 256)
        )
        self.layer2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'), # 8x8 -> 16x16
            GenResBlock(256, 128)
        )
        self.layer3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'), # 16x16 -> 32x32
            GenResBlock(128, 64)
        )
        self.layer4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'), # 32x32 -> 64x64
            GenResBlock(64, 32)
        )
        
        # Output layer
        self.to_rgba = nn.Sequential(
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
            nn.Tanh() # Scale to [-1, 1] for all channels (including alpha)
        )

    def forward(self, z, text_embed):
        """
        Args:
            z (torch.Tensor): Latent noise of shape (batch_size, latent_dim).
            text_embed (torch.Tensor): Text embedding of shape (batch_size, text_embed_dim).
        Returns:
            torch.Tensor: Generated image of shape (batch_size, 4, 64, 64).
        """
        # Concatenate latent noise and text embeddings
        x = torch.cat([z, text_embed], dim=1)
        
        # Project and reshape
        x = self.fc(x)
        x = x.view(-1, 512, 4, 4)
        
        # Upsample blocks
        x = self.layer1(x)  # 8x8
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 32x32
        x = self.layer4(x)  # 64x64
        
        # Output RGBA
        rgba = self.to_rgba(x)
        return rgba
