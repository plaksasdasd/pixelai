import torch
import torch.nn as nn

def layer_norm(num_channels):
    # WGAN-GP forbids BatchNorm/InstanceNorm in the critic because they break the
    # per-sample 1-Lipschitz constraint that the gradient penalty enforces. The paper
    # recommends LayerNorm (or no normalization). GroupNorm with a single group is a
    # convolutional LayerNorm: it normalizes over all channels/spatial positions of each
    # sample independently, with no batch statistics and no running state.
    return nn.GroupNorm(1, num_channels)

class DiscResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = layer_norm(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = layer_norm(out_channels)
        
        # Shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                layer_norm(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        return self.relu(out + residual)

class Discriminator(nn.Module):
    def __init__(self, text_embed_dim=384, in_channels=4):
        """
        Conditional Discriminator (Critic) for WGAN-GP.
        Args:
            text_embed_dim (int): Dimension of text embedding.
            in_channels (int): 4 channels (RGBA) for Minecraft skins.
        """
        super().__init__()
        
        # Downsampling blocks: 64x64 -> 32x32 -> 16x16 -> 8x8 -> 4x4
        self.conv_in = nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1) # 64x64 -> 32x32
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        
        self.block1 = DiscResBlock(64, 128, stride=2)   # 32x32 -> 16x16
        self.block2 = DiscResBlock(128, 256, stride=2)  # 16x16 -> 8x8
        self.block3 = DiscResBlock(256, 512, stride=2)  # 8x8 -> 4x4
        
        # Image features projection
        self.img_fc = nn.Sequential(
            nn.Linear(512 * 4 * 4, 512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Text embedding projection
        self.text_fc = nn.Sequential(
            nn.Linear(text_embed_dim, 512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Final evaluation layers (WGAN critic output: a single scalar, NO sigmoid!)
        self.final_eval = nn.Linear(512, 1)

    def forward(self, img, text_embed):
        """
        Args:
            img (torch.Tensor): Image tensor of shape (batch_size, 4, 64, 64).
            text_embed (torch.Tensor): Text embedding of shape (batch_size, text_embed_dim).
        Returns:
            torch.Tensor: Scalar validity score of shape (batch_size, 1).
        """
        # Process image
        x = self.conv_in(img)
        x = self.relu(x)
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        # Flatten and project image features
        img_features = x.view(x.size(0), -1)
        img_features = self.img_fc(img_features)
        
        # 1. Unconditional realism score
        realism_score = self.final_eval(img_features)
        
        # 2. Conditional matching score (Projection Discriminator math)
        text_features = self.text_fc(text_embed)
        matching_score = torch.sum(img_features * text_features, dim=1, keepdim=True)
        
        # Combine scores
        validity = realism_score + matching_score
        return validity
