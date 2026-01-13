
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffusionModel(nn.Module):
    def __init__(self, channels_in, channels_out, aux_channels=0, model_channels=64):
        super().__init__()
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.aux_channels = aux_channels
        
        # Calculate total input channels
        self.total_input_channels = channels_in + aux_channels
        
        # Initial projection
        self.conv_in = nn.Conv2d(self.total_input_channels, model_channels, kernel_size=3, padding=1)
        
        # Simple backbone (placeholder for a real U-Net)
        self.layer1 = nn.Sequential(
            nn.Conv2d(model_channels, model_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, model_channels),
            nn.SiLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(model_channels, model_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, model_channels),
            nn.SiLU(),
        )
        
        # Output projection
        self.conv_out = nn.Conv2d(model_channels, channels_out, kernel_size=3, padding=1)

    def forward(self, x, conditioning=None):
        """
        Forward pass of the diffusion model.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
            conditioning (torch.Tensor, optional): Auxiliary conditioning tensor of shape (B, C_aux, H, W).
                                                   Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape (B, C_out, H, W).
        """
        if self.aux_channels > 0:
            if conditioning is None:
                # If auxiliary channels are expected but not provided, pad with zeros.
                # This ensures backward compatibility and allows default conditioning=None.
                b, _, h, w = x.shape
                conditioning = torch.zeros((b, self.aux_channels, h, w), device=x.device, dtype=x.dtype)
            
            # Concatenate along channel dimension
            if conditioning.shape[1] != self.aux_channels:
                 raise ValueError(f"Expected conditioning to have {self.aux_channels} channels, but got {conditioning.shape[1]}")
                 
            x = torch.cat([x, conditioning], dim=1)
        
        # Initial convolution
        h = self.conv_in(x)
        
        # Backbone processing
        h = self.layer1(h)
        h = self.layer2(h)
        
        # Output projection
        out = self.conv_out(h)
        return out
