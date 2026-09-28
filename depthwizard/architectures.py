import torch
import torch.nn as nn
import torch.nn.functional as F

def upgrade_to_4_channels(model):
    """
    Prompt2DEM Architecture Upgrade:
    Modifies the HuggingFace DepthAnythingV2 backbone to accept a 4th channel (coarse DEM prompt).
    The 4th channel weights are initialized to zero so the model starts exactly as it was pretrained on RGB.
    """
    old_proj = model.backbone.embeddings.patch_embeddings.projection
    new_proj = nn.Conv2d(4, old_proj.out_channels, 
                         kernel_size=old_proj.kernel_size, 
                         stride=old_proj.stride, 
                         padding=old_proj.padding)
    with torch.no_grad():
        new_proj.weight[:, :3] = old_proj.weight
        new_proj.weight[:, 3] = 0.0  # Zero initialization for the DEM prompt channel
        new_proj.bias = old_proj.bias
        
    model.backbone.embeddings.patch_embeddings.projection = new_proj
    model.config.num_channels = 4
    return model


class BinnedClassificationHead(nn.Module):
    """
    HTC-DC Net Architecture Upgrade:
    Replaces the standard regression head with a Classification-then-Regression head.
    Prevents the network from under-predicting tall structures (the long-tail bias).
    """
    def __init__(self, in_channels, n_bins=64, max_height=100.0):
        super().__init__()
        self.n_bins = n_bins
        self.max_height = max_height
        
        # Output unnormalized logits for the bins
        self.classifier = nn.Conv2d(in_channels, n_bins, kernel_size=1)
        
        # Bin centers (e.g. 0m to 100m)
        bin_centers = torch.linspace(0, max_height, n_bins)
        self.register_buffer('bin_centers', bin_centers.view(1, n_bins, 1, 1))
        
    def forward(self, features):
        logits = self.classifier(features)
        probs = F.softmax(logits, dim=1)
        
        # Expected value (weighted sum of bin centers)
        height = torch.sum(probs * self.bin_centers, dim=1, keepdim=True)
        return height, logits

def upgrade_to_binned_head(model, max_height=100.0):
    """
    Injects the BinnedClassificationHead into a HuggingFace DepthAnythingV2 model.
    """
    # Find the output channel size of the decoder
    in_channels = model.head.head[0].in_channels
    new_head = BinnedClassificationHead(in_channels=in_channels, max_height=max_height)
    model.head.head = new_head
    return model
