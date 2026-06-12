import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleUNet(nn.Module):
    def __init__(self, in_channels, out_channels, base_filters=32):
        super(SimpleUNet, self).__init__()
        
        # Encoder
        self.enc1 = self.conv_block(in_channels, base_filters)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = self.conv_block(base_filters, base_filters * 2)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = self.conv_block(base_filters * 2, base_filters * 4)
        self.pool3 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = self.conv_block(base_filters * 4, base_filters * 8)
        
        # Decoder
        self.up3 = nn.ConvTranspose2d(base_filters * 8, base_filters * 4, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(base_filters * 8, base_filters * 4)
        
        self.up2 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(base_filters * 4, base_filters * 2)
        
        self.up1 = nn.ConvTranspose2d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(base_filters * 2, base_filters)
        
        # Output layer
        self.final = nn.Conv2d(base_filters, out_channels, kernel_size=1)

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        
        # Bottleneck
        b = self.bottleneck(p3)
        
        # Decoder
        d3 = self.up3(b)
        # Skip connection concatenation
        # Ensure shapes match if input size wasn't perfect power of 2 (though 256 is)
        if d3.shape != e3.shape:
            d3 = F.interpolate(d3, size=e3.shape[2:], mode='bilinear', align_corners=True)
        d3 = torch.cat((e3, d3), dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        if d2.shape != e2.shape:
            d2 = F.interpolate(d2, size=e2.shape[2:], mode='bilinear', align_corners=True)
        d2 = torch.cat((e2, d2), dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        if d1.shape != e1.shape:
            d1 = F.interpolate(d1, size=e1.shape[2:], mode='bilinear', align_corners=True)
        d1 = torch.cat((e1, d1), dim=1)
        d1 = self.dec1(d1)
        
        return self.final(d1)

def kl_divergence(p_logits, q_labels, reduction='batchmean'):
    """
    Calculates the Kullback-Leibler Divergence between two probability distributions.

    Args:
        p_logits (torch.Tensor): Logits from the model (predicted distribution).
                                 Shape: (N, C, H, W)
        q_labels (torch.Tensor): True labels, typically one-hot encoded.
                                 Shape: (N, C, H, W)
        reduction (str): Specifies the reduction to apply to the output:
                         'none' | 'batchmean' | 'sum' | 'mean'. 'batchmean' is the default
                         which is the sum divided by the batch size.

    Returns:
        torch.Tensor: The KL divergence.
    """
    # Apply log_softmax to the predicted logits to get log-probabilities
    p_log_softmax = F.log_softmax(p_logits, dim=1)

    # Ensure q_labels are probabilities (e.g., one-hot encoded)
    # If q_labels are class indices, they need to be converted to one-hot first.
    # Assuming q_labels are already one-hot encoded probabilities here.

    # F.kl_div expects log-probabilities for the input (p) and probabilities for the target (q)
    # The formula for KL divergence is sum(q * log(q / p))
    # F.kl_div computes sum(q * (log(q) - p_log_softmax))
    # So, if q is the target distribution and p_log_softmax is log(p), this is correct.
    return F.kl_div(p_log_softmax, q_labels, reduction=reduction)
