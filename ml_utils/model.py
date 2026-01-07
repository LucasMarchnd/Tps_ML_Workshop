import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

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

def calculate_distribution_from_segmentation(seg_mask, n_classes):
    """Calcule la distribution des classes à partir d'un masque de segmentation."""
    # seg_mask: (B, H, W) ou (H, W)
    # Assure que le masque est sur le CPU et en numpy pour le comptage
    if seg_mask.is_cuda:
        seg_mask = seg_mask.cpu()
        
    # Ajoute une dimension batch si elle n'existe pas
    if seg_mask.dim() == 2:
        seg_mask = seg_mask.unsqueeze(0)
        
    B, H, W = seg_mask.shape
    distributions = torch.zeros((B, n_classes))
    
    for i in range(B):
        mask_np = seg_mask[i].numpy()
        counts = np.bincount(mask_np.flatten(), minlength=n_classes)
        distributions[i] = torch.from_numpy(counts / (H * W))
        
    return distributions.squeeze() # Enlève la dimension batch si B=1


class GlobalPredictor(nn.Module):
    """
    Prédit la distribution des classes directement à partir de l'image entière.
    Utilise un ResNet18 pré-entraîné adapté pour 4 canaux en entrée.
    """
    def __init__(self, in_channels=4, out_features=10):
        super(GlobalPredictor, self).__init__()
        # Charger un ResNet18 pré-entraîné
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Adapter la première couche pour accepter 4 canaux au lieu de 3
        # On copie les poids de la couche originale et on moyenne pour le 4ème canal
        original_weights = self.model.conv1.weight.data
        new_conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        with torch.no_grad():
            new_conv1.weight[:, :3, :, :] = original_weights
            # Initialise le 4ème canal (infrarouge) avec la moyenne des 3 autres
            new_conv1.weight[:, 3, :, :] = torch.mean(original_weights, dim=1)
            
        self.model.conv1 = new_conv1
        
        # Adapter la dernière couche pour prédire les 10 classes
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, out_features)

    def forward(self, x):
        return self.model(x)


class CombinedModel(nn.Module):
    """
    Méta-modèle qui combine les prédictions du U-Net et du GlobalPredictor.
    """
    def __init__(self, unet, global_predictor, n_classes=10):
        super(CombinedModel, self).__init__()
        self.unet = unet
        self.global_predictor = global_predictor
        self.n_classes = n_classes
        
        # Couche de fusion qui apprend à combiner les deux distributions
        # Input: 2 * n_classes (concaténation des deux vecteurs de distribution)
        self.fusion_layer = nn.Sequential(
            nn.Linear(n_classes * 2, n_classes * 2),
            nn.ReLU(),
            nn.BatchNorm1d(n_classes * 2),
            nn.Linear(n_classes * 2, n_classes)
        )

    def forward(self, x):
        # 1. Prédiction de segmentation avec le U-Net
        seg_logits = self.unet(x) # (B, C, H, W)
        
        # 2. Prédiction de distribution globale
        dist_logits_global = self.global_predictor(x) # (B, C)
        
        # 3. Calcul de la distribution à partir de la segmentation
        # On utilise les logits pour garder l'information maximale avant la décision
        # On applique un softmax sur les pixels, puis on moyenne
        # Note: Cette approche est plus différentiable que de faire un argmax
        seg_probs_pixel = F.softmax(seg_logits, dim=1) # (B, C, H, W)
        dist_from_seg = torch.mean(seg_probs_pixel, dim=(2, 3)) # (B, C)
        
        # 4. Concaténer les deux vecteurs de "confiance" (logits/probabilités)
        # On utilise les probabilités ici pour que les deux entrées soient à la même échelle
        combined_dist = torch.cat([dist_from_seg, F.softmax(dist_logits_global, dim=1)], dim=1)
        
        # 5. Fusionner les deux distributions pour obtenir la prédiction finale
        final_dist_logits = self.fusion_layer(combined_dist)
        
        # On retourne les logits finaux, car la fonction de perte (comme CrossEntropy)
        # est plus stable numériquement avec des logits.
        return final_dist_logits
