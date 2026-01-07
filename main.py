from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from ml_utils.data import LandCoverData, LandCoverDataset
from ml_utils.viz import visualize_sample, show_image
# Import des nouveaux modèles
from ml_utils.model import SimpleUNet, GlobalPredictor, CombinedModel
import matplotlib.pyplot as plt

# --- NOUVELLE FONCTION UTILITAIRE ---
def get_distribution_from_mask(masks, n_classes):
    """
    Calcule la distribution de classes pour un batch de masques.
    
    Args:
        masks (Tensor): Batch de masques de segmentation (B, H, W).
        n_classes (int): Nombre de classes.
        
    Returns:
        Tensor: Batch de distributions (B, n_classes).
    """
    B, H, W = masks.shape
    distributions = torch.zeros((B, n_classes), device=masks.device)
    for i in range(B):
        # bincount est très efficace pour compter les occurrences
        counts = torch.bincount(masks[i].flatten(), minlength=n_classes)
        distributions[i] = counts.float() / (H * W)
    return distributions

# --- FONCTION D'ENTRAÎNEMENT MODIFIÉE ---
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20, device='cpu', n_classes=10):
    """
    Fonction d'entraînement adaptée pour le CombinedModel qui prédit une distribution.
    """
    print("Début de l'entraînement du modèle combiné...")
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            # Calculer la distribution de vérité terrain à partir des masques
            target_dists = get_distribution_from_mask(masks, n_classes).to(device)
            
            optimizer.zero_grad()
            
            # Le modèle retourne les logits de la distribution finale
            output_logits = model(images)
            
            # Appliquer le log_softmax car KLDivLoss attend des log-probabilités en entrée
            output_log_probs = F.log_softmax(output_logits, dim=1)
            
            # Calcul de la perte entre la log-distribution prédite et la distribution cible
            loss = criterion(output_log_probs, target_dists)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                target_dists = get_distribution_from_mask(masks, n_classes).to(device)
                
                output_logits = model(images)
                output_log_probs = F.log_softmax(output_logits, dim=1)
                
                loss = criterion(output_log_probs, target_dists)
                val_loss += loss.item() * images.size(0)
                
        val_loss = val_loss / len(val_loader.dataset)
        
        print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_loss:.4f} - Val Loss: {val_loss:.4f}')
        
    print("Entraînement terminé !")
    return model

def main():
    DATA_FOLDER = Path('/Users/alice/Documents/Cours/TPS/ML_Workshop/dataset/').expanduser()
    
    train_images_paths = sorted(list(DATA_FOLDER.glob('train/images/*.tif')))
    train_masks_paths = sorted(list(DATA_FOLDER.glob('train/masks/*.tif')))
    
    print(f'Number of train images: {len(train_images_paths)}')
    print(f'Number of train masks: {len(train_masks_paths)}')

    # --- NOUVELLE VÉRIFICATION ---
    if not train_images_paths or not train_masks_paths:
        print("\nERREUR: Aucune image ou masque d'entraînement n'a été trouvé.")
        print("Veuillez vérifier que le dossier './dataset/train/' contient bien vos images et masques.")
        return # Quitte la fonction main

    # if len(train_images_paths) > 0:
    #     idx = random.choice(range(len(train_images_paths)))
    #     visualize_sample(train_images_paths[idx], train_masks_paths[idx])

    print("Préparation des données...")
    subset_size = 1000
    full_dataset = LandCoverDataset(train_images_paths[:subset_size], train_masks_paths[:subset_size])
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    BATCH_SIZE = 8
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Données prêtes: {len(train_dataset)} images d'entraînement, {len(val_dataset)} images de validation.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilisation du device: {device}")
    
    # --- INITIALISATION DES MODÈLES ---
    # 1. Charger le U-Net (peut être pré-entraîné si vous le souhaitez)
    unet_model = SimpleUNet(in_channels=LandCoverData.N_CHANNELS, out_channels=LandCoverData.N_CLASSES)
    # Optionnel: Charger les poids si vous avez déjà un U-Net entraîné
    # unet_model.load_state_dict(torch.load('path_to_your_unet.pth'))

    # 2. Créer le prédicteur global
    global_predictor = GlobalPredictor(in_channels=LandCoverData.N_CHANNELS, out_features=LandCoverData.N_CLASSES)
    
    # 3. Créer le modèle combiné
    model = CombinedModel(unet=unet_model, global_predictor=global_predictor, n_classes=LandCoverData.N_CLASSES)
    model = model.to(device)
    
    MODEL_PATH = 'combined_model.pth'
    
    if Path(MODEL_PATH).exists():
        print(f"Modèle trouvé à '{MODEL_PATH}'. Chargement du modèle existant...")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        
        # --- ÉVALUATION DU MODÈLE ---
        print("\nDébut de l'évaluation du modèle sur le jeu de validation...")
        evaluate_model(model, val_loader, device=device, n_classes=LandCoverData.N_CLASSES)
        
    else:
        print("Aucun modèle trouvé. Démarrage de l'entraînement...")
        
        criterion = nn.KLDivLoss(reduction='batchmean')
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=80, device=device, n_classes=LandCoverData.N_CLASSES)
        
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Modèle sauvegardé sous '{MODEL_PATH}'")

    print("Script terminé.")

# --- NOUVELLE FONCTION D'ÉVALUATION ---
def evaluate_model(model, data_loader, device, n_classes):
    """
    Évalue le modèle sur un jeu de données et affiche la perte de divergence KL moyenne.
    """
    model.eval() # Mode évaluation
    total_kl_div = 0.0
    criterion = nn.KLDivLoss(reduction='sum') # 'sum' pour un contrôle plus fin

    with torch.no_grad():
        for images, masks in data_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            # Calculer la distribution de vérité terrain
            target_dists = get_distribution_from_mask(masks, n_classes).to(device)
            
            # Prédiction du modèle
            output_logits = model(images)
            output_log_probs = F.log_softmax(output_logits, dim=1)
            
            # Calcul de la divergence KL pour le batch
            # Note: KL(P||Q) = sum(P * log(P/Q))
            # PyTorch's KLDivLoss calcule sum(Q * (log(Q) - P)) où P est l'input (log-probs) et Q est la target (probs)
            # Pour avoir la vraie KL divergence, il faut que l'input soit log(pred) et target soit gt
            kl_div = criterion(output_log_probs, target_dists)
            total_kl_div += kl_div.item()

    # Calcul de la moyenne
    avg_kl_div = total_kl_div / len(data_loader.dataset)
    print(f"\n--- Performance du Modèle ---")
    print(f"Divergence de Kullback-Leibler (KL) moyenne par échantillon: {avg_kl_div:.6f}")
    print("----------------------------")
    print("(Plus cette valeur est proche de 0, meilleure est la performance)")

if __name__ == "__main__":
    main()
