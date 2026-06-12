from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F

from ml_utils.data import load_dataset, get_array_from_path, LandCoverData, LandCoverDataset
from ml_utils.viz import visualize_sample, show_image, show_mask
from ml_utils.model import SimpleUNet, kl_divergence
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import os

# Conseils :
# Faire des statistiques sur les images classes déséquilibrées 
# Faire un subset des classes sympas avec les classes
# Faire des augmentations de données (rotations, flips, etc.)
# Regrouper des classes classe végétales similaires
# Je crois que seulement 7/10 sont bcp utilisées,les classes : snow cloud et no_data sont peu utilisées

def predict_and_visualize(model, dataset, device, num_samples=3):
    print(f"Visualisation de {num_samples} prédictions...")
    model.eval()
    
    if len(dataset) < num_samples:
        indices = range(len(dataset))
    else:
        indices = random.sample(range(len(dataset)), num_samples)
    
    for idx in indices:
        sample = dataset[idx]
        if isinstance(sample, tuple):
            image, mask = sample
            has_mask = True
        else:
            image = sample
            mask = None
            has_mask = False
        
        input_tensor = image.unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
            
        img_display = image.permute(1, 2, 0).numpy().astype(np.float32)
        # Re-scale for display if it was normalized to 0-1
        # Visualization expects distinct colors, but here we just show the image
        
        classes_colorpalette = {c: color/255. for (c, color) in LandCoverData.CLASSES_COLORPALETTE.items()}
        color_mask = np.zeros((*pred_mask.shape, 3))
        for c, color in classes_colorpalette.items():
            color_mask[pred_mask == c] = color

        fig, ax = plt.subplots(figsize=(10, 8))
        plt.subplots_adjust(bottom=0.25)
        
        # Display expects [0, 1] floats or [0, 255] ints. 
        # Our images are now [0, 1] float32.
        show_image(img_display, display_min=0, display_max=1, ax=ax)
        ax.set_title(f"Prédiction (Sample {idx})")
        
        initial_alpha = 0.5
        mask_im = ax.imshow(color_mask, alpha=initial_alpha)
        
        ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        slider = Slider(ax_slider, 'Opacité', 0.0, 1.0, valinit=initial_alpha)
        
        def update(val):
            mask_im.set_alpha(slider.val)
            fig.canvas.draw_idle()
            
        slider.on_changed(update)
        
        import matplotlib.patches as mpatches
        handles = []
        for c, color in classes_colorpalette.items():
            handles.append(mpatches.Patch(color=color, label=LandCoverData.CLASSES[c]))
        ax.legend(handles=handles, loc='upper right', bbox_to_anchor=(1.3, 1))
        
        print(f"Affichage de l'image {idx}. Fermez la fenêtre pour passer à la suivante.")
        plt.show()

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20, device='cpu'):
    print("Début de l'entraînement...")
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for images, masks in train_loader:
            if torch.isnan(images).any() or torch.isinf(images).any():
                print("Warning: NaN or Inf found in inputs. Skipping batch.")
                continue
                
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            
            loss = criterion(outputs, masks)
            
            loss.backward()
            
            # Gradient clipping pour eviter l'explosion du gradient (essentiel avec des poids eleves)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        
        # IoU stats
        total_intersections = np.zeros(LandCoverData.N_CLASSES)
        total_unions = np.zeros(LandCoverData.N_CLASSES)
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)
                
                # IoU Calculation
                preds = torch.argmax(outputs, dim=1)
                
                for cls in range(LandCoverData.N_CLASSES):
                    if cls in LandCoverData.IGNORED_CLASSES_IDX:
                        continue
                        
                    pred_inds = (preds == cls)
                    target_inds = (masks == cls)
                    
                    intersection = (pred_inds & target_inds).sum().item()
                    union = (pred_inds | target_inds).sum().item()
                    
                    total_intersections[cls] += intersection
                    total_unions[cls] += union
                
        val_loss = val_loss / len(val_loader.dataset)
        
        # Compute IoU per class and Mean IoU
        ious = []
        valid_classes_iou = []
        for cls in range(LandCoverData.N_CLASSES):
            if cls in LandCoverData.IGNORED_CLASSES_IDX:
                ious.append(np.nan)
                continue
                
            if total_unions[cls] == 0:
                iou = np.nan # Class not present in this epoch's validation set
            else:
                iou = total_intersections[cls] / total_unions[cls]
                valid_classes_iou.append(iou)
            ious.append(iou)
            
        mean_iou = np.nanmean(valid_classes_iou)
        
        print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_loss:.4f} - Val Loss: {val_loss:.4f} - mIoU: {mean_iou:.4f}')
        
        # Optional: Print IoU for specific interesting classes
        print(f"  > IoU Snow: {ious[8]:.4f} | IoU Water: {ious[9]:.4f} | IoU Artificial: {ious[2]:.4f}")
        
        scheduler.step()
        
    print("Entraînement terminé !")
    return model

def evaluate_kl_divergence(model, data_loader, device):
    """
    Calcule la KL Divergence moyenne par pixel sur un jeu de données.
    """
    print("Calcul de la KL Divergence...")
    model.eval()
    total_kl_div = 0.0
    total_pixels = 0

    with torch.no_grad():
        for images, masks in data_loader:
            images = images.to(device)
            masks = masks.to(device)

            predicted_logits = model(images)

            true_labels_one_hot = F.one_hot(masks, num_classes=LandCoverData.N_CLASSES).permute(0, 3, 1, 2).float()

            kl_div = kl_divergence(predicted_logits, true_labels_one_hot, reduction='sum')
            
            total_kl_div += kl_div.item()
            
            total_pixels += masks.nelement()

    avg_kl_div = total_kl_div / total_pixels if total_pixels > 0 else 0
    return avg_kl_div

def main():
    # Enable blocking launch for better debug on CUDA errors
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    DATA_FOLDER = Path('./dataset/').expanduser()
    
    train_images_paths = sorted(list(DATA_FOLDER.glob('train/images/*.tif')))
    train_masks_paths = sorted(list(DATA_FOLDER.glob('train/masks/*.tif')))
    test_images_paths = sorted(list(DATA_FOLDER.glob('test/images/*.tif')))

    print(f'Number of train images: {len(train_images_paths)}')
    print(f'Number of train masks: {len(train_masks_paths)}')
    print(f'Number of test images: {len(test_images_paths)}')

    if len(train_images_paths) > 0:
        idx = random.choice(range(len(train_images_paths)))
        visualize_sample(train_images_paths[idx], train_masks_paths[idx])

    print("Préparation des données...")
    
    # Separation manuelle des chemins pour train et val afin d'appliquer l'augmentation uniquement sur le train
    indices = list(range(len(train_images_paths)))
    random.shuffle(indices)
    
    train_split = int(0.8 * len(indices))
    train_indices = indices[:train_split]
    val_indices = indices[train_split:]
    
    train_img_paths_split = [train_images_paths[i] for i in train_indices]
    train_mask_paths_split = [train_masks_paths[i] for i in train_indices]
    
    val_img_paths_split = [train_images_paths[i] for i in val_indices]
    val_mask_paths_split = [train_masks_paths[i] for i in val_indices]
    
    # Activation de l'augmentation de données pour le jeu d'entraînement
    train_dataset = LandCoverDataset(train_img_paths_split, train_mask_paths_split, augment=True)
    val_dataset = LandCoverDataset(val_img_paths_split, val_mask_paths_split, augment=False)
    
    BATCH_SIZE = 8
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Données prêtes: {len(train_dataset)} images d'entraînement (avec augmentation), {len(val_dataset)} images de validation.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilisation du device: {device}")
    
    model = SimpleUNet(in_channels=LandCoverData.N_CHANNELS, out_channels=LandCoverData.N_CLASSES, base_filters=32)
    model = model.to(device)
    
    MODEL_PATH = 'simple_unet_model.pth'
    
    # Calcul des poids pour gerer le desequilibre des classes
    print("Calcul des poids des classes...")
    counts = LandCoverData.TRAIN_CLASS_COUNTS
    weights = np.zeros(len(counts))
    
    # On ignore les classes 0 et 1 (no_data, clouds)
    valid_classes = [c for c in range(len(counts)) if c not in LandCoverData.IGNORED_CLASSES_IDX]
    valid_counts = counts[valid_classes]
    total_valid_pixels = valid_counts.sum()
    n_valid_classes = len(valid_classes)
    
    for i in range(len(counts)):
        if i in LandCoverData.IGNORED_CLASSES_IDX:
            weights[i] = 0.0
        else:
            # Poids inversement proportionnel a la frequence (avec lissage racine carrée pour éviter l'explosion)
            # Formule: (N_total / (N_classes * Count_class)) ** 0.5
            weights[i] = (total_valid_pixels / (n_valid_classes * counts[i])) ** 0.5
            
    print(f"Poids des classes (lissés) : {weights}")
    class_weights = torch.FloatTensor(weights).to(device)
    
    if Path(MODEL_PATH).exists():
        print(f"Modèle trouvé à '{MODEL_PATH}'. Chargement du modèle existant...")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    else:
        print("Aucun modèle trouvé. Démarrage de l'entraînement...")
        
        # Utilisation des poids dans la Loss
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Reduced learning rate from 0.001 to 0.0003 for stability
        optimizer = optim.Adam(model.parameters(), lr=0.0003)
        
        model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device=device)
        
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Modèle sauvegardé sous '{MODEL_PATH}'")


    print("Visualisation sur le jeu de validation (avec vérité terrain) :")
    predict_and_visualize(model, val_dataset, device)

    if len(test_images_paths) > 0:
        print("Visualisation sur le jeu de test (sans vérité terrain) :")
        test_dataset = LandCoverDataset(test_images_paths)
        predict_and_visualize(model, test_dataset, device, num_samples=10)



if __name__ == "__main__":
    main()
