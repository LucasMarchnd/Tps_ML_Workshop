from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from ml_utils.data import load_dataset, get_array_from_path, LandCoverData, LandCoverDataset
from ml_utils.viz import visualize_sample, show_image, show_mask
from ml_utils.model import SimpleUNet
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def predict_and_visualize(model, dataset, device, num_samples=3):
    """
    Fait des prédictions sur quelques images et les affiche avec un slider de transparence.
    """
    print(f"Visualisation de {num_samples} prédictions...")
    model.eval() # Mode évaluation
    
    # Handle dataset size smaller than num_samples
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
        
        # Préparation pour le modèle (ajout dimension batch)
        input_tensor = image.unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            # output est (1, N_CLASSES, H, W)
            pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
            
        # Préparation pour l'affichage
        # Image: (C, H, W) -> (H, W, C) et conversion numpy uint16 pour show_image
        img_display = image.permute(1, 2, 0).numpy().astype(np.uint16)
        
        # Prepare color mask
        classes_colorpalette = {c: color/255. for (c, color) in LandCoverData.CLASSES_COLORPALETTE.items()}
        color_mask = np.zeros((*pred_mask.shape, 3))
        for c, color in classes_colorpalette.items():
            color_mask[pred_mask == c] = color

        # Setup plot
        fig, ax = plt.subplots(figsize=(10, 8))
        plt.subplots_adjust(bottom=0.25)
        
        # Show original image
        show_image(img_display, display_min=0, display_max=2200, ax=ax)
        ax.set_title(f"Prédiction (Sample {idx})")
        
        # Show prediction overlay
        initial_alpha = 0.5
        mask_im = ax.imshow(color_mask, alpha=initial_alpha)
        
        # Slider
        ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        slider = Slider(ax_slider, 'Opacité', 0.0, 1.0, valinit=initial_alpha)
        
        def update(val):
            mask_im.set_alpha(slider.val)
            fig.canvas.draw_idle()
            
        slider.on_changed(update)
        
        # Add legend
        import matplotlib.patches as mpatches
        handles = []
        for c, color in classes_colorpalette.items():
            handles.append(mpatches.Patch(color=color, label=LandCoverData.CLASSES[c]))
        ax.legend(handles=handles, loc='upper right', bbox_to_anchor=(1.3, 1))
        
        print(f"Affichage de l'image {idx}. Fermez la fenêtre pour passer à la suivante.")
        plt.show()

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20, device='cpu'):
    """
    Fonction d'entraînement du modèle.
    
    Args:
        model: Le modèle PyTorch à entraîner
        train_loader: DataLoader pour les données d'entraînement
        val_loader: DataLoader pour les données de validation
        criterion: Fonction de perte (Loss function)
        optimizer: Optimiseur (ex: Adam)
        num_epochs: Nombre d'époques
        device: 'cuda' ou 'cpu'
    """
    print("Début de l'entraînement...")
    
    for epoch in range(num_epochs):
        model.train() # Met le modèle en mode entraînement
        running_loss = 0.0
        
        # Boucle sur les batches d'entraînement
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            # 1. Remise à zéro des gradients
            optimizer.zero_grad()
            
            # 2. Passage avant (Forward pass)
            outputs = model(images)
            
            # 3. Calcul de la perte
            loss = criterion(outputs, masks)
            
            # 4. Rétropropagation (Backward pass)
            loss.backward()
            
            # 5. Mise à jour des poids
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # Validation
        model.eval() # Met le modèle en mode évaluation
        val_loss = 0.0
        with torch.no_grad(): # Pas de calcul de gradient pour la validation
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)
                
        val_loss = val_loss / len(val_loader.dataset)
        
        print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_loss:.4f} - Val Loss: {val_loss:.4f}')
        
    print("Entraînement terminé !")
    return model

def main():
    # Define paths
    DATA_FOLDER = Path('./dataset/').expanduser()
    
    train_images_paths = sorted(list(DATA_FOLDER.glob('train/images/*.tif')))
    train_masks_paths = sorted(list(DATA_FOLDER.glob('train/masks/*.tif')))
    test_images_paths = sorted(list(DATA_FOLDER.glob('test/images/*.tif')))

    print(f'Number of train images: {len(train_images_paths)}')
    print(f'Number of train masks: {len(train_masks_paths)}')
    print(f'Number of test images: {len(test_images_paths)}')

    # Visualize a random sample to check data integrity
    if len(train_images_paths) > 0:
        idx = random.choice(range(len(train_images_paths)))
        visualize_sample(train_images_paths[idx], train_masks_paths[idx])

    # Load data for Machine Learning
    # Utilisation de notre Dataset personnalisé pour charger les données efficacement
    print("Préparation des données...")
    
    # On utilise un sous-ensemble pour l'exemple (limit=100) pour que ça tourne vite
    # Pour tout le dataset, enlevez le slicing [:100]
    subset_size = 1000
    full_dataset = LandCoverDataset(train_images_paths[:subset_size], train_masks_paths[:subset_size])
    
    # Séparation Train / Validation (80% / 20%)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Création des DataLoaders
    # batch_size: nombre d'images traitées en même temps
    # shuffle=True: mélange les données à chaque époque (important pour l'entraînement)
    BATCH_SIZE = 8
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Données prêtes: {len(train_dataset)} images d'entraînement, {len(val_dataset)} images de validation.")

    # Initialisation du modèle
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilisation du device: {device}")
    
    model = SimpleUNet(in_channels=LandCoverData.N_CHANNELS, out_channels=LandCoverData.N_CLASSES)
    model = model.to(device)
    
    MODEL_PATH = 'simple_unet_model.pth'
    
    if Path(MODEL_PATH).exists():
        print(f"Modèle trouvé à '{MODEL_PATH}'. Chargement du modèle existant...")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    else:
        print("Aucun modèle trouvé. Démarrage de l'entraînement...")
        
        # Définition de la Loss et de l'Optimizer
        # CrossEntropyLoss est standard pour la segmentation multi-classes
        criterion = nn.CrossEntropyLoss()
        
        # Adam est un optimiseur très performant et standard
        # lr (learning rate) contrôle la vitesse d'apprentissage
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Lancement de l'entraînement
        model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=80, device=device)
        
        # Sauvegarde du modèle
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Modèle sauvegardé sous '{MODEL_PATH}'")

    # Visualisation des résultats
    print("Visualisation sur le jeu de validation (avec vérité terrain) :")
    predict_and_visualize(model, val_dataset, device)

    # Visualisation sur le jeu de test (sans vérité terrain)
    if len(test_images_paths) > 0:
        print("Visualisation sur le jeu de test (sans vérité terrain) :")
        test_dataset = LandCoverDataset(test_images_paths)
        predict_and_visualize(model, test_dataset, device, num_samples=10)



if __name__ == "__main__":
    main()
