import numpy as np
import matplotlib.pyplot as plt
from tifffile import TiffFile
from .data import LandCoverData, get_array_from_path

def show_image(image, display_min=50, display_max=400, ax=None):
    """Show an image.
    Args:
        image (numpy.array[uint16]): the image. If the image is 16-bit, apply bytescaling to convert to 8-bit
    """
    if image.dtype == np.uint16:
        iscale = display_max - display_min
        scale = 255 / iscale
        byte_im = (image) * scale
        byte_im = (byte_im.clip(0, 255) + 0.5).astype(np.uint8)
        image = byte_im
    # show image
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    ax.axis("off")
    im = ax.imshow(image)
    return im

def show_mask(mask, classes_colorpalette, classes=None, add_legend=True, ax=None):
    """Show a a semantic segmentation mask.
    Args:
       mask (numpy.array[uint8]): the mask in 8-bit
       classes_colorpalette (dict[int, tuple]): dict mapping class index to an RGB color in [0, 1]
       classes (list[str], optional): list of class labels
       add_legend
    """
    show_mask = np.empty((*mask.shape, 3))
    for c, color in classes_colorpalette.items():
        show_mask[mask == c, :] = color
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    ax.axis("off")
    im = ax.imshow(show_mask)
    if add_legend:
        # show legend mapping pixel colors to class names
        import matplotlib.patches as mpatches
        handles = []
        for c, color in classes_colorpalette.items():
            handles.append(mpatches.Patch(color=color, label=classes[c]))
        ax.legend(handles=handles)
    return im

def print_stats(arr, name):
    print(f'Image shape: {arr.shape}, dtype: {arr.dtype}')
    print(f'Image name: {name}')
    for c in range(arr.shape[2]):
        print(f'  Channel {c}: min={arr[:,:,c].min()}, max={arr[:,:,c].max()}, mean={arr[:,:,c].mean():.2f}, std={arr[:,:,c].std():.2f}')

def visualize_sample(image_path, mask_path):
    assert image_path.name == mask_path.name
    
    img_arr = get_array_from_path(image_path)
    mask_arr = get_array_from_path(mask_path)
        
    print_stats(img_arr, image_path.name)
    
    fig, axs = plt.subplots(1, 2, figsize=(15, 10))
    
    # Plot Image
    im = show_image(img_arr, display_min=0, display_max=2200, ax=axs[0])
    axs[0].set_title(f'Image: {image_path.name}')
    fig.colorbar(im, ax=axs[0], fraction=0.046, pad=0.04)
    
    # Plot Mask
    classes_colorpalette = {c: color/255. for (c, color) in LandCoverData.CLASSES_COLORPALETTE.items()}
    im = show_mask(mask_arr,
            classes_colorpalette = classes_colorpalette,
            classes=LandCoverData.CLASSES,
            add_legend=True,
            ax=axs[1]
    )
    axs[1].set_title('Mask')
    fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)
    plt.show()
