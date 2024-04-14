import random
import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import os
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

## Visualisation

def imshow_sample(img, ax):
    img = img.numpy().transpose((1,2,0))
    ax.imshow(img)

def display_sampled_image(dataset, img_nr):
    image, mask = dataset[img_nr]
    _, axs = plt.subplots(1, 2, figsize = (10, 5))

    imshow_sample(image, ax = axs[0])
    axs[1].imshow(mask.numpy())

def display_n_sampled_images(dataset, n):
    rand_imgs = [random.randint(0, len(dataset)) for _ in range(n)]
    [display_sampled_image(dataset, r) for r in rand_imgs]
    return rand_imgs


def visualize_mask(mask, num_classes=4, class_mapping=None, original=None):

    colors = plt.cm.get_cmap('rainbow', num_classes)
    cmap = ListedColormap([colors(i) for i in range(num_classes)])

    mask = mask.cpu().numpy().squeeze()
    fig, axs = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)

    if original is not None:
        axs[0].imshow(original)
        axs[0].title.set_text('Original image')
        axs[0].axis('off')

    im = axs[1].imshow(mask, cmap=cmap, vmin=0, vmax=num_classes-1)
    axs[1].title.set_text('Predicted mask')
    axs[1].axis('off')

    patches = []
    for i in range(num_classes):
        color = cmap(i)
        if class_mapping:
            label = class_mapping[i] if i in class_mapping else ""
        else:
            label = f"Class {i}"
        patches.append(mpatches.Patch(color=color, label=label))

    fig.legend(handles=patches, bbox_to_anchor=(0.5, -0.05), loc='upper center')
    plt.show()


def visualize_mask_on_ax(mask, ax, num_classes = 4):
    colors = plt.cm.get_cmap('rainbow', num_classes)
    cmap = ListedColormap([colors(i) for i in range(num_classes)])
    
    ax.imshow(mask.cpu().numpy().squeeze(), cmap=cmap, vmin=0, vmax=num_classes-1)

def show_test_prediction(nr, test_dataset, outputs_mask):
    t_inputs, t_labels = test_dataset[nr]
    
    _, axs = plt.subplots(1,3)

    img = t_inputs.numpy().transpose((1,2,0))
    axs[0].imshow(img)

    img = t_labels.numpy()
    axs[1].imshow(img)

    mask = outputs_mask[nr].cpu().numpy().transpose((1,2,0))
    axs[2].imshow(mask)

    visualize_mask_on_ax(outputs_mask[nr], axs[2], num_classes = 4)

def show_sampled_test_predictions(test_dataset, outputs_mask, number):
    random_nrs = [random.randint(0, len(test_dataset)) for _ in range(number)]
    [show_test_prediction(n, test_dataset, outputs_mask) for n in random_nrs]
    return random_nrs

## Dataset creation

def apply_same_transformation_to_pair(image, mask, transform):
    seed = np.random.randint(2024) 
    random.seed(seed)
    torch.manual_seed(seed)
    image = transform(image)
    random.seed(seed)
    torch.manual_seed(seed)
    mask = transform(mask)
    return image, mask

def _get_image_ids(dataset_type, class_annotations):
    return list(class_annotations[class_annotations["type"] == dataset_type]["ImageID"])

def _add_suffix(id_list, suffix):
    return [l+suffix for l in id_list]

def _get_image_ids_according_to_type(dir, dataset_type, suffix, class_annotations):
    ids = sorted([i.replace(suffix, "") for i in os.listdir(dir) if i in _add_suffix(_get_image_ids(dataset_type = dataset_type, class_annotations=class_annotations), suffix)])
    return ids

def _get_dir_for_class(class_name, type):
    return os.path.join(f"images\{class_name.lower()}\{type}")

def apply_same_transformation_to_pair(image, mask, transform):
    seed = np.random.randint(2024) 
    random.seed(seed)
    torch.manual_seed(seed)
    image = transform(image)
    random.seed(seed)
    torch.manual_seed(seed)
    mask = transform(mask)
    return image, mask

def get_dir_for_class(class_name, masks_or_images="images"):
    return "images/"+class_name.lower()+"/"+masks_or_images

def get_mask_paths_for_image_id(image_id, downloaded_annotations):
    annotations_for_image = downloaded_annotations[downloaded_annotations["ImageID"] == image_id].reset_index(drop=True)
    annotations_for_image["full_mask_path"] = [get_dir_for_class(c, masks_or_images="masks/") for c in annotations_for_image["class"]]+annotations_for_image["MaskPath"]
    return annotations_for_image

def get_mask_dict_for_image_id(image_id, downloaded_annotations):
    annotations_for_image = get_mask_paths_for_image_id(image_id, downloaded_annotations)

    car_masks = list(annotations_for_image[annotations_for_image["class"] == "Car"]["full_mask_path"])
    skyscraper_masks = list(annotations_for_image[annotations_for_image["class"] == "Skyscraper"]["full_mask_path"])
    person_masks = list(annotations_for_image[annotations_for_image["class"] == "Person"]["full_mask_path"])

    mask_dict = {
        "Car":car_masks,
        "Skyscraper":skyscraper_masks,
        "Person":person_masks
    }
    return mask_dict

def combine_masks_for_a_class(mask_files, class_idx, target_size):
    if not mask_files:
        return None
    
    class_mask = None
    for mask_file in mask_files:

        mask = Image.open(mask_file).resize(target_size).convert('1')
        mask = np.array(mask, dtype=np.uint8)
        
        if class_mask is None:
            class_mask = mask
        else:
            class_mask = np.logical_or(class_mask, mask)
    

    class_mask = class_mask * class_idx

    return class_mask

def combine_all_masks(mask_dict, class_to_integer, target_size):
    combined_mask = np.zeros(target_size, dtype=np.uint)

    for class_name, mask_files in mask_dict.items():
        class_idx = class_to_integer[class_name]
        class_mask = combine_masks_for_a_class(mask_files, class_idx, target_size)

        if class_mask is not None:
            combined_mask = np.maximum(combined_mask, class_mask)

    return combined_mask

class CustomImageDataset(Dataset):

    to_tensor = transforms.ToTensor()

    def __init__(self, image_dir, class_annotations, class_mapping, transform=None, dataset_type="train", target_size=(64,64)):
        self.transform = transform
        self.target_size = target_size
        self.image_ids =  _get_image_ids_according_to_type(dir = image_dir, dataset_type=dataset_type, suffix=".jpg", class_annotations=class_annotations)
        self.images = [image_dir+"/"+i+".jpg" for i in self.image_ids]
        self.mask_dicts = [get_mask_dict_for_image_id(image_id, class_annotations) for image_id in self.image_ids]
        self.class_mapping = class_mapping

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).resize(self.target_size).convert("RGB")
        mask = combine_all_masks(self.mask_dicts[idx], self.class_mapping, self.target_size)
        mask = Image.fromarray(mask.astype(np.uint8)).resize(self.target_size)
        mask = torch.from_numpy(np.array(mask).astype(np.int64)).long()
        image = self.to_tensor(image)
        if self.transform:
            image, mask = apply_same_transformation_to_pair(image, mask, transform=self.transform)
        return image, mask