import zipfile
import os
import pandas as pd
import re
import shutil

def read_annotations(dataset_type="train"):
    df = pd.read_csv(dataset_type+"-annotations-object-segmentation.csv")
    df["type"] = dataset_type
    return df

def extract_specific_files_old(zip_path, target_files, output_dir):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for filename in zip_ref.namelist():
            for target_file in target_files:
                if target_file in filename:
                    zip_ref.extract(filename, output_dir)

def get_target_files(annotation_df, group, class_name):
    target_files = list(annotation_df[(annotation_df["ImageID_first_character"] == group) &
                                   (annotation_df["class"] == class_name)]["MaskPath"])
    return target_files

def extract_specific_files(zip_path, target_files, output_dir):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for filename in zip_ref.namelist():
            for target_file in target_files:
                if target_file in filename:
                    
                    file_data = zip_ref.read(filename)

                    new_file_path = os.path.join(output_dir, filename)

                    with open(new_file_path, 'wb') as f:
                        f.write(file_data)

def extract_specific_files_for_group(annotation_df, group, class_name, zip_prefix):
    target_files = get_target_files(annotation_df, group, class_name)
    extract_specific_files(zip_path = "images/"+zip_prefix+group+".zip", target_files = target_files, output_dir="images/"+class_name.lower()+"/masks")

def extract_specific_files_for_class(annotation_df, class_name, groups, zip_prefix):
    [extract_specific_files_for_group(annotation_df, g, class_name, zip_prefix) for g in groups]

def extract_specific_files_for_classes(annotation_df, classes, groups, zip_prefix = "train-masks-"):
    [extract_specific_files_for_class(annotation_df, c, groups, zip_prefix) for c in classes]

def combine_directories(source_dirs, target_dir):
    
    for source_dir in source_dirs:
        
        for file_name in os.listdir(source_dir):
           
            if file_name.endswith('.jpg') or file_name.endswith('.png'):
                
                source = os.path.join(source_dir, file_name)
                target = os.path.join(target_dir, file_name)
                
                shutil.copy(source, target)

def get_dir_for_class(class_name, masks_or_images="images"):
    return "images/"+class_name.lower()+"/"+masks_or_images

def delete_images_without_masks(images_dir, classes):

    mask_dirs = [get_dir_for_class(c, masks_or_images="masks") for c in classes]
    images_to_keep = sum([[i.replace(".png", "")+".jpg" for i in os.listdir(m)] for m in mask_dirs], [])
    images_to_keep =  [re.match('[^_]*', i).group()+".jpg" for i in images_to_keep]
    images_to_remove = [f for f in os.listdir(images_dir) if f not in images_to_keep]

    for file in images_to_remove:
        file_path = os.path.join(images_dir, file)
        os.remove(file_path)

    print(f"Deleted {len(images_to_remove)} images")