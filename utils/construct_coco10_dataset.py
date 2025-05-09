"""Code is adapted from https://github.com/google-research-datasets/bam

Copyright 2019 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""

import os
import numpy as np
from PIL import Image
import tensorflow as tf
from pycocotools.coco import COCO
import skimage.io as io
import torchvision.transforms as transforms
import hydra
from omegaconf import DictConfig

NUM_IMAGES_PER_CLASS = 1100
TRAIN_VAL_RATIO = 10 / 11


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def extract_coco_objects(config: DictConfig):
    """
    Creates COCO-10, a subset of the COCO dataset. We select 10 classes with 1000 training and 100 test images each.
    """
    coco_dir = os.path.join(config.data.data_path, "coco")
    data_type = "train2017"
    ann_file = "{}/annotations/instances_{}.json".format(coco_dir, data_type)
    coco = COCO(ann_file)

    # Step 0: Find classes that maximize the number of unique images
    # Get all category IDs
    cat_ids = coco.getCatIds()

    # Load all categories
    categories = coco.loadCats(cat_ids)

    # Count the number of images for each category
    category_image_counts = []
    for category in categories:
        cat_id = category["id"]
        img_ids = coco.getImgIds(catIds=[cat_id])
        category_image_counts.append((category["name"], len(img_ids)))

    # Sort the categories by the number of images in descending order
    category_image_counts.sort(key=lambda x: x[1], reverse=True)

    # Keep the top 30 categories
    top_categories = category_image_counts[:30]

    # Print the top 30 categories and the number of images for each category
    for category, count in top_categories:
        print(f"Name: {category}, Number of images: {count}")

    # Find the 20 categories with most unique images in greedy fashion.
    # We compute unique for all N=30 categories and iteratively remove the category with the least unique images.
    # We repeat this process until we have 20 categories.
    category_names = [category[0] for category in top_categories]
    for i in range(20):

        # Goal: Keep only unique images for each object class
        # Step 1: Create a dictionary to store sets of image IDs for each object
        obj_img_ids = {obj_name: set() for obj_name in category_names}

        # Step 2: Populate the sets with image IDs
        for obj_name in category_names:
            cat_ids = coco.getCatIds(catNms=[obj_name])
            img_ids = coco.getImgIds(catIds=cat_ids)
            obj_img_ids[obj_name].update(img_ids)

        # Step 3: Compare the sets to keep only unique IDs
        unique_img_ids = {}
        for obj_name, img_ids in obj_img_ids.items():
            unique_ids = img_ids.copy()
            for other_obj_name, other_img_ids in obj_img_ids.items():
                if obj_name != other_obj_name:
                    unique_ids -= other_img_ids
            unique_img_ids[obj_name] = unique_ids

        # Step 4: Count the number of unique images for each object class
        category_image_counts = []
        for obj_name, img_ids in unique_img_ids.items():
            category_image_counts.append((obj_name, len(img_ids)))
        category_image_counts.sort(key=lambda x: x[1], reverse=True)

        # Step 5: Remove the object class with the least unique images
        print(
            f"Removing {category_image_counts[-1][0]} with only {category_image_counts[-1][1]} unique images"
        )
        category_names.remove(category_image_counts[-1][0])

    # Post-hoc manual inspection found that dining table oftentimes focuses on the food itself,
    # posing a strong spurious correlation for the segmentation task, thus, we replace it with vase
    # Same argument with bowl that is always filled with food. Adding bed instead
    category_names.remove("dining table")
    category_names.remove("bowl")
    category_names.append("vase")
    category_names.append("bed")

    print("Selected categories:", category_names)

    # Storing images from the selected categories
    # Step 1: Create a dictionary to store sets of image IDs for each object
    obj_img_ids = {obj_name: set() for obj_name in category_names}

    # Step 2: Populate the sets with image IDs
    for obj_name in category_names:
        cat_ids = coco.getCatIds(catNms=[obj_name])
        img_ids = coco.getImgIds(catIds=cat_ids)
        obj_img_ids[obj_name].update(img_ids)

    # Step 3: Compare the sets to keep only unique IDs
    unique_img_ids = {}
    for obj_name, img_ids in obj_img_ids.items():
        unique_ids = img_ids.copy()
        for other_obj_name, other_img_ids in obj_img_ids.items():
            if obj_name != other_obj_name:
                unique_ids -= other_img_ids
        unique_img_ids[obj_name] = unique_ids

    # Step 4: Extract images and mask for each object class
    for obj_name in category_names:
        output_dir_train = os.path.join(coco_dir, "train")
        output_dir_val = os.path.join(coco_dir, "val")
        cat_ids = coco.getCatIds(catNms=[obj_name])
        imgs = coco.loadImgs(unique_img_ids[obj_name])
        dst_dir_train = os.path.join(output_dir_train, obj_name)
        dst_dir_val = os.path.join(output_dir_val, obj_name)
        dst_dir_mask_train = os.path.join(coco_dir, "mask", "train", obj_name)
        dst_dir_mask_val = os.path.join(coco_dir, "mask", "val", obj_name)
        if not tf.io.gfile.exists(dst_dir_train):
            tf.io.gfile.makedirs(dst_dir_train)
            tf.io.gfile.makedirs(dst_dir_val)
            tf.io.gfile.makedirs(dst_dir_mask_train)
            tf.io.gfile.makedirs(dst_dir_mask_val)

        num_imgs = 0
        for img in imgs:
            try:
                I = io.imread(
                    tf.io.gfile.GFile(
                        "%s/images/%s/%s" % (coco_dir, data_type, img["file_name"]),
                        "rb",
                    )
                )
                org_h, org_w = I.shape[0], I.shape[1]
                ann_ids = coco.getAnnIds(imgIds=img["id"], catIds=cat_ids, iscrowd=None)
                anns = coco.loadAnns(ann_ids)

                # If multiple annotations, merge them
                mask2d = np.zeros((org_h, org_w), dtype=np.bool_)
                for ann in anns:
                    mask2d_ann = coco.annToMask(ann)
                    mask2d = np.maximum(mask2d, mask2d_ann.astype(np.bool_))

                if img["file_name"][-3:] == "jpg":
                    fname = img["file_name"][:-3] + "png"
                else:
                    fname = img["file_name"]

                save_img = Image.fromarray(I)
                if num_imgs < TRAIN_VAL_RATIO * NUM_IMAGES_PER_CLASS:
                    save_img.save(
                        tf.io.gfile.GFile(os.path.join(dst_dir_train, fname), "w"),
                        format="png",
                    )
                    # Save mask as npy
                    np.save(
                        os.path.join(dst_dir_mask_train, fname[:-4] + ".npy"), mask2d
                    )

                else:
                    # Crop to 224x224. If mask too small, skip image
                    resize = transforms.Resize(256)
                    crop = transforms.CenterCrop(224)
                    cropped_save_img = crop(resize(save_img))
                    cropped_mask = np.array(crop(resize(Image.fromarray(mask2d))))

                    # Skip image if cropping removed too much of object or background
                    if (cropped_mask > 0).mean() < 0.05 or (
                        cropped_mask > 0
                    ).mean() > 0.95:
                        print("Skipped image")
                        continue

                    # Visualize image and mask
                    # import matplotlib.pyplot as plt
                    # plt.imshow(cropped_save_img)
                    # plt.imshow(cropped_mask, alpha=0.5)
                    else:
                        cropped_save_img.save(
                            tf.io.gfile.GFile(os.path.join(dst_dir_val, fname), "w"),
                            format="png",
                        )
                        # Save mask as npy
                        np.save(
                            os.path.join(dst_dir_mask_val, fname[:-4] + ".npy"),
                            cropped_mask,
                        )

                num_imgs += 1
                if num_imgs == NUM_IMAGES_PER_CLASS:
                    break
            except:
                continue


if __name__ == "__main__":
    extract_coco_objects()
