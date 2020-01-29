"""
takes a coco-format json and copies fields to match a sim10k2cs conversion with MUNIT
"""

import json
import os
import argparse
from copy import deepcopy
import tqdm


parser = argparse.ArgumentParser(description='Convert dataset')
parser.add_argument(
    '--inpath', help="data dir for cocootations to be converted", required=True)
parser.add_argument(
    '--outpath', help="output dir for json files", required=True)
parser.add_argument(
    '--imgdir', help="directory to parse for subdirs of MUNIT modes", required=True)

args = parser.parse_args()

# For each style vector MUNIT script generates a subdir
img_subdirs = [
    d for d in os.listdir(args.imgdir)
    if os.path.isdir(os.path.join(args.imgdir, d))
]

coco = json.load(open(args.inpath))
new_coco = {
    "images": [],
    "annotations": [],
    "categories": coco["categories"]
}

for img in tqdm.tqdm(coco["images"], desc="Converting image metadata"):
    for i, prefix in enumerate(img_subdirs):
        new_img = deepcopy(img)
        new_img["file_name"] = os.path.join(prefix, img["file_name"])
        new_img["id"] = img['id'] + i * len(coco["images"])
        # RESIZING FROM 2048x1024 TO 1024x512
        new_img["width"] = 928
        new_img["height"] = 512
        new_coco["images"].append(new_img)

for ann in tqdm.tqdm(coco["annotations"], desc="Converting annotations"):
    # del ann["segmentation"]
    for i, prefix in enumerate(img_subdirs):
        new_ann = deepcopy(ann)
        new_ann["id"] = f"{prefix}_{ann['id']}"
        new_ann["image_id"] = ann['image_id'] + i * len(coco["images"])
        # RESIZING FROM 1914x1052 TO 928x512
        for poly_idx in range(len(new_ann["segmentation"])):
            for coord_idx in range(len(new_ann["segmentation"][poly_idx])):
                if coord_idx % 2 == 0:
                    # X coordinate
                    new_ann["segmentation"][poly_idx][coord_idx] *= 928 / 1914
                else:
                    # Y coordinate
                    new_ann["segmentation"][poly_idx][coord_idx] *= 512 / 1052

                new_ann["segmentation"][poly_idx][coord_idx] = int(new_ann["segmentation"][poly_idx][coord_idx])

        x, y, w, h = ann["bbox"]
        # Horizontal resize ratio
        x *= 928 / 1914
        w *= 928 / 1914

        # Vertical resize ratio
        y *= 512 / 1052
        h *= 512 / 1052

        # DESIGN CHOICE
        # We don't know how area was computed in sim10k
        new_ann["area"] = w * h
        new_ann["bbox"] = list(map(int, (x, y, w, h)))
        new_coco["annotations"].append(new_ann)

print(f"Saving output to: {args.outpath}")
with open(args.outpath, "w") as f:
    json.dump(new_coco, f)

