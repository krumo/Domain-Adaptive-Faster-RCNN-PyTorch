import argparse
import h5py
import json
import os
import scipy.misc
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='Convert dataset')
    parser.add_argument(
        '--dataset', help="kitti", default=None, type=str)
    parser.add_argument(
        '--outdir', help="output dir for json files", default=None, type=str)
    parser.add_argument(
        '--datadir', help="data dir for annotations to be converted",
        default=None, type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

def xyxy_to_xywh(xyxy_box):
    xmin, ymin, xmax, ymax = xyxy_box
    TO_REMOVE = 1
    xywh_box = (xmin, ymin, xmax - xmin + TO_REMOVE, ymax - ymin + TO_REMOVE)
    return xywh_box

def convert_kitti_car_only(
        data_dir, out_dir):
    """Convert from kitti format to COCO format"""
    sets = [
        'training',
        # 'testing',
    ]
    img_dir = 'image_2'
    ann_dir = 'label_2'
    json_name = 'caronly_%s.json'
    img_id = 0
    ann_id = 0
    cat_id = 1
    category_dict = {}

    category_instancesonly = [
        'Car',
    ]

    for ind, cat in enumerate(category_instancesonly):
        category_dict[cat] = ind + 1

    for data_set in sets:
        print('Starting %s' % data_set)
        ann_dict = {}
        images = []
        annotations = []
        image_dir = os.path.join(data_dir, data_set, img_dir)
        annotation_dir = os.path.join(data_dir, data_set, ann_dir)

        for filename in os.listdir(image_dir):
            if filename.endswith('.png'):
                if len(images) % 50 == 0:
                    print("Processed %s images, %s annotations" % (
                        len(images), len(annotations)))
                image = {}
                image['id'] = img_id
                img_id += 1

                from PIL import Image
                img = Image.open(os.path.join(image_dir, filename))
                w, h = img.size

                image['width'] = w
                image['height'] = h
                image['file_name'] = filename
                image['seg_file_name'] = filename.replace('.png', '.txt')
                images.append(image)

                ann_file = os.path.join(annotation_dir, image['seg_file_name'])

                if os.path.isfile(ann_file):
                    with open(ann_file, 'r') as handle:
                        content = handle.readlines()
                    for line in content:
                        line = line.strip()
                        l = line.split(' ')
                        if l[0] not in category_instancesonly:
                            continue
                        x_min, y_min, x_max, y_max = float(l[4]), float(l[5]), float(l[6]), float(l[7])

                        ann = {}
                        ann['id'] = ann_id
                        ann_id += 1
                        ann['image_id'] = image['id']
                        ann['segmentation'] = []

                        ann['category_id'] = category_dict[l[0]]
                        ann['iscrowd'] = 0
                            
                        xyxy_box = (x_min, y_min, x_max, y_max)
                        xywh_box = xyxy_to_xywh(xyxy_box)
                        ann['bbox'] = xywh_box
                        ann['area'] = xywh_box[2]*xywh_box[3]

                        annotations.append(ann)

        ann_dict['images'] = images
        categories = [{"id": category_dict[name], "name": name} for name in
                      category_dict]
        ann_dict['categories'] = categories
        ann_dict['annotations'] = annotations
        print(categories)
        print("Num categories: %s" % len(categories))
        print("Num images: %s" % len(images))
        print("Num annotations: %s" % len(annotations))
        with open(os.path.join(out_dir, json_name % data_set), 'w') as outfile:
            outfile.write(json.dumps(ann_dict))


if __name__ == '__main__':
    args = parse_args()
    if args.dataset == "kitti":
        convert_kitti_car_only(args.datadir, args.outdir)
    else:
        print("Dataset not supported: %s" % args.dataset)
