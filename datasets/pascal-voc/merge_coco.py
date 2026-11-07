
import json
import argparse
from collections import OrderedDict


def merge_coco(json1_path, json2_path, output_path):
    with open(json1_path, 'r', encoding='utf-8') as f:
        coco1 = json.load(f)
    with open(json2_path, 'r', encoding='utf-8') as f:
        coco2 = json.load(f)

    categories = OrderedDict()
    for c in coco1.get('categories', []):
        categories[c['name']] = c
    for c in coco2.get('categories', []):
        if c['name'] not in categories:
            categories[c['name']] = c

    name_to_new_id = {}
    new_categories = []
    for new_id, (name, cat) in enumerate(categories.items(), start=1):
        cat_copy = dict(cat)
        cat_copy['id'] = new_id
        new_categories.append(cat_copy)
        name_to_new_id[name] = new_id

    merged_images = []
    merged_annotations = []

    image_id_map = {}
    ann_id = 1
    image_id = 1

    def add_dataset(coco):
        nonlocal image_id, ann_id
        old_to_new_img = {}
        for img in coco.get('images', []):
            new_img = dict(img)
            old_to_new_img[img['id']] = image_id
            new_img['id'] = image_id
            merged_images.append(new_img)
            image_id += 1

        for ann in coco.get('annotations', []):
            new_ann = dict(ann)
            new_ann['id'] = ann_id
            new_ann['image_id'] = old_to_new_img[ann['image_id']]
            cat_name = None
            for c in coco.get('categories', []):
                if c['id'] == ann['category_id']:
                    cat_name = c['name']
                    break
            if cat_name is None:
                continue
            new_ann['category_id'] = name_to_new_id[cat_name]
            merged_annotations.append(new_ann)
            ann_id += 1

    add_dataset(coco1)
    add_dataset(coco2)

    merged = {
        'images': merged_images,
        'annotations': merged_annotations,
        'categories': new_categories,
    }

    for key in ['info', 'licenses']:
        if key in coco1:
            merged[key] = coco1[key]
        elif key in coco2:
            merged[key] = coco2[key]

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json1', required=True)
    parser.add_argument('--json2', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    merge_coco(args.json1, args.json2, args.output)