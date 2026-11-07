
import os
import json
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET


def voc_to_coco(voc_dir, output_json, classes=None):
    voc_dir = Path(voc_dir)
    voc_root = voc_dir.name
    ann_dir = voc_dir / 'Annotations'
    img_dir = voc_dir / 'JPEGImages'
    splits_dir = voc_dir / 'ImageSets' / 'Main'

    if classes is None:
        classes = []
        for xml_file in sorted(ann_dir.glob('*.xml')):
            root = ET.parse(xml_file).getroot()
            for obj in root.findall('object'):
                name = obj.findtext('name')
                if name and name not in classes:
                    classes.append(name)
        classes = sorted(classes)

    cat2id = {name: i + 1 for i, name in enumerate(classes)}
    coco = {'images': [], 'annotations': [], 'categories': []}
    for name, cid in cat2id.items():
        coco['categories'].append({'id': cid, 'name': name, 'supercategory': 'none'})

    ann_id = 1
    img_id = 1
    for xml_file in sorted(ann_dir.glob('*.xml')):
        root = ET.parse(xml_file).getroot()
        filename = os.path.join(img_dir, root.findtext('filename'))
        size = root.find('size')
        width = int(size.findtext('width'))
        height = int(size.findtext('height'))
        image_entry = {
            'id': img_id,
            'file_name': filename if filename else xml_file.with_suffix('.jpg').name,
            'width': width,
            'height': height,
        }
        coco['images'].append(image_entry)

        for obj in root.findall('object'):
            name = obj.findtext('name')
            if name not in cat2id:
                continue
            bnd = obj.find('bndbox')
            xmin = float(bnd.findtext('xmin'))
            ymin = float(bnd.findtext('ymin'))
            xmax = float(bnd.findtext('xmax'))
            ymax = float(bnd.findtext('ymax'))
            w = xmax - xmin
            h = ymax - ymin
            if w <= 0 or h <= 0:
                continue
            coco['annotations'].append({
                'id': ann_id,
                'image_id': img_id,
                'category_id': cat2id[name],
                'bbox': [xmin, ymin, w, h],
                'area': w * h,
                'iscrowd': 0
            })
            ann_id += 1
        img_id += 1

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)

    return coco


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--voc_dir', required=True)
    parser.add_argument('--output_json', required=True)
    parser.add_argument('--classes', nargs='*', default=None)
    args = parser.parse_args()
    voc_to_coco(args.voc_dir, args.output_json, args.classes)