"""
Simple script to copy up to N representative images from data_split/test into
`static/slideshow/` for fast frontend slideshow serving.

Run:
    python scripts/populate_slideshow.py --count 20

This will resize images to a max width of 1200px and save as JPEG for performance.
"""
import os
import random
import argparse
import shutil
from PIL import Image

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_TEST = os.path.join(BASE, 'data_split', 'test')
OUT_DIR = os.path.join(BASE, 'static', 'slideshow')

VALID_EXT = ('.jpg', '.jpeg', '.png')


def main(count=20):
    if not os.path.isdir(DATA_TEST):
        print('data_split/test not found at', DATA_TEST)
        return
    os.makedirs(OUT_DIR, exist_ok=True)

    candidates = []
    for cls in os.listdir(DATA_TEST):
        cls_dir = os.path.join(DATA_TEST, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith(VALID_EXT):
                candidates.append((cls, os.path.join(cls_dir, fname)))

    if not candidates:
        print('No images found in data_split/test')
        return

    random.shuffle(candidates)
    selected = candidates[:count]

    for idx, (cls, path) in enumerate(selected):
        try:
            im = Image.open(path).convert('RGB')
            # Resize to max dimension width=1200 for good balance
            maxw = 1200
            if im.width > maxw:
                h = int(im.height * (maxw / im.width))
                im = im.resize((maxw, h), Image.LANCZOS)

            out_name = f'{idx:03d}_{cls}_{os.path.basename(path)}'
            out_path = os.path.join(OUT_DIR, out_name)
            im.save(out_path, format='JPEG', quality=85)
            print('Saved', out_path)
        except Exception as e:
            print('Error processing', path, e)

    print('Done. Files are in', OUT_DIR)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--count', type=int, default=20)
    args = parser.parse_args()
    main(count=args.count)
