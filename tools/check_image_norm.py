#!/usr/bin/env python3
"""Small script to check whether an image is normalized to [0,1].

Usage:
    python tools/check_image_norm.py /path/to/image.png

Output: prints raw pixel range, range after /255, and example pixels. If the pipeline
applies sub/div normalization (e.g. via config `data_norm`), you can simulate it manually.
"""
import sys
import numpy as np
from PIL import Image


def main(path):
    img = Image.open(path).convert('RGB')
    arr = np.array(img)
    arr_f = arr.astype(np.float32)

    print(f'Path: {path}')
    print(f'shape: {arr.shape}, dtype: {arr.dtype}')
    print('RAW stats: min={:.6f}, max={:.6f}, mean={:.6f}'.format(arr_f.min(), arr_f.max(), arr_f.mean()))
    pct = np.percentile(arr_f, [0,1,5,25,50,75,95,99,100])
    print('RAW percentiles: ', ['{:.3f}'.format(x) for x in pct])

    arr01 = arr_f / 255.0
    print('\nAfter /255.0: min={:.6f}, max={:.6f}, mean={:.6f}'.format(arr01.min(), arr01.max(), arr01.mean()))
    print('Within [0,1]? ->', float(arr01.min()) >= 0.0 and float(arr01.max()) <= 1.0)

    # Demonstrate how pipeline sub/div (e.g. config `data_norm`) would affect values.
    # Common examples: sub=0, div=1 (no-op) or sub=0, div=255 or sub=[0.4488,...], div=[0.2257,...]
    checks = [
        ('identity (no-op)', lambda x: x),
        ('/255', lambda x: x/255.0),
    ]
    for name, fn in checks:
        a = fn(arr_f)
        print(f"\n{name}: min={{:.6f}}, max={{:.6f}}, mean={{:.6f}}".format(a.min(), a.max(), a.mean()))

    # Show several sample pixels
    h, w = arr.shape[0], arr.shape[1]
    samples = [(0,0), (h//2, w//2), (h-1, w-1)]
    print('\nSample pixels (raw -> /255):')
    for (y,x) in samples:
        raw = arr[y,x]
        scaled = arr01[y,x]
        print(f'  ({y},{x}): raw={raw.tolist()} -> /255={scaled.tolist()}')

    # Is the image integer-typed?
    print('\nIs integer type?:', np.issubdtype(arr.dtype, np.integer))

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python tools/check_image_norm.py /path/to/image.png')
        sys.exit(1)
    main(sys.argv[1])
