import os

root = '/home/hypevr/data/projects/data/combined_human/train/'
image_dir = root + 'image/'
mask_dir = root + 'mask/'
path, dirs, files = next(os.walk(image_dir))
path_m, dirs_m, files_m = next(os.walk(image_dir))

for f in files:
    if f not in files_m:
        print(f)

for f in files_m:
    if f not in files:
        print(f)

