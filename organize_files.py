import os

data_path = "C:\\Users\\Admin\\Desktop\\PhD\\Projects\\VisionTransformer\\Data"

seg_counter = 0
img_counter = 0
for file in os.listdir(os.path.join(data_path, 'PCCT Cardiac')):
    if 'seg' in file:
        os.rename(os.path.join(data_path, 'PCCT Cardiac', file), os.path.join(data_path, 'Segmentations', file))
        print(f'{file} Done')
        seg_counter += 1
    if 'img' in file:
        os.rename(os.path.join(data_path, 'PCCT Cardiac', file), os.path.join(data_path, 'Images', file))
        print(f'{file} Done')
        img_counter += 1

print('Total Files moved to Segmentation: {}'.format(seg_counter))
print('Total Files moved to Segmentation: {}'.format(img_counter))