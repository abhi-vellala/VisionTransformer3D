import pandas as pd
import os

age_data = pd.read_excel('./Data/Einschluss Myocardial Aging.xlsx')

img_path = './Data/Images'
# seg_path = './Data/Segmentations'

df = pd.DataFrame(columns=['ID', 'image'])

df['ID'] = [*range(1, len(os.listdir(img_path))+1)]
# print([*range(1, len(os.listdir('./Data/Images'))+1)])
for fi in os.listdir(img_path):
    num = int(fi.split('_')[0])
    df.loc[df['ID'] == num, 'image'] = img_path + f'/{fi}'  

# for fi in os.listdir(seg_path):
#     num = int(fi.split('_')[0])
#     df.loc[df['ID'] == num, 'seg'] = seg_path + f'/{fi}'  

df = pd.merge(df, age_data[['ID', 'Age']], on='ID')

df['target'] = df['Age'].apply(lambda x: 0 if x <= 60 else 1)

print(df)
df.to_excel('./Data/image_data.xlsx')