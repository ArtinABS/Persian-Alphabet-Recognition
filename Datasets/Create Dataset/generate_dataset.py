import os
import shutil
from fontTools.ttLib import TTFont
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pandas as pd



FONT_FILE_LIST = [f"Datasets/Create Dataset/Fonts/{f}" for f in os.listdir("Datasets/Create Dataset/Fonts")]
FONT_SIZE = 32
FONT_IMAGE_SIZE = (64, 64)
FONT_POSITION = (16, 8)


PERSIAN_CHARACTERS = list("ابپتثجچحخدزرزژسشصضطظعغفقکگلمنوهی")
LABELS = {0: 'Alef', 1: 'Be', 2: 'Pe', 3: 'Te', 4: 'Se', 5: 'Jim', 6: 'Che', 7: 'H', 
          8: 'Khe', 9: 'Dal', 10: 'Zal', 11: 'Re', 12: 'Ze', 13: 'Zhe', 14: 'Sin', 
          15: 'Shin', 16: 'Sad', 17: 'Zad', 18: 'Ta', 19: 'Za', 20: 'Ayin', 21: 'Ghayin', 
          22: 'Fe', 23: 'Ghaf', 24: 'Kaf', 25: 'Gaf', 26: 'Lam', 27: 'Mim', 28: 'Noon', 
          29: 'Vav', 30: 'He', 31: 'Ye'}

data_images = []
data_labels = []
data_fonts = []
dataset = []


if os.path.exists("Datasets/DS-3"):
    shutil.rmtree("Datasets/DS-3", ignore_errors=True)
os.mkdir("Datasets/DS-3")


for i, char in enumerate(PERSIAN_CHARACTERS):
    label = LABELS[i]
    label_folder = os.path.join("Datasets/DS-3", label)

    print(f"> Character {i} : {label}")
    
    
    if not os.path.exists(label_folder):
        os.makedirs(label_folder)

    
    for font_file in FONT_FILE_LIST:
        print(f"\t>> Font Path : {font_file} \r")
        font = ImageFont.truetype(font_file, FONT_SIZE)
        FONT_INDEX = len(os.listdir(label_folder)) + 1 

        
        image = Image.new('RGB', FONT_IMAGE_SIZE, color="black")
        draw = ImageDraw.Draw(image)
        draw.text(FONT_POSITION, char, font=font, fill="white")
        
        
        image_path = os.path.join(label_folder, f"{FONT_INDEX}.jpg")
        image.save(image_path)

        
        data_dict = {
            "font": font_file,
            "char": char,
            "label": label,
            "image": image_path
        }
        dataset.append(data_dict)

        data_fonts.append(font_file)
        data_images.append(image)
        data_labels.append(i)


try:
    pd.DataFrame(dataset).to_excel("dataset.xlsx", index=False)
except:
    pass


np.savez_compressed("Datasets/Create Dataset/persian_alphabet.npz", images=data_images, labels=data_labels, fonts=data_fonts)
