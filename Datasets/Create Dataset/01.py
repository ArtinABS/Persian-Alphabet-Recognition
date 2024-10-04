import os
import shutil
from fontTools.ttLib import TTFont
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pandas as pd



FONT_FILE_LIST = [f"fonts/{f}" for f in os.listdir("fonts")]
FONT_SIZE = 48
FONT_IMAGE_SIZE = (64, 64)
FONT_POSITION = (8, 8)


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


if os.path.exists("dataset"):
    shutil.rmtree("dataset", ignore_errors=True)
os.mkdir("dataset")


for i, char in enumerate(PERSIAN_CHARACTERS):
    label = LABELS[i]
    label_folder = os.path.join("dataset", label)
    
    
    if not os.path.exists(label_folder):
        os.makedirs(label_folder)

    
    for font_file in FONT_FILE_LIST:
        font = ImageFont.truetype(font_file, FONT_SIZE)
        FONT_INDEX = len(os.listdir(label_folder)) + 1 

        
        image = Image.new('RGB', FONT_IMAGE_SIZE, color="black")
        draw = ImageDraw.Draw(image)
        draw.text(FONT_POSITION, char, font=font, fill="white")
        
        
        image_path = os.path.join(label_folder, f"{FONT_INDEX}.bmp")
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


pd.DataFrame(dataset).to_excel("dataset.xlsx", index=False)


np.savez_compressed("persian_alphabet.npz", images=data_images, labels=data_labels, fonts=data_fonts)
