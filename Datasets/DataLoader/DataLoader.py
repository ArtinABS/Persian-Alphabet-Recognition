try:

    import tensorflow as tf
    import cv2
    import os
    import pickle
    import numpy as np
    import warnings
    warnings.filterwarnings('ignore')
    print("Library Loaded Successfully ..........")
except:
    print("Library not Found ! ")



class DataLoader:
    def __init__(self, path = '', image_size = 50, padding = 10, invert = False) -> None:
        self.PATH = path
        self.IMAGE_SIZE = image_size
        self.PADDING = padding
        self.INVERT = invert

        self.image_data = []
        self.x_data = []
        self.y_data = [] 
        self.labels = []
        self.CATEGORIES = []

        self.list_categories = []


    def get_categories(self):
        for path in (os.listdir(self.PATH)):
            label = path.split("-")[0]
            self.labels.append(label)
            self.list_categories.append(path)
        try :
            self.list_categories = sorted(self.list_categories, key=lambda x : int(x.split("-")[0]))
        except: 
            pass

        print("Found Categories ",self.list_categories,'\n')
        return self.list_categories
    

    def preprocess_image(self, image):

        image_data_temp = cv2.imread(image,cv2.IMREAD_GRAYSCALE)                 # Read Image as numbers
        image_padded = cv2.copyMakeBorder(image_data_temp, self.PADDING, self.PADDING, self.PADDING, self.PADDING, cv2.BORDER_CONSTANT, value=0)
        if self.INVERT : image_padded = 255 - image_padded
        image_temp_resize = cv2.resize(image_padded,(self.IMAGE_SIZE,self.IMAGE_SIZE))

        # cv2.imshow('Padded Image', image_temp_resize)
        # cv2.waitKey(0)

        return image_temp_resize
        
    
    
    def Process_Images(self):
        try:
            """
            Return Numpy array of image
            :return: X_Data, Y_Data
            """
            self.CATEGORIES = self.get_categories()
            for categories in self.CATEGORIES:                                                  # Iterate over categories

                train_folder_path = os.path.join(self.PATH, categories)                         # Folder Path
                class_index = self.CATEGORIES.index(categories)                                 # this will get index for classification

                for img in os.listdir(train_folder_path):                                       # This will iterate in the Folder
                    new_path = os.path.join(train_folder_path, img)                             # image Path

                    try:
                        image_temp_resize = self.preprocess_image(new_path)

                        if image_temp_resize is not None:  
                            self.x_data.append(image_temp_resize)
                            self.y_data.append(class_index)

                    except Exception as e:
                        print(f"Error processing {new_path}: {e}")



            X_Data = np.asarray(self.x_data) / (255.0)
            Y_Data = np.asarray(self.y_data)

            # reshape x_Data

            X_Data = X_Data.reshape(-1, self.IMAGE_SIZE, self.IMAGE_SIZE)

            return X_Data, Y_Data
        except:
            print("Failed to run Function Process Image ")


    def load_data(self):

        print('Loading File and Dataset  ..........')

        X_Data,Y_Data = self.Process_Images()
        return X_Data,Y_Data
        



DATASET1 = "Datasets\DS-1"
DATASET2 = "Datasets\DS-2"


if __name__ == "__main__":

    dataset = DataLoader(path=DATASET2,
                    image_size=50,
                    padding=20,
                    invert=False)

    X_Data,Y_Data = dataset.load_data()

    print(X_Data.shape)
    print(Y_Data.shape)

    cv2.imshow('test', X_Data[400])

    cv2.waitKey(0)