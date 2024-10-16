try:

    import tensorflow as tf
    import cv2
    import os
    import pickle
    import numpy as np
    import warnings
    warnings.filterwarnings('ignore')
    print("Libraries Loaded Successfully")
except:
    print("Library not Found ! ")



class DataLoader:
    def __init__(self, path = '', image_size = 50, shrink = 0, padding = 10, invert = False) -> None:
        self.PATH = path
        self.IMAGE_SIZE = image_size
        self.PADDING = padding
        self.INVERT = invert
        self.SLICE = shrink

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
            try :
                self.list_categories = sorted(self.list_categories, key=lambda x : int(x))
            except:
                pass

        print("Found Categories :",self.list_categories,'\n')
        return self.list_categories
    

    def preprocess_image(self, image):

        image_data_temp = cv2.imread(image,cv2.IMREAD_GRAYSCALE)                 
        image_padded = cv2.copyMakeBorder(image_data_temp, self.PADDING, self.PADDING, self.PADDING, self.PADDING, cv2.BORDER_CONSTANT, value=0)
        if self.INVERT : image_padded = 255 - image_padded
        image_temp_resize = cv2.resize(image_padded,(self.IMAGE_SIZE,self.IMAGE_SIZE))

        # cv2.imshow('Padded Image', image_temp_resize)
        # cv2.waitKey(0)

        return image_temp_resize
        
    
    
    def Process_Images(self):
        # """
        # Return Numpy array of image\n
        # :return: X_Data, Y_Data
        # """
        # try:
            self.CATEGORIES = self.get_categories()
            for categories in self.CATEGORIES:                                                  

                train_folder_path = os.path.join(self.PATH, categories)                         
                class_index = self.CATEGORIES.index(categories)                                 

                for img in os.listdir(train_folder_path):                                       
                    new_path = os.path.join(train_folder_path, img)                             

                    try:
                        image_temp_resize = self.preprocess_image(new_path)

                        if image_temp_resize is not None:  
                            self.x_data.append(image_temp_resize[self.SLICE:self.IMAGE_SIZE-self.SLICE, self.SLICE:self.IMAGE_SIZE-self.SLICE])
                            self.y_data.append(class_index)

                    except Exception as e:
                        print(f"Error processing {new_path}: {e}")



            X_Data = np.asarray(self.x_data) / (255.0)
            Y_Data = np.asarray(self.y_data)

            

            X_Data = X_Data.reshape(-1, self.IMAGE_SIZE-2*self.SLICE, self.IMAGE_SIZE-2*self.SLICE)

            return X_Data, Y_Data
        
        # except:
        #     print("Failed to run Function Process Image ")


    def load_data(self):

        print('Loading Files and Dataset ...')

        X_Data,Y_Data = self.Process_Images()
        # np.savez_compressed("dataset", X_train=X_Data, y_train=Y_Data)


        # data = np.load("dataset.npz")
        # x_train = data["X_train"]
        return X_Data,Y_Data
        



DATASET1 = "Datasets\DS-1"
DATASET2 = "Datasets\DS-2"

LABELS = {0 : 'Alef',
          1 : 'Be',
          2 : 'Pe',
          3 : 'Te',
          4 : 'Se',
          5 : 'Jim',
          6 : 'Che',
          7 : 'H',
          8 : 'Khe',
          9 : 'Dal',
          10 : 'Zal',
          11 : 'Re',
          12 : 'Ze',
          13 : 'Zhe',
          14 : 'Sin',
          15 : 'Shin',
          16 : 'Sad', 
          17 : 'Zad',
          18 : 'Ta',
          19 : 'Za',
          20 : 'Ayin',
          21 : 'Ghayin',
          22 : 'Fe',
          23 : 'Ghaf', 
          24 : 'Kaf',
          25 : 'Gaf', 
          26 : 'Lam',
          27 : 'Mim',
          28 : 'Noon',
          29 : 'Vav', 
          30 : 'He',
          31 : 'Ye',
          32 : 'Zero',
          33 : 'One',
          34 : 'Two',
          35 : 'Three',
          36 : 'Four',
          37 : 'Five',
          38 : 'Six',
          39 : 'Seven',
          40 : 'Eight',
          41 : 'Nine',
          42 : 'Five'}


if __name__ == "__main__":

    dataset = DataLoader(path=DATASET1,
                    image_size=64,
                    shrink=22,
                    padding=0,
                    invert=True)

    X_Data,Y_Data = dataset.load_data()

    print(X_Data.shape)
    print(Y_Data.shape)

    cv2.imshow(f"test no. {Y_Data[400]}", X_Data[400])

    cv2.waitKey(0)