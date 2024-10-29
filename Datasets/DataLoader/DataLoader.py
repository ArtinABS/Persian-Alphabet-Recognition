import os
import cv2
import numpy as np
import warnings


warnings.filterwarnings('ignore')


try:
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    print("Libraries Loaded Successfully")
except ImportError:
    print("Failed to Load Libraries!")



class DataLoader:
    def __init__(self, path='', image_size=50, shrink=0, padding=10, contrast=100, zoom=0.4, invert=False):
        self.PATH = path
        self.IMAGE_SIZE = image_size
        self.PADDING = padding
        self.INVERT = invert
        self.CONTRAST = contrast
        self.ZOOM = zoom
        self.SLICE = shrink

        self.x_data = []
        self.y_data = []
        self.labels = []
        self.CATEGORIES = []
        self.list_categories = []


    def get_categories(self):
        """Get and sort categories based on folder names."""
        for folder in os.listdir(self.PATH):
            label = folder.split("-")[0]
            self.labels.append(label)
            self.list_categories.append(folder)
        
        try:
            self.list_categories = sorted(self.list_categories, key=lambda x: int(x.split("-")[0]))
        except ValueError:
            self.list_categories = sorted(self.list_categories)

        print("Found Categories:", self.list_categories, '\n')
        return self.list_categories
    
    
    def centerize(self, image):
        """Center the letter in the image using the bounding box of all contours."""
        
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return image  

        min_x, min_y = np.inf, np.inf
        max_x, max_y = -np.inf, -np.inf

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x + w)
            max_y = max(max_y, y + h)

        min_x = max(min_x, 0)
        min_y = max(min_y, 0)
        max_x = min(max_x, image.shape[1])
        max_y = min(max_y, image.shape[0])

        cropped_image = image[min_y:max_y, min_x:max_x]

        centered_image = np.zeros((self.IMAGE_SIZE, self.IMAGE_SIZE), dtype=np.uint8)

        w, h = cropped_image.shape[1], cropped_image.shape[0]
        start_x = (self.IMAGE_SIZE - w) // 2
        start_y = (self.IMAGE_SIZE - h) // 2

        centered_image[start_y:start_y+h, start_x:start_x+w] = cropped_image

        return centered_image
    


    def normalize(self, image):
        """Zoom the image based on the target size ratio, keeping the original shape."""
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return image  # Return original if no contours are found

        x, y, w, h = cv2.boundingRect(np.concatenate(contours))

        target_size = int(self.IMAGE_SIZE * self.ZOOM)
        scale_factor = min(target_size / w, target_size / h)

        # Resize the cropped image to the target size without altering the original image
        cropped_image = image[y:y+h, x:x+w]
        zoomed_image = cv2.resize(cropped_image, (int(w * scale_factor), int(h * scale_factor)))

        # Create a blank image with the same shape as the input image
        output_image = np.zeros_like(image)

        # Calculate the position to place the zoomed image in the center of the output image
        start_x = (output_image.shape[1] - zoomed_image.shape[1]) // 2
        start_y = (output_image.shape[0] - zoomed_image.shape[0]) // 2

        # Place the zoomed image in the center of the blank image
        output_image[start_y:start_y + zoomed_image.shape[0], start_x:start_x + zoomed_image.shape[1]] = zoomed_image

        return output_image



    def enhance(self, image):
        """Enhance the quality of image by boosting the values below a certain threshold"""

        _, enhanced_image = cv2.threshold(image, self.CONTRAST, 255, cv2.THRESH_BINARY)
        return enhanced_image
        

    def preprocess_image(self, image_path):
        """Preprocess the image by zooming in (shrinking) while keeping the final size constant."""

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None

        padded_image = cv2.copyMakeBorder(image, self.PADDING, self.PADDING, self.PADDING, self.PADDING, 
                                          cv2.BORDER_CONSTANT, value=0)
        if self.INVERT:
            padded_image = 255 - padded_image


        resized_image = cv2.resize(padded_image, (self.IMAGE_SIZE, self.IMAGE_SIZE))


        cropped_image = resized_image[self.SLICE:self.IMAGE_SIZE - self.SLICE,
                                      self.SLICE:self.IMAGE_SIZE - self.SLICE]


        zoomed_image = cv2.resize(cropped_image, (self.IMAGE_SIZE, self.IMAGE_SIZE))

        centered_image = self.centerize(zoomed_image)

        next_step = centered_image

        if self.ZOOM != None:
            normalized_image = self.normalize(centered_image)
            next_step = normalized_image

        if self.CONTRAST != None : 
            enhanced_image = self.enhance(next_step)
            next_step = enhanced_image

        return next_step


    def process_images(self):
        """Process all images from the dataset."""
        self.CATEGORIES = self.get_categories()

        for category in self.CATEGORIES:
            category_path = os.path.join(self.PATH, category)
            class_index = int(category.split('-')[0])-1

            for img_name in os.listdir(category_path):
                img_path = os.path.join(category_path, img_name)

                try:
                    image = self.preprocess_image(img_path)
                    if image is not None:
                        self.x_data.append(image)
                        self.y_data.append(class_index)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")

        X_Data = np.asarray(self.x_data) / 255.0
        Y_Data = np.asarray(self.y_data)
        X_Data = X_Data.reshape(-1, self.IMAGE_SIZE, self.IMAGE_SIZE)

        return X_Data, Y_Data
    

    def load_data(self):
        """Load and return the dataset."""

        print('Loading Files and Dataset ...')

        X_Data, Y_Data = self.process_images()

        X_train, X_test, y_train, y_test = train_test_split(X_Data, Y_Data, train_size=0.8, stratify=Y_Data, random_state=42)

        return X_train, y_train, X_test, y_test


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
    DATASET_PATH = "Datasets/DS-2"
    
    dataset_loader = DataLoader(path=DATASET_PATH, image_size=64, shrink=0, padding=20, contrast=None, invert=False)
    X_train, y_train, X_test, y_test = dataset_loader.load_data()

    print(f"X_Train shape: {X_train.shape}")
    print(f"Y_Train shape: {y_train.shape}")
    print(f"X_Test shape: {X_test.shape}")
    print(f"Y_Test shape: {y_test.shape}")

    exmaple_idx = 250

    cv2.imshow(f"Test no. {y_train[exmaple_idx]}", X_train[exmaple_idx])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
