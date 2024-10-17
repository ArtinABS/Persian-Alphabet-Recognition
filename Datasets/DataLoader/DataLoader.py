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
    def __init__(self, path='', image_size=50, shrink=0, padding=10, threshold=100, invert=False):
        self.PATH = path
        self.IMAGE_SIZE = image_size
        self.PADDING = padding
        self.INVERT = invert
        self.THRESHOLD = threshold
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
        """Center the letter in the image using contours to find the bounding box."""

        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return image


        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))

        cropped_image = image[y:y+h, x:x+w]

        centered_image = np.zeros((self.IMAGE_SIZE, self.IMAGE_SIZE), dtype=np.uint8)

        start_x = (self.IMAGE_SIZE - w) // 2
        start_y = (self.IMAGE_SIZE - h) // 2

        centered_image[start_y:start_y+h, start_x:start_x+w] = cropped_image

        return centered_image


    def enhance(self, image):
        """Enhance the quality of image by boosting the values below a certain threshold"""

        _, enhanced_image = cv2.threshold(image, self.THRESHOLD, 255, cv2.THRESH_BINARY)
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

        if self.THRESHOLD != None : 
            enhanced_image = self.enhance(centered_image)
            return enhanced_image
        else : return centered_image


    def process_images(self):
        """Process all images from the dataset."""
        self.CATEGORIES = self.get_categories()

        for category in self.CATEGORIES:
            category_path = os.path.join(self.PATH, category)
            class_index = self.CATEGORIES.index(category)

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

        X_train, X_test, y_train, y_test = train_test_split(X_Data, Y_Data, train_size=0.8, stratify=Y_Data, random_state=0)

        return (X_train, y_train), (X_test, y_test)


if __name__ == "__main__":
    DATASET_PATH = "Datasets/DS-2 changed"
    
    dataset_loader = DataLoader(path=DATASET_PATH, image_size=64, shrink=0, padding=15, threshold=None, invert=False)
    (X_train, y_train), (X_test, y_test) = dataset_loader.load_data()

    print(f"X_Train shape: {X_train.shape}")
    print(f"Y_Train shape: {y_train.shape}")
    print(f"X_Test shape: {X_test.shape}")
    print(f"Y_Test shape: {y_test.shape}")

    cv2.imshow(f"Test no. {y_train[20]}", X_train[20])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
