import cv2
import numpy as np
from keras.models import Sequential
from keras.preprocessing import image
from model.extract_bottleneck_features import *
from keras.applications.resnet50 import ResNet50
from keras.layers import GlobalAveragePooling2D, Dense
from keras.applications.resnet50 import preprocess_input, decode_predictions

class Model(object):
    '''
    Model is a class for load models and predict some new images
    '''

    def __init__(self):
        '''
            Initial function
        '''

        self.MODEL_FP = 'model/weights.best.Xception.hdf5'
        self.XML_FP = 'model/haarcascade_frontalface_alt.xml'
        self.model = None
        self.dog_names = ['01.Affenpinscher', '02.Afghan_hound', '03.Airedale_terrier', '04.Akita', 
            '05.Alaskan_malamute', '06.American_eskimo_dog', '07.American_foxhound', '08.American_staffordshire_terrier', 
            '09.American_water_spaniel', '10.Anatolian_shepherd_dog', '11.Australian_cattle_dog', '12.Australian_shepherd', 
            '13.Australian_terrier', '14.Basenji', '15.Basset_hound', '16.Beagle', '17.Bearded_collie', '18.Beauceron', 
            '19.Bedlington_terrier', '20.Belgian_malinois', '21.Belgian_sheepdog', '22.Belgian_tervuren', '23.Bernese_mountain_dog', 
            '24.Bichon_frise', '25.Black_and_tan_coonhound', '26.Black_russian_terrier', '27.Bloodhound', '28.Bluetick_coonhound', 
            '29.Border_collie', '30.Border_terrier', '31.Borzoi', '32.Boston_terrier', '33.Bouvier_des_flandres', '34.Boxer', 
            '35.Boykin_spaniel', '36.Briard', '37.Brittany', '38.Brussels_griffon', '39.Bull_terrier', '40.Bulldog', '41.Bullmastiff', 
            '42.Cairn_terrier', '43.Canaan_dog', '44.Cane_corso', '45.Cardigan_welsh_corgi', '46.Cavalier_king_charles_spaniel', 
            '47.Chesapeake_bay_retriever', '48.Chihuahua', '49.Chinese_crested', '50.Chinese_shar-pei', '51.Chow_chow', 
            '52.Clumber_spaniel', '53.Cocker_spaniel', '54.Collie', '55.Curly-coated_retriever', '56.Dachshund', '57.Dalmatian', 
            '58.Dandie_dinmont_terrier', '59.Doberman_pinscher', '60.Dogue_de_bordeaux', '61.English_cocker_spaniel', '62.English_setter', 
            '63.English_springer_spaniel', '64.English_toy_spaniel', '65.Entlebucher_mountain_dog', '66.Field_spaniel', '67.Finnish_spitz', 
            '68.Flat-coated_retriever', '69.French_bulldog', '70.German_pinscher', '71.German_shepherd_dog', '72.German_shorthaired_pointer', 
            '73.German_wirehaired_pointer', '74.Giant_schnauzer', '75.Glen_of_imaal_terrier', '76.Golden_retriever', '77.Gordon_setter', 
            '78.Great_dane', '79.Great_pyrenees', '80.Greater_swiss_mountain_dog', '81.Greyhound', '82.Havanese', '83.Ibizan_hound', 
            '84.Icelandic_sheepdog', '85.Irish_red_and_white_setter', '86.Irish_setter', '87.Irish_terrier', '88.Irish_water_spaniel', 
            '89.Irish_wolfhound', '90.Italian_greyhound', '91.Japanese_chin', '92.Keeshond', '93.Kerry_blue_terrier', '94.Komondor', 
            '95.Kuvasz', '96.Labrador_retriever', '97.Lakeland_terrier', '98.Leonberger', '99.Lhasa_apso', '00.Lowchen', '01.Maltese', 
            '02.Manchester_terrier', '03.Mastiff', '04.Miniature_schnauzer', '05.Neapolitan_mastiff', '06.Newfoundland', '07.Norfolk_terrier', 
            '08.Norwegian_buhund', '09.Norwegian_elkhound', '10.Norwegian_lundehund', '11.Norwich_terrier', '12.Nova_scotia_duck_tolling_retriever', 
            '13.Old_english_sheepdog', '14.Otterhound', '15.Papillon', '16.Parson_russell_terrier', '17.Pekingese', '18.Pembroke_welsh_corgi', 
            '19.Petit_basset_griffon_vendeen', '20.Pharaoh_hound', '21.Plott', '22.Pointer', '23.Pomeranian', '24.Poodle', '25.Portuguese_water_dog', 
            '26.Saint_bernard', '27.Silky_terrier', '28.Smooth_fox_terrier', '29.Tibetan_mastiff', '30.Welsh_springer_spaniel', 
            '31.Wirehaired_pointing_griffon', '32.Xoloitzcuintli', '33.Yorkshire_terrier']

    def path_to_tensor(self, img_path):
        '''
            a function to load image
        '''

        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        return np.expand_dims(x, axis=0)

    def face_detector(self, img_path):
        '''
            a function to detector a face in the image
        '''

        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(self.XML_FP)
        faces = face_cascade.detectMultiScale(gray)
        return len(faces) > 0

    def ResNet50_predict_labels(self, img_path):
        '''
            a function use ResNet to predict image
        '''

        img = preprocess_input(self.path_to_tensor(img_path))
        ResNet50_model = ResNet50(weights='imagenet')
        return np.argmax(ResNet50_model.predict(img))

    def dog_detector(self, img_path):
        '''
            a function to detector a dog in the image
        '''

        prediction = self.ResNet50_predict_labels(img_path)
        return ((prediction <= 268) & (prediction >= 151)) 

    def load(self):
        '''
            a function to load model
        '''

        Xception_model = Sequential()
        Xception_model.add(GlobalAveragePooling2D(input_shape=(7, 7, 2048)))
        Xception_model.add(Dense(133, activation='softmax'))
        Xception_model.load_weights(self.MODEL_FP)
        self.model = Xception_model
        return(Xception_model)

    def predict(self, img_path):
        '''
            a function to predict image
        '''

        bottleneck_feature = extract_Xception(self.path_to_tensor(img_path))
        predicted_vector = self.model.predict(bottleneck_feature)
        return(self.dog_names[np.argmax(predicted_vector)])

    def get_result(self, fp):
        '''
            a function to predict image with face & dog detector
        '''

        face_pred = self.face_detector(fp)
        dog_pred = self.dog_detector(fp)
        pred = self.predict(fp)
    
        if (not face_pred) and (not dog_pred):
            rst = 'Human Or Dog?'
        
        if face_pred:
            rst = 'Hello human'
        
        if dog_pred:
            rst = 'Hello dog'
        
        return('{}, You look like a... {}'.format(rst, pred))
