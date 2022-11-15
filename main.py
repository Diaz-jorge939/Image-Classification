import tensorflow as tf
import numpy as np
import json 

from kivy.utils import platform
#from keras.models import load_model

from kivymd.app import MDApp
from kivy.uix.screenmanager import ScreenManager, Screen

from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.datatables import MDDataTable
from kivymd.uix.dialog import MDDialog
from kivymd.uix.button import MDFlatButton, MDRectangleFlatButton, MDRoundFlatButton

from kivy.lang.builder import Builder
from kivy.core.window import Window
from kivy.metrics import dp

# screen resolution: 320, 568
Window.size = (568,320)

screen_helper = """
ScreenManager:
    MenuScreen:
    CheckoutScreen:
    CameraScreen:

<MenuScreen>:
    name: 'menu'
    MDRectangleFlatButton:
        text: 'Checkout'
        pos_hint: {'center_x':0.4,'center_y':0.5}
        on_press: root.manager.current = 'checkout'
        canvas.before:
            PushMatrix
            Rotate:
                angle: 90
                origin: self.center
        canvas.after:
            PopMatrix

<CheckoutScreen>:
    name: 'checkout'
    on_parent: root.keyboard()
    MDTextField:
        id: searchbar
        hint_text: 'Enter Product'
        disabled: True
        multiline: False
        pos_hint: {'center_x':0.5,'center_y':0.9}
        size_hint_x: None
        width: 300
    MDRectangleFlatButton:
        text: 'Delete'
        pos_hint: {'center_x':0.3,'center_y':0.25}
        on_release: root.delete_search()
    MDRectangleFlatButton:
        text: 'Menu'
        pos_hint: {'center_x':0.1,'center_y':0.1}
        on_press: root.manager.current = 'menu'
    MDRectangleFlatButton:
        text: 'Camera'
        pos_hint: {'center_x':0.9,'center_y':0.1}
        on_press: 
            root.manager.transition.direction = 'left'
            root.manager.current = 'camera'

<CameraScreen>:
    name: 'camera'

    MDBoxLayout:
        orientation: 'vertical'
        
        Camera:
            id: camera
            resolution: (1080,720)
            play: True
            allow_stretch: True

        MDIconButton:
            icon: 'camera'
            pos_hint: {'center_x':0.5,'center_y':0.9}
            on_press: root.capture()
            
    MDLabel:
        id: prediction_label
        text: "Prediction: "
        font_size: 14
        pos_hint: {'x':0.08,'y':-.4}

    MDRectangleFlatButton:
        text: 'Checkout'
        pos_hint: {'center_x':0.9,'center_y':0.090}
        on_press: 
            root.manager.transition.direction = 'right'
            root.manager.current = 'checkout'
"""


class MenuScreen(Screen):
    pass

class CheckoutScreen(Screen):

    # Keyboard Start
    def keyboard(self):
        row1='qwertyuiop'
        row2='asdfghjkl'
        row3='zxcvbnm<'
        for i in range(len(row1)):
            key = MDRoundFlatButton(text=row1[i],pos_hint={'center_x':(i + 1) * 0.09,'center_y':0.7})
            key.bind(on_press = lambda x, key=row1[i] : self.get_key(key))
            self.add_widget(key)
        for i in range(len(row2)):
            key = MDRoundFlatButton(text=row2[i],pos_hint={'center_x':(i + 1.65) * 0.085,'center_y':0.55})
            key.bind(on_press = lambda x, key=row2[i] : self.get_key(key))
            self.add_widget(key)
        for i in range(len(row3)):
            key = MDRoundFlatButton(text=row3[i],pos_hint={'center_x':(i + 1.8) * 0.1,'center_y':0.40})
            key.bind(on_press = lambda x, key=row3[i] : self.get_key(key))
            self.add_widget(key)
    # Keyboard End

    # Supporting Functions Start
    def get_key(self, key):
        global item
        item = None
        self.prior_key = self.ids.searchbar.text
        if key != '<':
            if self.prior_key == "Search: ":
                self.ids.searchbar.text = ''
                self.ids.searchbar.text = f'Search: {key}'
            else:
                self.ids.searchbar.text = f'{self.prior_key}{key}'
        else:
            self.ids.searchbar.text = f'{self.prior_key[:-1]}'

        item = self.ids.searchbar.text

    def delete_search(self):
        self.ids.searchbar.text = ""

    def get_item(self):
        if 'item' in globals():
            return item
        return False
    # Supporting Functions End


class CameraScreen(Screen):
    
    def capture(self, *args):
        
        camera = self.ids['camera']
        camera.export_to_png('Fruit.png')
        
        self.preprocess('Fruit.png')

    def preprocess(self, filepath):
    
        new_img = tf.keras.utils.load_img(filepath, target_size=(100, 100))
        img = tf.keras.utils.img_to_array(new_img)
        img = np.expand_dims(img, axis=0)
        img = img/255.0

        self.model_prediction(img)

    '''
        #function uses keras_model h5 file to make preidction
        model = 'Fruits-CNN_small'

        def model_prediction(self, img):

        y_prob = model.predict(img) #returns numpy array of class probabilities
        y_classes = y_prob.argmax(axis=1)

        img_label_index = y_classes.item()      
        label_prob = y_prob[0][img_label_index] * 100 
        label_prob = round(label_prob, 2)
        
        with open('class_labels.json') as read_file:
            class_dict = json.load(read_file)
            labels = class_dict["class_labels"]
        
        pred_label = labels[img_label_index]
        
        self.ids.prediction_label.text = "Prediction: " + pred_label + " " + str(label_prob) + "%" '''
    def model_prediction(self,img):
        # Load TFLite model and allocate tensors.
        interpreter = tf.lite.Interpreter(model_path="cnn_fruits.tflite") #cnn_fruits.tflite
        interpreter.allocate_tensors()

        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        input_data = img
        interpreter.set_tensor(input_details[0]['index'], input_data)

        interpreter.invoke()

        # The function `get_tensor()` returns a copy of the tensor data.
        output_data = interpreter.get_tensor(output_details[0]['index'])
        output_label = output_data.argmax(axis=1)
        
        prediction_index = output_label.item()     
       
        prediction_prob = output_data[0][prediction_index] * 100 
        prediction_prob = round(prediction_prob, 2)
        
        with open('class_labels.json') as read_file:
            class_dict = json.load(read_file)
            labels = class_dict["class_labels"]
        
        prediction = labels[prediction_index]
        
        self.ids.prediction_label.text = "Prediction: " + prediction + " " + str(prediction_prob) + "%"

# Create the screen manager
sm = ScreenManager()
sm.add_widget(MenuScreen(name='menu'))
sm.add_widget(CheckoutScreen(name='checkout'))
sm.add_widget(CameraScreen(name='camera'))

class POS(MDApp):
    def build(self):

        if platform == "android":
            from android.permissions import request_permissions, Permission
            request_permissions([
                Permission.CAMERA,
                Permission.WRITE_EXTERNAL_STORAGE,
                Permission.READ_EXTERNAL_STORAGE
            ])
        screen = Builder.load_string(screen_helper)
        return screen

POS().run()
