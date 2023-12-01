from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

model = load_model('./models/model_10epochs_rmsprop_dropout.h5')
class_mapping = pd.read_csv('./German_Dataset/signname.csv') 

def load_image():
    file_path = filedialog.askopenfilename(filetypes=[("Images", ["*.jpg", "*.png", "*.jpeg"])])

    if file_path:
        original_image = Image.open(file_path)
        max_width = 450
        max_height = 500
        image = original_image.resize((max_width, max_height), Image.ANTIALIAS)
        tk_image = ImageTk.PhotoImage(image)
        image_label.configure(image=tk_image)
        image_label.image = tk_image

        phrase_label.config(text="Carregando...")

        try:
            img = Image.open(file_path)
        except Exception as e:
            print(f"Error: {e}")

        if img.mode != 'RGB':
            img = img.convert('RGB')

        img = img.convert('RGB')

        # Resize image to (32, 32, 3)
        img = img.resize((32, 32))
        img_array = np.array(img)
        img_array = img_array.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)

        top_indices = np.argsort(prediction[0])[-3:][::-1] # Getting top 3 prob
        top_classes = class_mapping.loc[top_indices, 'SignName'].values

        # Obter as três maiores probabilidades
        top_probabilities = prediction[0][top_indices]
        predicted_class_index = np.argmax(prediction)

        _return = ''
        for i, (class_name, probability) in enumerate(zip(top_classes, top_probabilities), 1):
            _return += f'Top {i}: {class_name} - Probability {probability:.2%}\n'
            print(f'Top {i}: Class {class_name} - Probability {probability:.2%}')

        phrase_label.config(text=retorno)


root = tk.Tk()
root.title("Classifier")
image_label = tk.Label(root)
image_label.pack(padx=10, pady=10)
phrase_label = tk.Label(root, text="-", font=("Helvetica", 14))
phrase_label.pack(pady=10)
load_button = tk.Button(root, text="Load Image", command=load_image)
load_button.pack(pady=10)

root.mainloop()




