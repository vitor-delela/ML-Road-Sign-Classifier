from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

model = load_model('./models/german_model_10epochs_rmsprop_comDropout_semDataAug.h5')
# Carregar um arquivo CSV que mapeia índices de classe para nomes de classe
class_mapping = pd.read_csv('/Users/vitordelela/Documents/PUCRS/7 SEM/AM/T2/German_Dataset/signname.csv') 

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

        # image_path = '/Users/vitordelela/Downloads/ice.jpg'
        try:
            img = Image.open(file_path)
        except Exception as e:
            print(f"Erro ao abrir a imagem: {e}")

        # Verificar o modo da imagem (deve ser 'RGB')
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Remover o canal alfa se estiver presente
        img = img.convert('RGB')

        # Redimensionar a imagem para o formato do seu modelo (32, 32, 3)
        img = img.resize((32, 32))
        img_array = np.array(img)
        img_array = img_array.astype('float32') / 255.0 # Normalizar os valores para o intervalo [0, 1]
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)

        top_indices = np.argsort(prediction[0])[-3:][::-1] # Obter os índices das três maiores probabilidades
        top_classes = class_mapping.loc[top_indices, 'SignName'].values # Mapear os índices para os nomes das classes usando o arquivo CSV

        # Obter as três maiores probabilidades
        top_probabilities = prediction[0][top_indices]
        predicted_class_index = np.argmax(prediction)

        retorno = ''
        # Imprimir os índices e probabilidades
        for i, (class_name, probability) in enumerate(zip(top_classes, top_probabilities), 1):
            retorno += f'Top {i}: {class_name} - probabilidade {probability:.2%}\n'
            print(f'Top {i}: Classe {class_name} - probabilidade {probability:.2%}')

        phrase_label.config(text=retorno)

# Cria a janela principal
root = tk.Tk()
root.title("Classificador")

# Cria uma label para exibir a imagem
image_label = tk.Label(root)
image_label.pack(padx=10, pady=10)

phrase_label = tk.Label(root, text="-", font=("Helvetica", 14))
phrase_label.pack(pady=10)

# Botão para selecionar uma imagem
load_button = tk.Button(root, text="Carregar Imagem", command=load_image)
load_button.pack(pady=10)

# Inicia o loop principal da interface gráfica
root.mainloop()




