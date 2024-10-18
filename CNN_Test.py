import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image, ImageDraw, ImageFont
import numpy as np
import tensorflow as tf
from tensorflow import keras

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(96, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(25, activation='softmax')
    ])
    return model

win = tk.Tk()
win.configure(bg='#FFBF00')

def b1_click():
    global path2
    try:
        model = create_model()
        model.load_weights("model1.h5")
        print("Loaded model from disk")
        
        label = ["Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___Healthy",
                 "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
                 "Corn_(maize)___Healthy", "Corn_(maize)___Northern_Leaf_Blight", "Grape___Black_rot",
                 "Grape___Esca_(Black_Measles)", "Grape___Healthy", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
                 "Potato___Early_blight", "Potato___Healthy", "Potato___Late_blight", "Tomato___Bacterial_spot",
                 "Tomato___Early_blight", "Tomato___Healthy", "Tomato___Late_blight", "Tomato___Leaf_Mold",
                 "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
                 "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus"]
        
        path2 = filedialog.askopenfilename()
        print(path2)
        
        test_image = tf.keras.utils.load_img(path2, target_size=(128, 128))
        test_image = np.expand_dims(test_image, axis=0)
        result = model.predict(test_image)
        print(result)
        label2 = label[result.argmax()]
        print(label2)
        lbl.configure(text=label2)
        
        img = Image.open(path2)
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype("arial.ttf", 20)
        for i in range(0, 2000):
            draw.rectangle((10, 10, 100, 100), outline="white", width=5)
            draw.rectangle((10, 10, 100, 100), outline="red", width=5)
            draw.rectangle((10, 10, 100, 100), outline="green", width=5)
        draw.text((10, 120), label2, font=font, fill="white")
        
        img = img.resize((300, 300), Image.LANCZOS)
        img = ImageTk.PhotoImage(img)
        panel.configure(image=img, border=8, bg='green')
        panel.image = img
    except IOError:
        pass
    draw.rectangle((10, 10, 100, 100), outline="red", width=5)

label1 = tk.Label(win, width=16, text="Green vision", fg='lightgreen', bg='black', font=("Arial", 16, "bold"))
label1.pack()

panel = tk.Label(win)
panel.pack()

b1 = tk.Button(win, text='Pick leaf', width=16, height=1, fg='White', font=("Arial", 25, "bold"), command=b1_click, border=10, bg='green')
b1.pack()

lbl = tk.Button(win, text="Result", width=25, fg='white', bg='#007FFF', font=("Arial", 16, "bold"), border=10)
lbl.pack()

win.geometry("800x500")
win.title("Green Vision AI")
win.bind("<Return>", lambda event: b1_click())
win.mainloop()