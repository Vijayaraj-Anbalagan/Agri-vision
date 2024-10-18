import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

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

def load_labels():
    return ["Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___Healthy",
            "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
            "Corn_(maize)___Healthy", "Corn_(maize)___Northern_Leaf_Blight", "Grape___Black_rot",
            "Grape___Esca_(Black_Measles)", "Grape___Healthy", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
            "Potato___Early_blight", "Potato___Healthy", "Potato___Late_blight", "Tomato___Bacterial_spot",
            "Tomato___Early_blight", "Tomato___Healthy", "Tomato___Late_blight", "Tomato___Leaf_Mold",
            "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
            "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus"]

def main():
    st.set_page_config(page_title="Green Vision AI", page_icon="🌿")
    
    # Add custom CSS
    st.markdown("""
        <style>
        .stApp {
            background-color: #f0f8f0;
        }
        .title {
            color: #2e7d32;
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown("<h1 class='title'>Green Vision AI 🌿</h1>", unsafe_allow_html=True)
    st.markdown("<h3 class='title'>Plant Disease Detection System</h3>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        try:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            if st.button("Analyze Image"):
                with st.spinner("Analyzing..."):
                    # Load model and make prediction
                    model = create_model()
                    try:
                        model.load_weights("model1.h5")
                        
                        # Preprocess image
                        test_image = image.resize((128, 128))
                        test_image = np.array(test_image)
                        test_image = np.expand_dims(test_image, axis=0)
                        
                        # Make prediction
                        result = model.predict(test_image)
                        predicted_label = load_labels()[result.argmax()]
                        
                        # Display results
                        st.success("Analysis Complete!")
                        st.markdown(f"### Detected Condition:")
                        st.markdown(f"**{predicted_label.replace('___', ' - ').replace('_', ' ')}**")
                        
                        # Display confidence score
                        confidence = float(result.max()) * 100
                        st.markdown(f"Confidence: {confidence:.2f}%")
                        
                    except FileNotFoundError:
                        st.error("Model weights file (model1.h5) not found. Please ensure the model is properly trained and saved.")
                        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()