import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv, find_dotenv
import os
import google.generativeai as genai
import time

# Load environment variables
load_dotenv(find_dotenv())
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Color scheme
PRIMARY_COLOR = "#4CAF50"
SECONDARY_COLOR = "#8BC34A"
BACKGROUND_COLOR = "#F1F8E9"
TEXT_COLOR = "#000000"

# Initialize session state for tab control and disease info
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "Disease Detection"
if 'disease_info' not in st.session_state:
    st.session_state.disease_info = None

# Page config
st.set_page_config(page_title="AgriVision", page_icon="🌾", layout="wide")

# Custom CSS
st.markdown(f"""
    <style>
    .stApp {{
        background-color: {BACKGROUND_COLOR};
        font-family: 'Roboto', sans-serif;
        color: {TEXT_COLOR};
    }}
    .stButton>button {{
        background-color: {PRIMARY_COLOR};
        color: white;
        font-size: 16px;
        border-radius: 20px;
        padding: 10px 24px;
        transition: all 0.3s ease;
    }}
    .stButton>button:hover {{
        background-color: {SECONDARY_COLOR};
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }}
    .disease-button {{
        background-color: #FF5722 !important;
        margin: 10px 0 !important;
    }}
    .disease-button:hover {{
        background-color: #F4511E !important;
    }}
    .title {{
        color: {PRIMARY_COLOR};
        text-align: center;
    }}
    @keyframes fadeIn {{
        from {{ opacity: 0; }}
        to {{ opacity: 1; }}
    }}
    .fade-in {{
        animation: fadeIn 0.5s ease-in-out;
    }}
    </style>
    """, unsafe_allow_html=True)

# Helper functions
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

def get_gemini_response(input, image=None):
    model = genai.GenerativeModel("gemini-1.5-pro-latest")
    if image:
        response = model.generate_content([input, image[0]])
    else:
        response = model.generate_content(input)
    return response.text

def input_image_setup(uploaded_file):
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        image_parts = [
            {
                "mime_type": uploaded_file.type,
                "data": bytes_data,
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No image uploaded")

def switch_to_farming_assistant(disease_name):
    st.session_state.active_tab = "Farming Assistant"
    st.session_state.disease_info = disease_name
    st.rerun()

# Main App
st.title("AgriVision 🌾")
st.markdown("---")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["Disease Detection", "Crop Analysis", "Farming Assistant", "Crop Planning"])

# Set the active tab
if st.session_state.active_tab == "Disease Detection":
    current_tab = tab1
elif st.session_state.active_tab == "Crop Analysis":
    current_tab = tab2
elif st.session_state.active_tab == "Farming Assistant":
    current_tab = tab3
else:
    current_tab = tab4

# Disease Detection Tab
with tab1:
    st.header("Plant Disease Detection")
    uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"], key="disease_detection")
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            if st.button("Detect Disease"):
                with st.spinner("Analyzing..."):
                    model = create_model()
                    try:
                        model.load_weights("model1.h5")
                        test_image = image.resize((128, 128))
                        test_image = np.array(test_image)
                        test_image = np.expand_dims(test_image, axis=0)
                        result = model.predict(test_image)
                        predicted_label = load_labels()[result.argmax()]
                        confidence = float(result.max()) * 100
                        
                        disease_name = predicted_label.replace('___', ' - ').replace('_', ' ')
                        
                        st.success("Analysis Complete!")
                        st.markdown(f"### Detected Condition:")
                        st.markdown(f"**{disease_name}**")
                        st.markdown(f"Confidence: {confidence:.2f}%")
                        
                        # Create a button with the disease name
                        if "Healthy" not in predicted_label:
                            col1, col2, col3 = st.columns([1, 2, 1])
                            with col2:
                                if st.button(f"Get Treatment Advice for {disease_name}", 
                                           key="disease_button",
                                           help="Click to get detailed treatment advice",
                                           use_container_width=True):
                                    switch_to_farming_assistant(disease_name)
                        
                    except FileNotFoundError:
                        st.error("Model weights file (model1.h5) not found.")
                        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Crop Analysis Tab
with tab2:
    st.header("Crop Analysis")
    uploaded_file = st.file_uploader("Upload an image of your crop or field...", type=["jpg", "jpeg", "png"], key="crop_analysis")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Analyze Crop"):
            input_prompt = """
            You are an expert agricultural analyst examining the crop or field in the image.
            Provide a detailed analysis including:
            1. Identification of the crop (if visible)
            2. Assessment of crop health
            3. Identification of any visible issues (pests, diseases, nutrient deficiencies)
            4. Suggestions for improvement or treatment
            5. Estimated crop yield based on visible conditions
            """

            with st.spinner("Analyzing..."):
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                image_data = input_image_setup(uploaded_file)
                response = get_gemini_response(input_prompt, image_data)
            st.success("Analysis Complete!")
            st.markdown(response)

# Farming Assistant Tab
with tab3:
    st.header("Farming Assistant")
    
    # Set default question if disease info is available
    if st.session_state.disease_info:
        default_question = f"What is the best treatment and prevention method for {st.session_state.disease_info}? Please provide detailed steps and organic/chemical options."
    else:
        default_question = ""
    
    user_input = st.text_area("Ask any farming-related question:", 
                             value=default_question,
                             height=100)
    
    if st.button("Get Answer"):
        with st.spinner("Generating advice..."):
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            response = get_gemini_response(f"As an agricultural expert, please answer the following question: {user_input}")
        st.markdown(response)
        
        # Reset disease info after displaying the response
        st.session_state.disease_info = None

# Crop Planning Tab
with tab4:
    st.header("Crop Planning Assistant")
    
    with st.form("crop_planning_form"):
        st.subheader("Enter Your Farm Details")
        location = st.text_input("Location (City, Country)")
        soil_type = st.selectbox("Soil Type", ["Clay", "Sandy", "Loamy", "Silt", "Peat", "Chalky"])
        farm_size = st.number_input("Farm Size (in acres)", min_value=0.1, step=0.1)
        season = st.selectbox("Planting Season", ["Spring", "Summer", "Fall", "Winter"])
        water_availability = st.slider("Water Availability", 1, 10, 5)
        
        submit_button = st.form_submit_button("Generate Crop Recommendations")

    if submit_button:
        prompt = f"""
        As an agricultural expert, provide crop recommendations based on the following farm details:
        Location: {location}
        Soil Type: {soil_type}
        Farm Size: {farm_size} acres
        Planting Season: {season}
        Water Availability: {water_availability}/10

        Please suggest:
        1. Top 3 suitable crops for these conditions
        2. Optimal planting and harvesting times for each crop
        3. Estimated yield per acre for each crop
        4. Any special considerations or farming practices for the suggested crops
        5. Potential challenges and how to mitigate them
        """

        with st.spinner("Generating recommendations..."):
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.02)
                progress_bar.progress(i + 1)
            response = get_gemini_response(prompt)
        st.markdown(response)

# Footer
st.markdown("---")
st.markdown("Powered by AgriVision 🌾 | Helping farmers grow smarter")
