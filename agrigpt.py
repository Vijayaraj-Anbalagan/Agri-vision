import streamlit as st
from dotenv import load_dotenv, find_dotenv
import os
import google.generativeai as genai
from PIL import Image
import time

# Load environment variables from the .env file
load_dotenv(find_dotenv())

# Configure Streamlit page settings
st.set_page_config(page_title="AgriGPT: Your Agricultural Assistant", page_icon="🌾", layout="wide")

# Configure Google Generative AI library with an API key from environment variables
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Define color scheme
PRIMARY_COLOR = "#4CAF50"
SECONDARY_COLOR = "#8BC34A"
BACKGROUND_COLOR = "#F1F8E9"
TEXT_COLOR = "#33691E"

# Apply custom CSS to enhance the Streamlit app's appearance
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
    .stTextInput>div>div>input {{
        border-radius: 20px;
    }}
    .stTextArea textarea {{
        border-radius: 20px;
    }}
    .stSelectbox>div>div>select {{
        border-radius: 20px;
    }}
    .css-1v3fvcr {{
        background-color: {BACKGROUND_COLOR};
    }}
    .css-1d391kg {{
        background-color: {PRIMARY_COLOR};
    }}
    .stProgress > div > div > div > div {{
        background-color: {SECONDARY_COLOR};
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

# Define functions to handle Gemini API responses
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

# Sidebar configuration
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Crop Analysis", "Farming Assistant", "Crop Planning"])

# Main content area
st.title("AgriGPT: Your Agricultural Assistant")
st.markdown("---")

if page == "Crop Analysis":
    st.header("Crop Analysis")
    uploaded_file = st.file_uploader("Upload an image of your crop or field...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

    submit = st.button("Analyze Crop")
    
    input_prompt = """
    You are an expert agricultural analyst examining the crop or field in the image.
    Provide a detailed analysis including:
    1. Identification of the crop (if visible)
    2. Assessment of crop health
    3. Identification of any visible issues (pests, diseases, nutrient deficiencies)
    4. Suggestions for improvement or treatment
    5. Estimated crop yield based on visible conditions

    If the image does not contain crops or a field, clearly state that no agricultural scene is detected.
    """

    if submit:
        with st.spinner("Analyzing..."):
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            image_data = input_image_setup(uploaded_file)
            response = get_gemini_response(input_prompt, image_data)
        st.success("Analysis Complete!")
        st.markdown('<div class="fade-in">', unsafe_allow_html=True)
        st.subheader("Crop Analysis Results")
        st.write(response)
        st.markdown('</div>', unsafe_allow_html=True)

elif page == "Farming Assistant":
    st.header("Farming Assistant")
    user_input = st.text_area("Ask any farming-related question:", height=100)
    if st.button("Get Answer"):
        with st.spinner("AgriGPT is thinking..."):
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            response = get_gemini_response(f"As an agricultural expert, please answer the following question: {user_input}")
        st.markdown('<div class="fade-in">', unsafe_allow_html=True)
        st.subheader("AgriGPT's Response:")
        st.write(response)
        st.markdown('</div>', unsafe_allow_html=True)

elif page == "Crop Planning":
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

        with st.spinner("Generating crop recommendations..."):
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.02)
                progress_bar.progress(i + 1)
            response = get_gemini_response(prompt)
        
        st.markdown('<div class="fade-in">', unsafe_allow_html=True)
        st.subheader("Crop Recommendations:")
        st.write(response)
        st.markdown('</div>', unsafe_allow_html=True)

# Add a footer
st.markdown("---")
st.markdown("Powered by AgriGPT 🌾 | Helping farmers grow smarter")