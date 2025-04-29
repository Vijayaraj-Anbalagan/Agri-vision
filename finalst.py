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
    /* Style for tab labels - make them dark to be visible on light background */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {{
        color: #333333;
        font-weight: bold;
    }}
    /* Style for active tab - highlight with the primary color without bottom border */
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] [data-testid="stMarkdownContainer"] p {{
        color: {PRIMARY_COLOR};
        font-weight: bold;
    }}
    /* Add hover effect for better user experience */
    .stTabs [data-baseweb="tab-list"] button:hover [data-testid="stMarkdownContainer"] p {{
        color: {PRIMARY_COLOR};
    }}
    /* Enhanced card styles for results */
    .card {{
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        background-color: white;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }}
    .result-section {{
        margin-top: 10px;
        padding: 15px;
        border-radius: 8px;
        background-color: #f8f9fa;
        border-left: 5px solid {PRIMARY_COLOR};
    }}
    .confidence-high {{
        color: #2e7d32;
        font-weight: bold;
    }}
    .confidence-medium {{
        color: #ff9800;
        font-weight: bold;
    }}
    .confidence-low {{
        color: #f44336;
        font-weight: bold;
    }}
    /* Quick facts section styling */
    .quick-facts {{
        background-color: #e8f5e9;
        padding: 15px;
        border-radius: 10px;
        margin: 15px 0;
    }}
    /* Tooltip enhancements */
    div[data-baseweb="tooltip"] {{
        background-color: #424242;
        color: white;
        padding: 8px;
        border-radius: 6px;
        font-size: 14px;
    }}
    /* Better form styling */
    .stTextInput>div>div>input, .stSelectbox>div>div>select, .stTextArea>div>div>textarea {{
        border-radius: 8px;
        border: 1px solid #ccc;
        padding: 10px;
    }}
    .stTextInput>div>div>input:focus, .stSelectbox>div>div>select:focus, .stTextArea>div>div>textarea:focus {{
        border-color: {PRIMARY_COLOR};
        box-shadow: 0 0 0 2px rgba(76, 175, 80, 0.2);
    }}
    /* Image container enhancement */
    .image-container {{
        padding: 10px;
        background-color: white;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}
    /* Collapsible section styling */
    .collapsible {{
        cursor: pointer;
        background-color: #f1f8e9;
        border-left: 4px solid {PRIMARY_COLOR};
        padding: 10px;
        width: 100%;
        text-align: left;
        margin: 5px 0;
        border-radius: 5px;
        transition: 0.3s;
    }}
    .collapsible:hover {{
        background-color: #e8f5e9;
    }}
    /* Action button highlighting */
    .action-button {{
        background-color: #ff9800 !important;
        color: white !important;
    }}
    /* Loading indicator improvements */
    .stProgress > div > div > div > div {{
        background-color: {PRIMARY_COLOR};
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
    uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"], key="disease_detection", 
                                   help="Upload a clear image of the plant leaf for accurate disease detection")
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            # Enhanced image display with container styling
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            if st.button("Detect Disease", help="Click to analyze the image and detect plant diseases"):
                with st.spinner("Analyzing leaf image..."):
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
                        
                        # Determine confidence level indicator
                        if confidence >= 90:
                            confidence_class = "confidence-high"
                            confidence_level = "High"
                        elif confidence >= 70:
                            confidence_class = "confidence-medium"
                            confidence_level = "Medium"
                        else:
                            confidence_class = "confidence-low"
                            confidence_level = "Low"
                        
                        st.success("Analysis Complete!")
                        
                        # Display results in a well-formatted card
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        
                        # Show confidence warning for low confidence results
                        if confidence < 70:
                            st.warning("⚠️ Low confidence detection. Results may not be fully reliable. Consider uploading a clearer image or consulting with an agricultural expert.")
                        
                        # Display disease information in formatted sections
                        st.markdown(f"### 📊 Diagnostic Results")
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"**Detected Condition:** {disease_name}")
                        with col2:
                            st.markdown(f"**Confidence:** <span class='{confidence_class}'>{confidence:.1f}% ({confidence_level})</span>", unsafe_allow_html=True)
                            
                        # Handle healthy vs. disease cases differently
                        if "Healthy" in predicted_label:
                            st.markdown(f"""
                            <div class='result-section'>
                                <h4>✅ Health Assessment</h4>
                                <p>Your plant appears to be healthy. No signs of common diseases detected.</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown(f"""
                            <div class='result-section'>
                                <h4>🌱 Maintenance Recommendations</h4>
                                <p>Continue with regular care:</p>
                                <ul>
                                    <li>Maintain proper watering schedule</li>
                                    <li>Ensure adequate sunlight exposure</li>
                                    <li>Monitor for any changes in leaf color or texture</li>
                                    <li>Apply balanced fertilizer as needed for the specific crop</li>
                                </ul>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            # For disease cases, show more detailed info with collapsible sections
                            crop_type = predicted_label.split('___')[0].replace('_', ' ')
                            
                            # Quick facts about the disease
                            st.markdown("""
                            <div class='quick-facts'>
                                <h4>📋 Quick Facts</h4>
                                <ul>
                                    <li><strong>Crop Type:</strong> {}</li>
                                    <li><strong>Detection Reliability:</strong> {}</li>
                                    <li><strong>Recommended Action:</strong> {}</li>
                                </ul>
                            </div>
                            """.format(
                                crop_type, 
                                confidence_level, 
                                "Immediate attention required" if confidence > 85 else "Further investigation suggested"
                            ), unsafe_allow_html=True)
                            
                            # Add a "Get Treatment Advice" button with improved styling
                            st.markdown("<div style='margin: 20px 0;'>", unsafe_allow_html=True)
                            if st.button(f"🔍 Get Detailed Treatment Advice", 
                                       key="disease_button",
                                       help="Click to get comprehensive treatment and prevention guidance",
                                       use_container_width=True):
                                switch_to_farming_assistant(disease_name)
                            
                            # Add "See Similar Cases" button
                            if st.button(f"👁️ View Similar Cases", 
                                       key="similar_cases",
                                       help="View similar disease cases and treatment outcomes",
                                       use_container_width=True):
                                st.info("This feature will be available in the next update.")
                            st.markdown("</div>", unsafe_allow_html=True)
                            
                            # Next steps action plan
                            st.markdown(f"""
                            <div class='result-section'>
                                <h4>🚩 Recommended Next Steps</h4>
                                <ol>
                                    <li>Click "Get Detailed Treatment Advice" for specific guidance</li>
                                    <li>Isolate affected plants if possible to prevent spread</li>
                                    <li>Document the affected areas for monitoring progress</li>
                                    <li>Consider consulting with a local agricultural extension service</li>
                                </ol>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                    except FileNotFoundError:
                        st.error("Model weights file (model1.h5) not found. Please ensure the model file is available.")
                        
        except Exception as e:
            st.error(f"An error occurred during processing: {str(e)}. Please try a different image.")
    else:
        # Show helpful guidance when no image is uploaded
        st.info("📸 Please upload a clear, well-lit image of a plant leaf to detect diseases. For best results:")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            - Ensure good lighting
            - Focus on affected leaves
            - Include multiple leaves if possible
            """)
        with col2:
            st.markdown("""
            - Avoid shadows or glare
            - Use a solid background if possible
            - Capture the entire leaf when possible
            """)
        
        # Display supported crops
        st.markdown("### 🌿 Supported Crops")
        st.markdown("""
        Currently, our system can detect diseases in:
        - Apple
        - Corn (Maize)
        - Grape
        - Potato
        - Tomato
        """)

# Crop Analysis Tab
with tab2:
    st.header("Crop Analysis")
    
    # Introduction text for better guidance
    st.markdown("""
    Upload an image of your field or crop for comprehensive analysis. Our AI system will identify the crop, 
    assess its health, identify issues, and provide actionable recommendations.
    """)
    
    # Enhanced file uploader with better guidance
    uploaded_file = st.file_uploader(
        "Upload crop or field image",
        type=["jpg", "jpeg", "png"],
        key="crop_analysis",
        help="For best results, upload a clear image showing the entire crop or field area"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        # Enhanced image display
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Add location context for better analysis
        with st.expander("🌍 Add Location Context (Optional)"):
            st.markdown("Adding location information helps provide more relevant recommendations.")
            col1, col2 = st.columns(2)
            with col1:
                region = st.selectbox(
                    "Region of India",
                    ["Northern India", "Southern India", "Eastern India", "Western India", "Central India", "North-East India", "Not in India"],
                    help="Select the region where this crop is grown"
                )
            with col2:
                growing_season = st.selectbox(
                    "Current Growing Season",
                    ["Kharif (Monsoon)", "Rabi (Winter)", "Zaid (Summer)", "Year-round"],
                    help="Select the current growing season for better context"
                )
        
        # Enhanced analysis button with better wording
        analyze_col1, analyze_col2 = st.columns([3, 1])
        with analyze_col1:
            analyze = st.button(
                "🔍 Analyze Crop & Field", 
                use_container_width=True,
                help="Click to start AI-powered analysis of your crop or field"
            )
        
        # Quick options for analysis focus
        with analyze_col2:
            analysis_focus = st.selectbox(
                "Focus on",
                ["All aspects", "Health only", "Yield estimate", "Issues only"],
                help="Select what aspects to focus on in the analysis"
            )

        if analyze:
            # Enhanced prompt with more specific instructions based on user selections
            input_prompt = f"""
            You are an expert agricultural analyst examining the crop or field in the image.
            
            The image is from {region if region != "Not in India" else "an unspecified location"}, and it's currently the {growing_season} season.
            
            The farmer is particularly interested in {analysis_focus.lower()}.
            
            Provide a detailed analysis including:
            
            1. Identification of the crop (if visible) with scientific name and common varieties
            2. Current growth stage assessment
            3. Assessment of crop health on a scale of 1-10 with justification
            4. Identification of any visible issues (pests, diseases, nutrient deficiencies) with confidence level
            5. Specific actionable suggestions for improvement or treatment that can be implemented immediately
            6. Estimated crop yield based on visible conditions with comparison to typical yields
            7. Long-term recommendations for improving cultivation practices
            
            Format your response with clear headings for each section and prioritize practical, actionable advice.
            For treatments, mention both organic/traditional options and modern approaches where appropriate.
            Include specific product types that would help but avoid mentioning specific brand names.
            """

            with st.spinner("Analyzing crop and field conditions..."):
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                    
                try:
                    image_data = input_image_setup(uploaded_file)
                    response = get_gemini_response(input_prompt, image_data)
                    
                    # Success message with actionability cue
                    st.success("✅ Analysis Complete! Review the insights below and take action to improve your crop.")
                    
                    # Display response in a well-formatted card
                    st.markdown('<div class="card fade-in">', unsafe_allow_html=True)
                    st.markdown(response)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Extract crop name from response for showing additional resources
                    # This is a simple extraction - in a real implementation would need more robust parsing
                    first_lines = response.split('\n')[:10]
                    crop_name = ""
                    for line in first_lines:
                        if "crop" in line.lower() and ":" in line:
                            crop_name = line.split(":")[-1].strip()
                            break
                    
                    if crop_name:
                        # Show additional resources tailored to the detected crop
                        st.markdown("### 📚 Additional Resources")
                        st.markdown('<div class="result-section">', unsafe_allow_html=True)
                        st.markdown(f"Based on your {crop_name} crop, these resources might help:")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("- Best practices for cultivation")
                            st.markdown("- Common diseases and treatments")
                            st.markdown("- Harvesting guidelines")
                        with col2:
                            st.markdown("- Post-harvest handling")
                            st.markdown("- Market connection opportunities")
                            st.markdown("- Government schemes for this crop")
                        
                        if st.button("🔗 Access Resources", key="resources"):
                            st.info("Detailed resources will be available in the next update.")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Action plan and summary
                    st.markdown("### 🚜 Quick Action Plan")
                    st.markdown('<div class="result-section">', unsafe_allow_html=True)
                    
                    # Get a condensed action plan from the AI
                    action_prompt = f"Based on the following crop analysis, provide a BRIEF 3-5 point action plan with ONLY the most urgent and high-impact actions the farmer should take. Keep each point under 15 words. Analysis: {response[:1000]}"
                    action_plan = get_gemini_response(action_prompt)
                    
                    st.markdown(action_plan)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}. Please try again with a different image.")
    else:
        # Show example images and guidance when no image is uploaded
        st.info("📸 Upload a clear image of your crop or field for AI-powered analysis")
        
        st.markdown("### How to take good field images:")
        cols = st.columns(3)
        with cols[0]:
            st.markdown("- **Distance**: Capture both close-ups and wider views")
            st.markdown("- **Lighting**: Take photos in good natural light")
        with cols[1]:
            st.markdown("- **Quantity**: Multiple images give better results")
            st.markdown("- **Focus**: Ensure images are clear and not blurry")
        with cols[2]:
            st.markdown("- **Context**: Include typical and problem areas")
            st.markdown("- **Timing**: Morning light often gives best results")

# Farming Assistant Tab
with tab3:
    st.header("Farming Assistant")
    
    # Add session state for saving responses
    if 'saved_responses' not in st.session_state:
        st.session_state.saved_responses = []
    
    # Create predefined question templates
    st.subheader("🧠 Common Agricultural Queries")
    
    # Organize common questions by category using expanders
    with st.expander("Disease & Pest Management"):
        template_questions = [
            "What are organic treatments for tomato early blight?",
            "How to prevent apple black rot recurrence next season?",
            "Best practices for controlling aphids without chemicals?",
            "How to identify and treat wheat rust?",
            "Integrated pest management for rice crops?"
        ]
        
        cols = st.columns(2)
        for i, question in enumerate(template_questions):
            with cols[i % 2]:
                if st.button(question, key=f"q_disease_{i}"):
                    st.session_state.template_question = question
                    st.rerun()
    
    with st.expander("Soil Health & Fertilization"):
        template_questions = [
            "How to improve clay soil for vegetable gardening?",
            "Natural ways to enhance soil fertility?",
            "Best fertilizers for organic rice cultivation?",
            "How to correct nitrogen deficiency in corn?",
            "Soil preparation techniques for wheat in sandy soil?"
        ]
        
        cols = st.columns(2)
        for i, question in enumerate(template_questions):
            with cols[i % 2]:
                if st.button(question, key=f"q_soil_{i}"):
                    st.session_state.template_question = question
                    st.rerun()
    
    with st.expander("Water & Irrigation"):
        template_questions = [
            "Efficient irrigation methods for water conservation?",
            "Signs of overwatering in tomato plants?",
            "How to set up drip irrigation for small farms?",
            "Rainwater harvesting techniques for agriculture?",
            "Water management during drought conditions?"
        ]
        
        cols = st.columns(2)
        for i, question in enumerate(template_questions):
            with cols[i % 2]:
                if st.button(question, key=f"q_water_{i}"):
                    st.session_state.template_question = question
                    st.rerun()
    
    st.markdown("---")
    st.subheader("✍️ Ask Your Question")
    
    # Set default question if disease info is available or template question selected
    if st.session_state.disease_info:
        default_question = f"What is the best treatment and prevention method for {st.session_state.disease_info}? Please provide detailed steps and organic/chemical options."
    elif 'template_question' in st.session_state and st.session_state.template_question:
        default_question = st.session_state.template_question
        # Clear the template question after using it
        st.session_state.template_question = None
    else:
        default_question = ""
    
    user_input = st.text_area("Enter your farming question:", 
                             value=default_question,
                             height=100,
                             placeholder="Example: What are the best practices for organic rice cultivation?",
                             help="Be specific about your crop, region, or particular challenge for more relevant advice")
    
    col1, col2 = st.columns([4, 1])
    with col1:
        submit = st.button("🔍 Get Expert Advice", use_container_width=True, help="Get AI-powered agricultural expert advice")
    
    # Add easy clear button
    with col2:
        if st.button("🔄 Clear", use_container_width=True):
            user_input = ""
            st.rerun()
    
    if submit and user_input:
        # Improved prompt for better quality responses
        enhanced_prompt = f"""
        As an agricultural expert specializing in Indian farming methods, please answer the following question comprehensively:
        
        "{user_input}"
        
        Structure your response in the following format:
        
        1. Direct answer to the question
        2. Detailed explanation with scientific backing when relevant
        3. Step-by-step implementation guidance
        4. Both traditional and modern approach options
        5. Organic/natural solutions first, followed by conventional options if necessary
        6. Region-specific considerations for Indian agriculture
        7. Additional resources or references if applicable
        
        Prioritize sustainable, practical advice that can be implemented with resources commonly available in rural India.
        """
        
        with st.spinner("Consulting agricultural knowledge base..."):
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            response = get_gemini_response(enhanced_prompt)
        
        # Display response in a well-formatted card
        st.markdown('<div class="card fade-in">', unsafe_allow_html=True)
        st.markdown("### 📝 Expert Agricultural Advice")
        st.markdown(response)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Add action buttons for the response
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Save Response", key="save_response"):
                if len(st.session_state.saved_responses) >= 5:
                    st.session_state.saved_responses.pop(0)  # Remove oldest
                st.session_state.saved_responses.append({
                    "question": user_input,
                    "answer": response,
                    "date": time.strftime("%Y-%m-%d %H:%M")
                })
                st.success("Response saved! Access your saved responses below.")
        
        with col2:
            if st.button("📥 Download as PDF", key="download_pdf"):
                st.info("PDF download feature will be available in the next update.")
        
        # Generate related questions based on the current query and response
        st.markdown("### 🔄 Related Questions")
        related_prompt = f"Based on this farming question: '{user_input}', suggest 3 related follow-up questions that the farmer might want to ask next. Return ONLY the questions as a numbered list without any other text."
        related_questions = get_gemini_response(related_prompt).strip()
        
        # Display related questions as clickable buttons
        st.markdown('<div class="result-section">', unsafe_allow_html=True)
        lines = related_questions.split('\n')
        for line in lines:
            # Remove numbers and any leading symbols
            clean_question = line.strip()
            if clean_question:
                # Extract just the question text, removing any numbering
                if '.' in clean_question[:5]:  # If it starts with numbering like "1. "
                    clean_question = clean_question.split('.', 1)[1].strip()
                
                if st.button(f"🔍 {clean_question}", key=f"related_{hash(clean_question)}"):
                    st.session_state.template_question = clean_question
                    st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Reset disease info after displaying the response
        st.session_state.disease_info = None
    
    # Display saved responses if any
    if st.session_state.saved_responses:
        with st.expander("📚 Your Saved Responses"):
            for i, saved in enumerate(reversed(st.session_state.saved_responses)):
                st.markdown(f"**Question {i+1} ({saved['date']}):** {saved['question']}")
                st.markdown('<div class="result-section" style="margin-bottom: 15px;">', unsafe_allow_html=True)
                st.markdown(saved['answer'][:300] + "..." if len(saved['answer']) > 300 else saved['answer'])
                if st.button(f"View Full Answer", key=f"view_saved_{i}"):
                    st.session_state.template_question = saved['question']
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

# Crop Planning Tab
with tab4:
    st.header("Crop Planning Assistant")
    
    # Initialize or get session state for selected location
    if 'specific_region' not in st.session_state:
        st.session_state.specific_region = None
    
    # Add a Indian region selector before the form
    st.subheader("Select Your Region in India")
    
    col1, col2 = st.columns(2)
    with col1:
        region_category = st.selectbox(
            "Region Category",
            ["Northern India", "Southern India", "Eastern India", "Western India", "Central India", "North-East India"]
        )
    
    # Define specific locations based on the selected region
    region_locations = {
        "Northern India": ["Delhi", "Chandigarh", "Jammu", "Srinagar", "Shimla", "Dehradun", "Lucknow", "Jaipur"],
        "Southern India": ["Chennai", "Bangalore", "Hyderabad", "Trivandrum", "Kochi", "Coorg", "Madurai", "Mysore"],
        "Eastern India": ["Kolkata", "Bhubaneswar", "Patna", "Ranchi", "Cuttack"],
        "Western India": ["Mumbai", "Ahmedabad", "Pune", "Surat", "Vadodara", "Rajkot", "Nagpur"],
        "Central India": ["Bhopal", "Indore", "Jabalpur", "Raipur", "Nagpur"],
        "North-East India": ["Guwahati", "Shillong", "Imphal", "Agartala", "Itanagar", "Aizawl", "Kohima"]
    }
    
    with col2:
        specific_location = st.selectbox(
            "Specific Location", 
            region_locations[region_category],
            help="Select the location closest to your farm"
        )
        st.session_state.specific_region = specific_location
    
    # Add regional climate information based on selected location
    climate_info = {
        "Northern India": "Generally hot summers and cold winters. Monsoon from July to September.",
        "Southern India": "Hot and humid with moderate temperature variations. Southwest monsoon from June to September.",
        "Eastern India": "Hot and humid summers with moderate winters. Heavy rainfall during monsoon.",
        "Western India": "Tropical climate with hot summers and mild winters. Heavy rainfall in coastal regions.",
        "Central India": "Hot summers and mild winters with moderate rainfall.",
        "North-East India": "Humid subtropical climate with heavy rainfall."
    }
    
    st.info(f"**Regional Climate**: {climate_info[region_category]}")
    
    with st.form("crop_planning_form"):
        st.subheader("Enter Your Farm Details")
        
        # Location now includes the preselected region
        location_placeholder = f"{specific_location}, India"
        location = st.text_input("Location", 
                                value=location_placeholder,
                                placeholder="e.g., Village name, District, State",
                                help="Please provide more specific details about your farm location")
        
        # Additional farm details
        soil_type = st.selectbox(
            "Soil Type", 
            ["Clay", "Sandy", "Loamy", "Silt", "Peat", "Chalky", "Black Cotton", "Red", "Alluvial"],
            help="Select the type of soil in your farm"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            farm_size = st.number_input(
                "Farm Size (in acres)", 
                min_value=0.1, 
                step=0.1, 
                value=1.0,
                help="Enter the total area of your farm in acres"
            )
        
        with col2:
            irrigation_source = st.selectbox(
                "Irrigation Source",
                ["Well", "Canal", "River", "Rainwater", "Drip Irrigation", "Sprinkler", "Pond", "None"],
                help="Select your primary source of irrigation"
            )
        
        col1, col2 = st.columns(2)
        with col1:
            season = st.selectbox(
                "Planting Season", 
                ["Kharif (Monsoon)", "Rabi (Winter)", "Zaid (Summer)", "Year-round"],
                help="Select the season when you plan to cultivate"
            )
        
        with col2:
            water_availability = st.slider(
                "Water Availability", 
                1, 10, 5,
                help="Rate the water availability in your area (1: Very Low, 10: Abundant)"
            )
        
        additional_info = st.text_area(
            "Additional Information (Optional)",
            placeholder="Enter any additional information about your farm, specific requirements, or constraints...",
            help="Any other details that might help in providing better crop recommendations"
        )
        
        submit_button = st.form_submit_button("Generate Crop Recommendations")

    if submit_button:
        # Enhanced prompt with region-specific information
        prompt = f"""
        As an agricultural expert familiar with Indian farming conditions, provide crop recommendations based on the following farm details:
        
        Location: {location}
        Region of India: {region_category} - {specific_location}
        Regional Climate: {climate_info[region_category]}
        Soil Type: {soil_type}
        Farm Size: {farm_size} acres
        Irrigation Source: {irrigation_source}
        Planting Season: {season}
        Water Availability: {water_availability}/10
        
        Additional Information: {additional_info if additional_info else "None provided"}
        
        Please provide detailed recommendations specifically tailored to farming in {specific_location}, including:
        
        1. Top 3-5 suitable crops for these conditions and location
        2. Optimal planting and harvesting times for each crop considering local climate patterns
        3. Estimated yield per acre for each crop under typical conditions in {specific_location}
        4. Expected Return on Investment (ROI) for each recommended crop with approximate costs and revenue estimates
        5. Companion planting suggestions to maximize land use and improve pest management naturally
        6. Region-specific farming practices recommended for {region_category}
        7. Potential challenges specific to farming in {specific_location} and how to mitigate them
        8. Suitable crop rotations for soil health maintenance
        9. Local market potential and economic viability for recommended crops
        
        Format your response with clear headings for each section. Present ROI information in a simple table format.
        Please provide practical, actionable advice that considers the specific agricultural conditions in {specific_location}, {region_category}.
        """

        with st.spinner("Generating region-specific crop recommendations..."):
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.02)
                progress_bar.progress(i + 1)
            response = get_gemini_response(prompt)
        
        # Display the response in a better formatted way
        st.success(f"📝 Crop Recommendations for {specific_location}, {region_category}")
        
        # Display response in a well-formatted card
        st.markdown('<div class="card fade-in">', unsafe_allow_html=True)
        st.markdown(response)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Add ROI calculator button - this would be implemented as a feature in the future
        st.markdown("### 💰 Detailed ROI Calculator")
        st.markdown('<div class="result-section">', unsafe_allow_html=True)
        st.markdown("""
        Want to calculate detailed ROI for your specific farm conditions? Use our advanced calculator to plan your finances for the upcoming season.
        """)
        if st.button("Open ROI Calculator", key="roi_calc"):
            st.info("The detailed ROI calculator will be available in the next update.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Add crop rotation planner
        st.markdown("### 🔄 Crop Rotation Planner")
        st.markdown('<div class="result-section">', unsafe_allow_html=True)
        st.markdown(f"""
        For {specific_location}, planning a multi-year crop rotation is essential for soil health and pest management.
        Our planner helps you visualize and plan rotations over several seasons.
        """)
        if st.button("Plan Crop Rotation", key="rotation_planner"):
            st.info("The crop rotation planner will be available in the next update.")
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("Powered by AgriVision 🌾 | Helping farmers grow smarter")
