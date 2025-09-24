import streamlit as st
import pickle
import numpy as np
import os

# Set page configuration
st.set_page_config(
    page_title="Shoe Size Prediction",
    page_icon="👟",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }

    .title-container {
        text-align: center;
        margin-bottom: 3rem;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }

    .title-container h1 {
        margin: 0;
        font-size: 3rem;
        font-weight: 700;
    }

    .subtitle {
        font-size: 1.2rem;
        margin-top: 1rem;
        opacity: 0.9;
    }

    .input-container {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        margin: 2rem 0;
        border: 1px solid #e0e0e0;
    }

    .prediction-container {
        text-align: center;
        padding: 2rem;
        margin: 2rem 0;
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(76, 175, 80, 0.3);
        animation: fadeIn 0.5s ease-in;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .prediction-container h2 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }

    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 1rem 2rem;
        font-size: 1.2rem;
        font-weight: 600;
        margin-top: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }

    .info-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #17a2b8;
        margin: 1rem 0;
    }

    .footer {
        text-align: center;
        padding: 2rem 0;
        color: #666;
        border-top: 1px solid #e0e0e0;
        margin-top: 3rem;
    }

    .stNumberInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        padding: 0.5rem;
    }

    .stSelectbox > div > div > div {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Load the trained model and label encoder with error handling"""
    try:
        # Check if assets directory exists
        if not os.path.exists('assets'):
            st.error("❌ Assets folder not found! Please create an 'assets/' folder with the required model files.")
            st.stop()

        # Load the regression model
        model_path = 'assets/model.pkl'
        if not os.path.exists(model_path):
            st.error("❌ model.pkl not found in assets/ folder!")
            st.stop()

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        # Load the label encoder
        encoder_path = 'assets/label_encoder.pkl'
        if not os.path.exists(encoder_path):
            st.error("❌ label_encoder.pkl not found in assets/ folder!")
            st.stop()

        with open(encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)

        return model, label_encoder

    except Exception as e:
        st.error(f"❌ Error loading model files: {str(e)}")
        st.error("Please ensure both 'model.pkl' and 'label_encoder.pkl' are in the 'assets/' folder")
        st.stop()


def predict_shoe_size(age, height, gender, model, label_encoder):
    """Make shoe size prediction with proper error handling"""
    try:
        # Encode gender using the label encoder
        gender_encoded = label_encoder.transform([gender])[0]

        # Create input array in the format [Age, Height, Gender_encoded]
        input_data = np.array([[age, height, gender_encoded]])

        # Make prediction
        prediction = model.predict(input_data)[0]

        # Round to 2 decimal places
        return round(prediction, 2)

    except ValueError as e:
        st.error(f"❌ Invalid gender value: {gender}. Please select from available options.")
        return None
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        return None


def main():
    # Title and header
    st.markdown("""
    <div class="title-container">
        <h1>👟 Shoe Size Prediction</h1>
        <div class="subtitle">Enter your details below to estimate your shoe size.</div>
    </div>
    """, unsafe_allow_html=True)

    # Load models
    model, label_encoder = load_models()

    # Input section
    st.markdown('<div class="input-container">', unsafe_allow_html=True)

    # Create columns for better layout
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📊 Personal Details")

        # Age input with validation
        age = st.number_input(
            "Age (years)",
            min_value=1,
            max_value=100,
            value=18,
            step=1,
            help="Enter your age between 1 and 100 years"
        )

        # Height input with validation
        height = st.number_input(
            "Height (cm)",
            min_value=50,
            max_value=250,
            value=170,
            step=1,
            help="Enter your height in centimeters (50-250 cm)"
        )

    with col2:
        st.markdown("### 👤 Gender Information")

        # Get available gender options from label encoder
        try:
            gender_options = list(label_encoder.classes_)
            gender = st.selectbox(
                "Gender",
                options=gender_options,
                help="Select your gender from the available options"
            )
        except Exception as e:
            st.error(f"Error loading gender options: {str(e)}")
            gender = "Unknown"

        # Display current selections
        st.markdown("---")
        st.markdown("**Current Selection:**")
        st.write(f"👶 Age: {age} years")
        st.write(f"📏 Height: {height} cm")
        st.write(f"👤 Gender: {gender}")

    st.markdown('</div>', unsafe_allow_html=True)

    # Prediction button
    predict_button = st.button("🔮 Predict Shoe Size")

    if predict_button:
        # Validate inputs
        if age < 1 or age > 100:
            st.error("❌ Please enter an age between 1 and 100 years")
        elif height < 50 or height > 250:
            st.error("❌ Please enter a height between 50 and 250 cm")
        elif not gender:
            st.error("❌ Please select a gender")
        else:
            # Show loading spinner
            with st.spinner("🔄 Calculating your shoe size..."):
                # Make prediction
                predicted_size = predict_shoe_size(age, height, gender, model, label_encoder)

                if predicted_size is not None:
                    # Display result
                    st.markdown(f"""
                    <div class="prediction-container">
                        <h2>✅ Predicted Shoe Size: {predicted_size}</h2>
                        <p style="font-size: 1.1rem; margin-top: 1rem; opacity: 0.9;">
                            Based on your age ({age} years), height ({height} cm), and gender ({gender})
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Additional information
                    st.markdown("""
                    <div class="info-box">
                        <h4>💡 Important Note</h4>
                        <p>This prediction is based on statistical modeling and may vary from actual shoe sizes. 
                        Shoe sizes can differ between brands and styles. Always try on shoes before purchasing for the best fit!</p>
                    </div>
                    """, unsafe_allow_html=True)

    # Divider
    st.markdown("---")

    # Footer
    st.markdown("""
    <div class="footer">
        <p>🤖 <strong>Powered by Machine Learning</strong> | Built with ❤️ using Streamlit</p>
        <p style="font-size: 0.9rem; margin-top: 0.5rem;">
            Accurate predictions through advanced regression modeling
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()