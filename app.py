import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from streamlit_option_menu import option_menu
import plotly.graph_objects as go
import time
import io
from tensorflow.keras.utils import plot_model
import os
import matplotlib.pyplot as plt


MODEL_PATH = 'brain_tumor_classification.h5'
CLASS_NAMES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
IMG_SIZE = (128, 128)
plot_path = "model_architecture.png"

# Custom CSS for a modern look
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 0.5em 1em;
        font-size: 16px;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    .uploadedFileName {
        font-weight: bold;
        color: #333333;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

with st.sidebar:
    selected = option_menu(
        "Navigation", ["Home", "Model Info"],
        icons=['house', 'info-circle'],
        menu_icon="cast",
        default_index=0
    )

    st.markdown("""
    ### Supported Classes:
    - Glioma
    - Meningioma
    - Pituitary
    - No Tumor
    """)

if selected == "Home":
    st.title("Brain Tumor Classifier")
    st.markdown("""
    Upload a brain MRI image (JPG/PNG), and this app will classify the type of brain tumor detected.
    """)

    uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("L")
        #image_rgb = image.convert("RGB")
        st.image(image, caption="Uploaded MRI Image", width = 300)

        img_resized = image.resize(IMG_SIZE)
        img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
        #img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        if st.button("🔍 Classify Tumor"):
            with st.spinner("Classifying..."):
                prediction = model.predict(img_array)
                predicted_class = CLASS_NAMES[np.argmax(prediction)]
                confidence = 100 * np.max(prediction)

                st.success(f"🧾 **Prediction:** {predicted_class}")
                st.info(f"📊 **Confidence:** {confidence:.2f}%")

                # probability bar chart
                fig = go.Figure([go.Bar(
                    x=CLASS_NAMES,
                    y=prediction[0],
                    text=prediction[0],
                    textposition='auto',
                    marker=dict(color='rgba(58, 85, 163, 0.6)')
                )])
                fig.update_layout(title="Class Probabilities",
                                  xaxis_title="Class",
                                  yaxis_title="Probability",
                                  template="plotly_dark")
                st.plotly_chart(fig)

elif selected == "Model Info":
    st.header("📈 Model Performance")
    st.write("""
    **Model Summary:**
    - Architecture: CNN
    - Input Size: 128x128x1

    **Accuracy Scores:**
    - Training Accuracy: **91.8%**
    - Validation Accuracy: **90.2%**
    
    This model was trained on a public brain tumor MRI dataset with data augmentation and early stopping. It is designed for experimental or research purposes only.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.write("""
            **Test Scores**
            - Accuracy:  **82.2**
            - Precision:  **83.2**
            - Recall:  **82.3**
            - F1:  **82.3**

            **NOTE: The difference of approximately 8% in accuracy might suggest in overfitting of the model on the training set,
            there is a lot of room for improvment to the model**
        """)

    with col2:
        st.subheader("Confusion Matrix")
        try:
            st.image("confusion_matrix.png", caption="Confusion Matrix")
        except Exception as e:
            st.warning(f"Could not load confusion matrix: {e}")

    # Model Summary Button with toggle
    if st.button("View Model Architecture"):
        try:
            plot_model(model, to_file=plot_path, show_shapes=True, show_layer_names=True)
            st.image(plot_path, caption="Model Architecture")
        except Exception as e:
            st.error(f"Unable to generate model architecture plot. Error: {e}")

    # Class-wise Pie chart
    st.subheader("Class Distribution in Training Dataset")
    labels = CLASS_NAMES
    values = [23.2, 23.4, 27.9, 25.5]
    fig_pie = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3)])
    fig_pie.update_layout(title="Class Distribution", template="plotly_dark")
    st.plotly_chart(fig_pie)


