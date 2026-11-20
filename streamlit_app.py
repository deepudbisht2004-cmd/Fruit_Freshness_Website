
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Fruit Freshness Detection",
    page_icon="🍎",
    layout="centered"
)

st.title("🍎 Fruit Freshness Detection")
st.write("Upload a fruit image to check its freshness using CNN.")

@st.cache_resource
def load_cnn_model():
    return tf.keras.models.load_model("Fruit_Freshness_CNN.keras")

try:
    model = load_cnn_model()
    st.success("CNN model loaded successfully!")
except Exception as e:
    st.error(f"Unable to load model: {e}")
    st.stop()

# Use the exact class order printed during model training
class_names = ["Fresh", "Mild", "Rotten"]

uploaded_file = st.file_uploader(
    "Choose a fruit image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Fruit", width=300)

    if st.button("Detect Freshness"):
        image = image.resize((128, 128))
        image_array = np.array(image, dtype=np.float32)
        image_array = np.expand_dims(image_array, axis=0)

        predictions = model.predict(image_array)
        index = int(np.argmax(predictions[0]))

        result = class_names[index]
        confidence = float(predictions[0][index]) * 100

        st.subheader("Detection Result")

        if result == "Fresh":
            st.success("🍏 Fresh Fruit")
        elif result == "Mild":
            st.warning("🍎 Mildly Spoiled Fruit")
        else:
            st.error("🍂 Rotten Fruit")

        st.write(f"**Prediction:** {result}")
        st.write(f"**Model confidence:** {confidence:.2f}%")

st.divider()
st.caption("Powered by CNN and Deep Learning")
st.caption("This model estimates visible freshness; it cannot detect foodborne pathogens or prove food is safe.")
