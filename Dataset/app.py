import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow import keras
from PIL import Image

# Page Config
st.set_page_config(page_title="AI Food System", page_icon="🍔", layout="wide")

# ===================== TITLE =====================
st.title("🍔 MEGA AI Food Recognition System")
st.write("Upload food image or take photo to detect food with nutrition info")

# ===================== MODEL =====================
model = keras.models.load_model("image_classify.keras")

data_cat = [
'Bread','Dairy product','Dessert','Egg','Fried food',
'Meat','Noodles- pasta','Rice','Soup','Vegetable-Fruits','seafood'
]

img_height = 256
img_width = 256

# ===================== NUTRITION =====================
food_info = {
"Bread":{"Calories":"265 kcal","Protein":"9g","Carbs":"49g","Fat":"3g"},
"Dairy product":{"Calories":"150 kcal","Protein":"8g","Carbs":"12g","Fat":"8g"},
"Dessert":{"Calories":"350 kcal","Protein":"4g","Carbs":"50g","Fat":"15g"},
"Egg":{"Calories":"155 kcal","Protein":"13g","Carbs":"1g","Fat":"11g"},
"Fried food":{"Calories":"320 kcal","Protein":"15g","Carbs":"20g","Fat":"22g"},
"Meat":{"Calories":"250 kcal","Protein":"26g","Carbs":"0g","Fat":"15g"},
"Noodles- pasta":{"Calories":"158 kcal","Protein":"6g","Carbs":"31g","Fat":"1g"},
"Rice":{"Calories":"130 kcal","Protein":"2g","Carbs":"28g","Fat":"0g"},
"Soup":{"Calories":"80 kcal","Protein":"3g","Carbs":"10g","Fat":"2g"},
"Vegetable-Fruits":{"Calories":"90 kcal","Protein":"2g","Carbs":"22g","Fat":"0g"},
"seafood":{"Calories":"206 kcal","Protein":"22g","Carbs":"0g","Fat":"12g"}
}

# ===================== SIDEBAR SEARCH (FIXED) =====================
import os
from PIL import Image

st.sidebar.title("🔍 Food Image Search")

search = st.sidebar.text_input("Search food name").strip().lower()

folder_path = "dataset"

if search:

    found = False

    for folder in os.listdir(folder_path):

        # match folder name
        if search in folder.lower():

            st.sidebar.success("Food Found: " + folder)

            path = os.path.join(folder_path, folder)

            for img in os.listdir(path):

                img_file = Image.open(os.path.join(path, img))
                st.sidebar.image(img_file, width=150)

            found = True
            break

    if not found:
        st.sidebar.error("❌ Food not found in dataset folder")

# ===================== INPUT =====================
col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("📤 Upload Food Image", type=["jpg","png","jpeg"])

with col2:
    camera_image = st.camera_input("📷 Take Photo")

image = None

if uploaded_file:
    image = Image.open(uploaded_file)

elif camera_image:
    image = Image.open(camera_image)

# ===================== PREDICTION =====================
if image:

    st.image(image, width=300)

    img = image.resize((img_height, img_width))
    img_array = keras.utils.img_to_array(img)
    img_batch = tf.expand_dims(img_array, 0)

    prediction = model.predict(img_batch)
    score = tf.nn.softmax(prediction[0])

    predicted_index = np.argmax(score)
    predicted_class = data_cat[predicted_index]
    confidence = float(np.max(score)) * 100

    st.subheader("🍽 Prediction Result")
    st.success(f"Food : {predicted_class}")
    st.write(f"Confidence : {confidence:.2f}%")

    st.progress(int(confidence))

    # Nutrition (SAFE)
    st.subheader("🥗 Nutrition Info")

    info = food_info.get(predicted_class)

    if info:
        st.write("Calories :", info["Calories"])
        st.write("Protein :", info["Protein"])
        st.write("Carbs :", info["Carbs"])
        st.write("Fat :", info["Fat"])
    else:
        st.warning("No nutrition data found")

    # Top 3
    st.subheader("Top 3 Predictions")

    top3 = np.argsort(score)[-3:][::-1]
    for i in top3:
        st.write(f"{data_cat[i]} : {score[i]*100:.2f}%")

    # Chart
    st.subheader("Prediction Chart")

    chart_data = {data_cat[i]: float(score[i]) for i in range(len(data_cat))}
    st.bar_chart(chart_data)

else:
    st.info("Upload image or take photo to start detection")