import os

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

@st.cache_resource
def load_model_cached():
    return load_model("image_classify.keras", compile=False)

model = load_model_cached()

st.header('Food Image Classification')
img = st.text_input('Enter Image name', value='Apple.jpg')
@st.cache_resource
def load_model_cached():
    return tf.keras.models.load_model(
        "image_classify.keras",
        compile=False
    )

model = load_model_cached()
# Dataset folder
folder_path = r"C:\Users\manis\OneDrive\Desktop\food_image _classifi\Dataset"

data_cat = ['Bread',
 'Dairy product',
 'Dessert',
 'Egg',
 'Fried food',
 'Meat',
 'Noodles- pasta',
 'Rice',
 'Soup',
 'Vegetable-Fruits',
 'seafood']
img_height = 180
img_width = 180

# ===================== NUTRITION DATA =====================
food_info = {

    "Bread": {
        "Calories": 265,
        "Protein": 9,
        "Carbs": 49,
        "Fat": 3.2
    },
    "Dairy product": {
        "Calories": 150,
        "Protein": 8,
        "Carbs": 12,
        "Fat": 8
    },
    "Dessert": {
        "Calories": 350,
        "Protein": 4,
        "Carbs": 50,
        "Fat": 15
    },
    "Egg": {
        "Calories": 155,
        "Protein": 13,
        "Carbs": 1.1,
        "Fat": 11
    },
    "Fried food": {
        "Calories": 300,
        "Protein": 10,
        "Carbs": 30,
        "Fat": 20
    },
    "Meat": {
        "Calories": 250,
        "Protein": 26,
        "Carbs": 0,
        "Fat": 15
    },
    "Noodles- pasta": {
        "Calories": 200,
        "Protein": 7,
        "Carbs": 42,
        "Fat": 1.5
    },
    "Rice": {
        "Calories": 130,
        "Protein": 2.7,
        "Carbs": 28,
        "Fat": 0.3
    },
    "Soup": {
        "Calories": 150,
        "Protein": 5,
        "Carbs": 20,
        "Fat": 5
    },
    "Vegetable-Fruits": {
        "Calories": 50,
        "Protein": 2,
        "Carbs": 12,
        "Fat": 0.5
    },
    "seafood": {
        "Calories": 100,
        "Protein": 20,
        "Carbs": 0,
        "Fat": 2
    }
}

# ===================== SIDEBAR SEARCH =====================
st.sidebar.title("🔍 Food Image Search")

search = st.sidebar.text_input(
    "Search food name"
).strip().lower()

if search:

    found = False

    if os.path.exists(folder_path):

        for folder in os.listdir(folder_path):

            if search in folder.lower():

                st.sidebar.success(
                    "Food Found : " + folder
                )
                path = os.path.join(folder_path, folder)

                for img_file in os.listdir(path):

                    img_path = os.path.join(path, img_file)

                    if (
                        os.path.isfile(img_path)
                        and img_file.lower().endswith(
                            (".jpg", ".jpeg", ".png", ".webp")
                        )
                    ):
                        try:
                            image = Image.open(img_path)

                            st.sidebar.image(
                                image,
                                width=150
                            )

                        except:
                            pass

                found = True
                break
    if not found:

        st.sidebar.error(
            "Food not found in dataset"
        )   
        
        
        # ===================== INPUT SECTION =====================
col1, col2 = st.columns(2)

with col1:

    uploaded_file = st.file_uploader(
        "📤 Upload Food Image",
        type=["jpg", "jpeg", "png", "webp"]
    )

with col2: 
     camera_image = st.camera_input(
        "📷 Take Photo"
    )
image = None

if uploaded_file:

    image = Image.open(uploaded_file)

elif camera_image:

    image = Image.open(camera_image)  
    
# ===================== PREDICTION =====================
if image:

    # Show Image
    st.image(image, width=250)

    # Resize Image
    img_resized = image.resize(
        (img_height, img_width)
    )

    # Convert to Array
    img_array = tf.keras.utils.img_to_array(
        img_resized
    )

    # Expand Dimension
    img_batch = tf.expand_dims(
        img_array,
        0
    )

   # Prediction
    prediction = model.predict(img_batch)

    score = tf.nn.softmax(
        prediction[0]
    )

    predicted_index = np.argmax(score)

    predicted_class = data_cat[
        predicted_index
    ]

    confidence = float(
        np.max(score)
    ) * 100
   
   # ===================== RESULT =====================
    st.subheader("🍽 Prediction Result")

    st.success(
        f"Food : {predicted_class}"
    )

    st.write(
        f"Confidence : {confidence:.2f}%"
    )

    st.progress(
        int(confidence)
    )
 
 # ===================== NUTRITION =====================
    st.subheader("🥗 Nutrition Info")

    info = food_info.get(predicted_class)

    if info:

        st.write(
            f"🔥 Calories : {info['Calories']} kcal"
        )

        st.write(
            f"💪 Protein : {info['Protein']} g"
        )

        st.write(
            f"🍞 Carbs : {info['Carbs']} g"
        )

        st.write(
            f"🧈 Fat : {info['Fat']} g"
        ) 
    else:

        st.warning(
            "No nutrition data found"
        )
        
        # ===================== TOP 3 =====================
    st.subheader("🏆 Top 3 Predictions")

    top3 = np.argsort(score)[-3:][::-1]

    for i in top3:

        st.write(
            f"{data_cat[i]} : {score[i]*100:.2f}%"
        )
        
         # ===================== CHART =====================
    st.subheader("📊 Prediction Chart")

    chart_data = {

        data_cat[i]: float(score[i])

        for i in range(len(data_cat))
    }

    st.bar_chart(chart_data)

else:

    st.info(
        "Upload image or take photo to start detection"
    )



img_load= tf.keras.utils.load_img(img, target_size=(img_height, img_width))
img_array = tf.keras.utils.img_to_array(img_load)
img_batch = tf.expand_dims(img_array, 0)


predictions = model.predict(img_batch)
score = tf.nn.softmax(predictions[0])

st.image(img, width=180)
st.write('Veg/Fruit in image is ' + data_cat[np.argmax(score)]) 
st.write('with accuracy of ' + str(np.max(score) * 100) + '%')


 
 