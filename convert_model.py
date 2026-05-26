from tensorflow.keras.models import load_model

model = load_model("image_classify.keras")
model.save("model.h5")