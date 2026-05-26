from tensorflow.keras.models import load_model

model = load_model("image_classify.keras", compile=False)

model.save("new_model.h5")