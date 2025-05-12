import tensorflow as tf

# Load your existing model
model = tf.keras.models.load_model("model.h5")
model.save("compressed_model.h5", include_optimizer=False)



