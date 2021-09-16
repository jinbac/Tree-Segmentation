import tensorflow as tf

input_shape = (224, 224, 3)
base_model = tf.keras.applications.ResNet50(input_shape=input_shape, include_top=False, weights='imagenet')

# Get the output of conv4
conv4_block6_out = base_model.get_layer('conv4_block6_out').output
x = tf.keras.layers.Softmax()(conv4_block6_out)
model = tf.keras.Model(base_model.input, x)
model.summary()