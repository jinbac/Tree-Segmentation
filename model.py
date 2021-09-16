import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.layers import AveragePooling2D, GlobalAveragePooling2D, UpSampling2D, Reshape, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
import tensorflow as tf

def SqueezeAndExcite(inputs, ratio=8):     #generalizing function used at whims of developer
    init = inputs
    filters = init.shape[-1]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se) #dense layers FC??????????
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    x = init * se
    return x


def ASPP(inputs):
    "image pooling"   # takes in conv_block_6 output (end of resnet50 before fc and pooling), condenses 1024 deep tensor into 256
    shape = inputs.shape
    y1 = AveragePooling2D(pool_size=(shape[1], shape[2]))(inputs)   #averages 32,32,1024 to 1,1,1024
    y1 = Conv2D(filters=256, kernel_size=1, padding="same", use_bias=False)(y1)   # 1x1 convolution down to 1,1,256
    y1 = BatchNormalization()(y1)
    y1 = Activation("relu")(y1)
    y1 = UpSampling2D((shape[1], shape[2]), interpolation="bilinear")(y1)   # upsamples to 32,32,256

    "1x1 convolution"
    y2 = Conv2D(256, 1, padding="same", use_bias=False)(inputs)
    y2 = BatchNormalization()(y2)
    y2 = Activation("relu")(y2)

    "3x3 conv dilation rate =6"
    y3 = Conv2D(256, 3, padding="same", use_bias=False, dilation_rate=6)(inputs)
    y3 = BatchNormalization()(y3)
    y3 = Activation("relu")(y3)

    "3x3 conv dilation rate =12"
    y4 = Conv2D(256, 3, padding="same", use_bias=False, dilation_rate=12)(inputs)
    y4 = BatchNormalization()(y4)
    y4 = Activation("relu")(y4)

    "3x3 conv dilation rate =18"
    y5 = Conv2D(256, 3, padding="same", use_bias=False, dilation_rate=18)(inputs)
    y5 = BatchNormalization()(y5)
    y5 = Activation("relu")(y5)


    y = Concatenate()([y1,y2,y3,y4,y5])    # stacks all the feature maps created above together to make one tensor for squeeze and excite
    y = Conv2D(256, 1, padding="same", use_bias=False)(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)

    return y

def deeplabv3_plus(shape):
    ' Input '
    inputs = Input(shape)   # Input is a keras function that instantiates a blank tensor with a shape

    '''encoder'''
    encoder = ResNet50(weights="imagenet", include_top=False, input_tensor=inputs)

    image_features = encoder.get_layer("conv4_block6_out").output   # output from end of resnet50 just before pooling and FC layers
    x_a = ASPP(image_features)   #sends output to function above to create squeeze and excite layers
    x_a = UpSampling2D((4,4), interpolation="bilinear")(x_a)

    x_b = encoder.get_layer("conv2_block2_out").output    # output from somewhere in the middle of resnet50 just after residual is added
    x_b = Conv2D(filters=48, kernel_size=1, padding="same", use_bias=False)(x_b)
    x_b = BatchNormalization()(x_b)
    x_b = Activation("relu")(x_b)

    x = Concatenate()([x_a, x_b])     # stack x_a (128,128,256) from end of resnet50 with x_b (128,128,48) from middle of resnet50
    x = SqueezeAndExcite(x)     # placed here by whims of developer, could squeeze and excite after ever activation

    x = Conv2D(filters=256, kernel_size=3, padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(filters=256, kernel_size=3, padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = SqueezeAndExcite(x)

    x = UpSampling2D((4, 4), interpolation="bilinear")(x)
    x = Conv2D(1, 1)(x)
    x = Activation("sigmoid")(x)    # last activation is sigmoid because this step is binary segmentation, OUTPUTS PREDICTED MASK

    model = Model(inputs, x)
    return model

if __name__ == "__main__":
    model = deeplabv3_plus((512, 512, 3))
    model.summary()