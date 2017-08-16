from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Model

def create_model(input_shape, classes):
    img_input = Input(shape=input_shape)

    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(32, activation='relu', name='fc1')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)

    model = Model(inputs=img_input, outputs=x)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model


def create_model_2(input_shape, classes):
    img_input = Input(input_shape, name='input')
    x = Dropout(0.3, name='input_drop')(img_input)
    x = Conv2D(64, 3, activation='relu', padding='same', name='conv_1')(x)
    x = Dropout(0.75, name='conv_1_drop')(x)
    x = Conv2D(128, 3, activation='relu', padding='same', name='conv_2')(x)
    x = Dropout(0.75, name='conv_2_drop')(x)
    x = Conv2D(256, 3, activation='relu', padding='same', name='conv_3')(x)
    x = Dropout(0.75, name='conv_3_drop')(x)
    x = Conv2D(512, 3, activation='relu', padding='same', name='conv_4')(x)
    x = Dropout(0.75, name='conv_4_drop')(x)
    x = Conv2D(512, 3, activation='relu', padding='same', name='conv_5')(x)
    x = Dropout(0.75, name='conv_5_drop')(x)
    x = Flatten(name='flatten')(x)
    x = Dense(128, activation='relu', name='dense_1')(x)
    x = Dropout(0.75, name='dense_1_drop')(x)
    out = Dense(classes, activation='softmax', name='output')(x)
    model = Model(inputs=img_input, outputs=out)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
