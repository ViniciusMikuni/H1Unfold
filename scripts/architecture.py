from tensorflow.keras.activations import swish
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras.backend as K


def weighted_binary_crossentropy(y_true, y_pred):
    weights = tf.gather(y_true, [1], axis=1) # event weights
    y_true = tf.gather(y_true, [0], axis=1) # actual y_true for loss
    
    # Clip the prediction value to prevent NaN's and Inf's
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    t_loss = -weights * ((y_true) * K.log(y_pred) +
                         (1 - y_true) * K.log(1 - y_pred))
    return K.mean(t_loss)


def DeepSetsAtt(
        num_feat,
        num_evt,
        num_heads=4,
        num_transformer= 4,
        projection_dim= 64,
):

    #act = layers.LeakyReLU(alpha=0.01)
    act = swish
    inputs = layers.Input((None,num_feat))    
    masked_features = layers.Masking(mask_value=0.0,name='Mask')(inputs)
    masked_features = act(layers.Dense(projection_dim)(masked_features))
    
    #Conditional information
    inputs_evt = layers.Input((num_evt))    
    evt_embed = act(layers.Dense(projection_dim,activation=None)(inputs_evt))
    evt_tile = layers.Reshape((1,-1))(evt_embed)
    evt_tile = tf.tile(evt_tile,(1,tf.shape(inputs)[1],1))

    concat = layers.Concatenate(-1)([masked_features,evt_tile])
    encoded_patches = act(layers.Dense(projection_dim,activation=None)(concat))
        
    for _ in range(num_transformer):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            dropout=0.2,num_heads=num_heads, key_dim=16)(x1, x1)
        
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        
        # Layer normalization 2.

        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)                
        x3 = layers.Dense(2*projection_dim,activation="gelu")(x3)
        x3 = layers.Dense(projection_dim,activation="gelu")(x3)
        
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])
        
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.GlobalAveragePooling1D()(representation)
    
    #Add embedding information once again
    evt_embed = act(layers.Dense(128)(evt_embed))    
    concat = layers.Concatenate(-1)([representation,evt_embed])
    
    representation = act(layers.Dense(128)(concat))
    representation = layers.Dropout(0.1)(representation)
    representation = act(layers.Dense(64)(representation))
    outputs = layers.Dense(1,activation='sigmoid')(representation)
    
    return  inputs,inputs_evt, outputs
