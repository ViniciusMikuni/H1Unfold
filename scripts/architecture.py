from tensorflow.keras.activations import swish
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras.backend as K
import numpy as np

def weighted_binary_crossentropy(y_true, y_pred):
    weights = tf.gather(y_true, [1], axis=1) # event weights
    y_true = tf.gather(y_true, [0], axis=1) # actual y_true for loss


    # Clip the prediction value to prevent NaN's and Inf's
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    t_loss = -weights * ((y_true) * K.log(y_pred) +
                         (1 - y_true) * K.log(1 - y_pred))
    return K.mean(t_loss)

def FF(features,num_features,max_proj=8,min_proj=6,expand=False):
    #Gaussian features to the inputs
    freq = tf.range(start=min_proj, limit=max_proj, dtype=tf.float32)
    freq = 2.**(freq) * 2 * np.pi        
    x = tf.math.sigmoid(features)
    if expand:
        freq = tf.tile(freq[None, None,:], ( 1, 1, num_features))
        mask = features[:,:,0]!=0
    else:
        freq = tf.tile(freq[None, :], ( 1, num_features))
        mask = features[:,0]!=0
    h = tf.repeat(x, max_proj-min_proj, axis=-1)
    angle = h*freq
    h = tf.concat([tf.math.sin(angle),tf.math.cos(angle)],-1)
    return tf.concat([features,h],-1)*tf.cast(mask[...,None],tf.float32)


def DeepSetsAtt(
        num_feat,
        num_evt,
        num_heads=4,
        num_transformer= 0,
        projection_dim= 64,
        nrep = 1,
):

    #act = layers.LeakyReLU(alpha=0.01)
    act = swish
    inputs = layers.Input((None,num_feat))
    inputs_evt = layers.Input((num_evt))
    net_trials = []

    for _ in range(nrep):
        inputs_FF = FF(inputs,num_feat,expand=True)
        masked_features = layers.Masking(mask_value=0.0)(inputs_FF)
        masked_features = act(layers.Dense(2*projection_dim)(masked_features))
        masked_features = act(layers.Dense(projection_dim)(masked_features))
        
        #Conditional information
        evt_embed = FF(inputs_evt,num_feat)
        evt_embed = act(layers.Dense(projection_dim)(evt_embed))
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
            x4 = layers.Dense(projection_dim,activation="gelu")(evt_tile)
            # Skip connection 2.
            encoded_patches = layers.Add()([x3, x2,x4])
        
        representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        representation = layers.GlobalAveragePooling1D()(representation)
        evt_embed = act(layers.Dense(projection_dim)(evt_embed))
        concat = layers.Add()([representation,evt_embed])
    
        representation = act(layers.Dense(128)(concat))
        representation = layers.Dropout(0.1)(representation)
        representation = act(layers.Dense(64)(representation))
        output1 = layers.Dense(1,activation='sigmoid',
                              kernel_initializer="zeros",
                              #kernel_regularizer='l2',
                              #use_bias=False
                              )(representation)
        # evt_embed = act(layers.Dense(projection_dim)(evt_embed))
        # output2 = layers.Dense(1,activation='sigmoid',
        #                        kernel_initializer="zeros")(evt_embed)
        
        net_trials.append(output1)
        outputs = tf.reduce_mean(net_trials,0) #Average over trials
    
    return  inputs,inputs_evt, outputs
