from tensorflow.keras.activations import swish
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras.backend as K
import numpy as np
from tensorflow import keras


def weighted_binary_crossentropy(y_true, y_pred):
    weights = tf.cast(tf.gather(y_true, [1], axis=1),tf.float32) # event weights
    y_true = tf.cast(tf.gather(y_true, [0], axis=1),tf.float32) # actual y_true for loss

    
    # Clip the prediction value to prevent NaN's and Inf's
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    t_loss = -weights * ((y_true) * K.log(y_pred) +
                         (1 - y_true) * K.log(1 - y_pred))
    
    return K.mean(t_loss)

def FF(features,num_features,max_proj=8,min_proj=2,expand=False):
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
        #inputs_FF = FF(inputs,self.num_feat,expand=True)
        masked_features = layers.Masking(mask_value=0.0)(inputs)
        masked_features = act(layers.Dense(2*projection_dim)(masked_features))
        masked_features = act(layers.Dense(projection_dim)(masked_features))
            
        #Conditional information
        #evt_embed = FF(inputs_evt,self.num_evt)
        evt_embed = act(layers.Dense(projection_dim)(inputs_evt))
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
        #layers.Add()([representation,evt_embed])
            
        representation = act(layers.Dense(128)(concat))
        representation = layers.Dropout(0.1)(representation)
        representation = act(layers.Dense(64)(representation))
        output1 = layers.Dense(1,activation='sigmoid',
                               kernel_initializer="zeros",
                               #kernel_regularizer='l2',
                               #use_bias=False
                               )(representation)
        
        net_trials.append(output1)
    outputs = tf.reduce_mean(net_trials,0) #Average over trials
    
    return  inputs,inputs_evt, outputs


class Classifier(keras.Model):
    """OmniFold Classifier class"""
    def __init__(self,
                 num_feat,
                 num_evt,
                 num_heads=4,
                 num_transformer= 1,
                 projection_dim= 64,
                 nrep = 1,):
        super(Classifier, self).__init__()
        self.num_feat = num_feat
        self.num_evt = num_evt
        inputs = layers.Input((None,self.num_feat))
        inputs_evt = layers.Input((self.num_evt))

        outputs = self.DeepSetsAtt(inputs,
                                   inputs_evt,
                                   num_heads,
                                   num_transformer,
                                   projection_dim,
                                   nrep)
        self.classifier = keras.Model(inputs=[inputs,inputs_evt], outputs=outputs)
        self.model_ema = keras.models.clone_model(self.classifier)
        self.ema = 0.999
        self.loss_tracker = keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        """List of the model's metrics.
        We make sure the loss tracker is listed as part of `model.metrics`
        so that `fit()` and `evaluate()` are able to `reset()` the loss tracker
        at the start of each epoch and at the start of an `evaluate()` call.
        """
        return [self.loss_tracker]

    def call(self,x):
        return self.classifier(x)
        
    def train_step(self, inputs):
        x,y = inputs
        with tf.GradientTape() as tape:
            y_pred = self.classifier(x)
            loss = weighted_binary_crossentropy(y, y_pred)
        trainable_vars = self.classifier.trainable_variables
        # Update weights
        self.optimizer.minimize(loss,trainable_vars,tape=tape)
        for weight, ema_weight in zip(self.classifier.weights, self.model_ema.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, inputs):
        x,y = inputs            
        y_pred = self.classifier(x)
        loss = weighted_binary_crossentropy(y, y_pred)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}
        

    def DeepSetsAtt(
            self,
            inputs,
            inputs_evt,
            num_heads=4,
            num_transformer= 0,
            projection_dim= 64,
            nrep = 1,
    ):

        #act = layers.LeakyReLU(alpha=0.01)
        act = swish
    
    
        net_trials = []

        for _ in range(nrep):
            #inputs_FF = FF(inputs,self.num_feat,expand=True)
            masked_features = layers.Masking(mask_value=0.0)(inputs)
            masked_features = act(layers.Dense(2*projection_dim)(masked_features))
            masked_features = act(layers.Dense(projection_dim)(masked_features))
            
            #Conditional information
            #evt_embed = FF(inputs_evt,self.num_evt)
            evt_embed = act(layers.Dense(projection_dim)(inputs_evt))
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
            #layers.Add()([representation,evt_embed])
            
            representation = act(layers.Dense(128)(concat))
            representation = layers.Dropout(0.1)(representation)
            representation = act(layers.Dense(64)(representation))
            output1 = layers.Dense(1,activation='sigmoid',
                                  kernel_initializer="zeros",
                                  #kernel_regularizer='l2',
                                  #use_bias=False
                                  )(representation)

            # evt_embed = act(layers.Dense(2*projection_dim)(evt_embed))
            # evt_embed = act(layers.Dense(projection_dim)(evt_embed))
            # output1 = layers.Dense(1,activation='sigmoid',
            #                       kernel_initializer="zeros",
            #                       #kernel_regularizer='l2',
            #                       #use_bias=False
            #                       )(evt_embed)
            
            net_trials.append(output1)        
        outputs = tf.reduce_mean(net_trials,0) #Average over trials
        return outputs
    
