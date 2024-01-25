import tensorflow as tf
from tensorflow.keras.activations import swish
from tensorflow.keras import layers
import tensorflow.keras.backend as K
import numpy as np
from tensorflow import keras
from layers import StochasticDepth, TalkingHeadAttention, LayerScale,SimpleHeadAttention

def weighted_binary_crossentropy(y_true, y_pred):
    weights = tf.cast(tf.gather(y_true, [1], axis=1),tf.float32) # event weights
    y_true = tf.cast(tf.gather(y_true, [0], axis=1),tf.float32) # actual y_true for loss

    
    # Clip the prediction value to prevent NaN's and Inf's
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    t_loss = -weights * ((y_true) * K.log(y_pred) +
                         (1 - y_true) * K.log(1 - y_pred))
    
    return K.mean(t_loss)



class Classifier(keras.Model):
    """OmniFold Classifier class"""
    def __init__(self,
                 num_feat,
                 num_evt,
                 num_part=132,
                 num_heads=4,
                 num_transformer= 1,
                 projection_dim= 64,
                 step=1,
                 nrep = 1,):
        super(Classifier, self).__init__()
        self.num_feat = num_feat
        self.num_evt = num_evt
        inputs_part = layers.Input((num_part,self.num_feat),name='inputs_particle_{}'.format(step))
        inputs_evt = layers.Input((num_evt),name='inputs_event_{}'.format(step))
        inputs_point = layers.Input((num_part,2),name='inputs_point_{}'.format(step))
        inputs_mask = layers.Input((num_part,1),name='inputs_mask_{}'.format(step))

        outputs = self.PET(inputs_part,
                           inputs_evt,
                           inputs_point,
                           inputs_mask,
                           num_heads,
                           num_transformer,
                           projection_dim,
                           nrep)
        self.classifier = keras.Model(inputs=[inputs_part,inputs_evt,inputs_point,inputs_mask],
                                      outputs=outputs)
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
        #self.optimizer.minimize(loss,trainable_vars,tape=tape)
        g = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(g, trainable_vars))
        
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
        

    def PET(
            self,
            inputs_part,
            inputs_evt, 
            inputs_points,
            inputs_mask,            
            num_heads=8,
            num_transformer= 8,
            projection_dim= 64,
            nrep = 1,
            local = True, K = 10,
            num_local = 2,
            interaction = False,
            interaction_dim=64,
            num_class_layers=2,
            drop_probability = 0.0,
            avg = False, layer_scale = True,
            layer_scale_init = 1e-3,
            talking_head = False
    ):

        stochastic_drop = drop_probability>0.0
        mask_matrix = tf.matmul(inputs_mask,tf.transpose(inputs_mask,perm=[0,2,1]))

        #Event info will be added as a representative particle
        event_encoded = layers.Dense(2*projection_dim,activation='gelu')(inputs_evt[:,None])
        event_encoded = layers.Dense(projection_dim)(event_encoded)
        
        encoded = layers.Dense(projection_dim)(inputs_part)
        encoded = layers.Add()([encoded,event_encoded])*inputs_mask
        if local:
            coord_shift = tf.multiply(999., tf.cast(tf.equal(inputs_mask, 0), dtype='float32'))        
            points = inputs_points[:,:,:2]
            local_features = inputs_part
            for _ in range(num_local):
                local_features = get_neighbors(coord_shift+points,local_features,projection_dim,K)
                points = local_features
    
            if layer_scale:
                local_features = LayerScale(layer_scale_init, projection_dim)(local_features,inputs_mask) 
            encoded = layers.Add()([local_features,encoded])
        if interaction:
            global_features = get_interaction(inputs_points,interaction_dim,num_heads,mask_matrix)
        else:
            global_features = None

        
        for i in range(num_transformer):
            x1 = layers.GroupNormalization(groups=1)(encoded)
            if talking_head:
                updates, _ = TalkingHeadAttention(projection_dim, num_heads, 0.1)(
                    x1,global_features,mask_matrix[:,None])
            else:
                updates,_ = SimpleHeadAttention(projection_dim, num_heads, 0.1)(
                    x1,global_features,mask_matrix[:,None])
            
            updates = layers.GroupNormalization(groups=1)(updates)*inputs_mask
            if layer_scale:
                updates = LayerScale(layer_scale_init, projection_dim)(updates,inputs_mask)
            if stochastic_drop:
                updates = StochasticDepth(drop_probability)(updates)
            

            x2 = layers.Add()([updates,encoded])
            x3 = layers.GroupNormalization(groups=1)(x2)
            x3 = layers.Dense(4*projection_dim,activation="gelu")(x3)
            x3 = layers.Dense(projection_dim)(x3)
            if layer_scale:
                x3 = LayerScale(layer_scale_init, projection_dim)(x3,inputs_mask)
            if stochastic_drop:
                x3 = StochasticDepth(drop_probability)(x3)
            encoded = layers.Add()([x3,x2])*inputs_mask
        

        if avg:
            encoded = layers.GroupNormalization(groups=1)(encoded)
            representation = layers.GlobalAveragePooling1D()(encoded)     
            representation = layers.Dense(4*projection_dim,activation='gelu')(representation)
            representation = layers.Dropout(0.1)(representation)
            outputs = layers.Dense(1,activation='sigmoid',)(representation)

        else:        
            class_tokens = tf.Variable(tf.zeros(shape=(1, projection_dim)),trainable = True)    
            class_tokens = tf.tile(class_tokens[None, :, :], [tf.shape(encoded)[0], 1, 1])
            for _ in range(num_class_layers):
                concatenated = tf.concat([class_tokens, encoded],1)
                x1 = layers.GroupNormalization(groups=1)(concatenated)
            
                updates = layers.MultiHeadAttention(num_heads=num_heads,key_dim=projection_dim//num_heads)(
                    query=x1[:,:1], value=x1, key=x1)
                updates = layers.GroupNormalization(groups=1)(updates)
                if layer_scale:
                    updates = LayerScale(layer_scale_init, projection_dim)(updates)
            
            
                x2 = layers.Add()([updates,class_tokens])
                x3 = layers.GroupNormalization(groups=1)(x2)
                x3 = layers.Dense(4*projection_dim,activation="gelu")(x3)
                x3 = layers.Dense(projection_dim)(x3)
                if layer_scale:
                    x3 = LayerScale(layer_scale_init, projection_dim)(x3)
                class_tokens = layers.Add()([x3,x2])

            concatenated = tf.concat([class_tokens, encoded],1)
            concatenated = layers.GroupNormalization(groups=1)(concatenated)
            outputs = layers.Dense(1,activation='sigmoid')(concatenated[:,0])
                
        return outputs

def get_interaction(points,projection_dim,num_heads,mask):    
    pi = points[:,None]
    pj = tf.transpose(pi, perm=[0, 2, 1, 3])
    drij = pairwise_distance(points[:,:,:2])
    kt = tf.minimum(pi[:,:,:,2], pj[:,:,:,2])
    m2 = 2*pi[:,:,:,2]*pj[:,:,:,2]*(tf.cosh(pi[:,:,:,0]-pj[:,:,:,0])-tf.cos(pi[:,:,:,1]-pj[:,:,:,1]))
    
    inter = tf.stack([drij,kt*drij,kt/(pi[:,:,:,2]+ pj[:,:,:,2]+1e-6),m2],-1)
    inter = tf.where(inter!=0.0,tf.math.log(inter),tf.zeros_like(inter))
    inter = layers.Dense(projection_dim,activation='gelu')(inter)
    inter = layers.Dense(projection_dim)(inter)
    inter = mask[:,:,:,None]*layers.Dense(num_heads)(inter)
    inter = layers.GroupNormalization(groups=1)(inter,mask=mask[:,:,:,None])*mask[:,:,:,None]
    return tf.transpose(inter, perm=[0, 3, 1, 2])

def get_neighbors(points,features,projection_dim,K):
    drij = pairwise_distance(points)  # (N, P, P)
    _, indices = tf.nn.top_k(-drij, k=K + 1)  # (N, P, K+1)
    indices = indices[:, :, 1:]  # (N, P, K)
    knn_fts = knn(tf.shape(points)[1], K, indices, features)  # (N, P, K, C)
    knn_fts_center = tf.broadcast_to(tf.expand_dims(features, 2), tf.shape(knn_fts))
    #local = tf.concat([knn_fts-knn_fts_center,knn_fts_center],-1)
    local = knn_fts-knn_fts_center
    local = layers.Dense(2*projection_dim,activation='gelu')(local)
    local = layers.Dense(projection_dim)(local)
    local = tf.reduce_mean(local,-2)
    local = layers.GroupNormalization(groups=1)(local,mask=None)
    
    return local


def pairwise_distance(point_cloud):
    r = tf.reduce_sum(point_cloud * point_cloud, axis=2, keepdims=True)
    m = tf.matmul(point_cloud, point_cloud, transpose_b = True)
    D = r - 2 * m + tf.transpose(r, perm=(0, 2, 1))
    return tf.abs(D)


def knn(num_points, k, topk_indices, features):
    # topk_indices: (N, P, K)
    # features: (N, P, C)    
    batch_size = tf.shape(features)[0]

    batch_indices = tf.reshape(tf.range(batch_size), (-1, 1, 1))
    batch_indices = tf.tile(batch_indices, (1, num_points, k))
    indices = tf.stack([batch_indices, topk_indices], axis=-1)
    return tf.gather_nd(features, indices)


