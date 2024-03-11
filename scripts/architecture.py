import tensorflow as tf
from tensorflow.keras.activations import swish
from tensorflow.keras import layers
import tensorflow.keras.backend as K
import numpy as np
from tensorflow import keras
from layers import StochasticDepth, TalkingHeadAttention, LayerScale,SimpleHeadAttention
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.losses import mse

def reset_weights(model, default_initializer=GlorotUniform):
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            # If the layer is a model itself (e.g., Sequential or another Functional Model),
            # recursively reset its weights
            reset_weights(layer, default_initializer)
        else:
            # For layers with weights
            for w in layer.weights:
                # Check if the layer has an initializer
                if hasattr(w, 'initializer') and w.initializer:
                    initializer = w.initializer
                else:
                    # Use the default initializer
                    initializer = default_initializer()
                
                # Assign new weights
                new_w = initializer(w.shape, dtype=w.dtype)
                w.assign(new_w)


                
def weighted_binary_crossentropy(y_true, y_pred):
    weights = tf.cast(tf.gather(y_true, [1], axis=1),tf.float32) # event weights
    y_true = tf.cast(tf.gather(y_true, [0], axis=1),tf.float32) # actual y_true for loss
    
    # Clip the prediction value to prevent NaN's and Inf's
    # epsilon = K.epsilon()
    # y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    # t_loss = -weights * ((y_true) * K.log(y_pred) +
    #                      (1 - y_true) * K.log(1 - y_pred))

    t_loss = weights*tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    #entropy term
    probs = tf.nn.sigmoid(y_pred)
    entropy = -weights*(probs * tf.math.log(probs + 1e-6))
    return K.mean(t_loss)
    return K.mean(t_loss - 0.1*entropy)



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
                 nrep = 1,
                 ):
        super(Classifier, self).__init__()
        self.num_feat = num_feat
        self.num_evt = num_evt
        self.step = step
        inputs_part = layers.Input((num_part,self.num_feat),name='inputs_particle_{}'.format(step))
        inputs_evt = layers.Input((num_evt),name='inputs_event_{}'.format(step))
        inputs_point = layers.Input((num_part,3),name='inputs_point_{}'.format(step))
        inputs_mask = layers.Input((num_part,1),name='inputs_mask_{}'.format(step))

        outputs_body,event_encoded = self.PET_body(inputs_part,
                                     inputs_evt,
                                     inputs_point,
                                     inputs_mask,
                                     num_part,
                                     num_heads,
                                     num_transformer,
                                     projection_dim,
                                     nrep)
                
        self.body = keras.Model(inputs=[inputs_part,inputs_evt,inputs_point,inputs_mask],
                                outputs=outputs_body)

        outputs_head = self.PET_head(outputs_body,
                                     event_encoded,
                                     projection_dim
                                     )

        self.classifier = keras.Model(inputs=[inputs_part,inputs_evt,inputs_point,inputs_mask],
                                      outputs=outputs_head)
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
        with tf.GradientTape(persistent=True) as tape:
            y_pred,y_evt = self.classifier(x)
            loss_pred = weighted_binary_crossentropy(y, y_pred)
            loss_evt = mse(x['inputs_event_{}'.format(self.step)],y_evt)
            loss = loss_pred+loss_evt
        trainable_vars = self.classifier.trainable_variables


        self.optimizer.minimize(loss,trainable_vars,tape=tape)

            
        #self.optimizer.apply_gradients(zip(averaged_gradients,trainable_vars))
        # self.head_optimizer.apply_gradients(zip(g, self.head.trainable_variables))
        
        for weight, ema_weight in zip(self.classifier.weights, self.model_ema.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)

        # self.compiled_metrics.update_state(y, y_pred)
        # return {m.name: m.result() for m in self.metrics}
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, inputs):
        x,y = inputs            
        y_pred,y_evt = self.classifier(x)
        loss_evt = mse(x['inputs_event_{}'.format(self.step)],y_evt)
        loss_pred = weighted_binary_crossentropy(y, y_pred)
        loss = loss_pred+loss_evt
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}
        

    def PET_body(
            self,
            inputs_part,
            inputs_evt, 
            inputs_points,
            inputs_mask,
            num_parts,
            num_heads=4,
            num_transformer= 8,
            projection_dim= 64,
            nrep = 1,
            local = True, K = 5,
            num_local = 2,
            interaction = False,
            interaction_dim=64,
            max_int=64,
            drop_probability = 0.0,
            layer_scale = True,
            layer_scale_init = 1e-3,
            talking_head = False
    ):

        stochastic_drop = drop_probability>0.0
        
        event_encoded = layers.GroupNormalization(groups=1)(inputs_evt[:,None])
        event_encoded = get_encodding(event_encoded,projection_dim)

        encoded = layers.GroupNormalization(groups=1)(inputs_part)
        encoded = get_encodding(encoded,projection_dim)
        #mask = inputs_mask
        # mask = tf.concat([tf.ones(shape=(tf.shape(encoded)[0],1, 1)),
        #                   inputs_mask],1)
        mask_matrix = tf.matmul(inputs_mask,inputs_mask,transpose_b=True)

        if local:
            coord_shift = tf.multiply(999., tf.cast(tf.equal(inputs_mask, 0), dtype='float32'))        
            points = inputs_points[:,:,:2]
            local_features = inputs_part
            for _ in range(num_local):
                local_features = get_neighbors(coord_shift+points,local_features,projection_dim,K)
                points = local_features
                
            encoded = layers.Add()([local_features,encoded])
        if interaction:
            global_features = get_interaction(inputs_points,interaction_dim,
                                              num_heads,num_parts,mask_matrix[:,:,:,None],max_int)
        else:
            global_features = None
            

        encoded = layers.Add()([event_encoded,encoded])
                               
        for i in range(num_transformer):
            x1 = layers.GroupNormalization(groups=1)(encoded)
            if talking_head:
                updates, _ = TalkingHeadAttention(projection_dim, num_heads, 0.0)(
                    x1,global_features)
            else:
                updates = layers.MultiHeadAttention(num_heads=num_heads,
                                                    key_dim=projection_dim//num_heads)(
                                                        x1,x1,attention_mask=mask_matrix)
                
            updates = layers.GroupNormalization(groups=1)(updates)
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
                        
        return encoded, event_encoded


    def PET_head(
            self,
            encoded,
            event_encoded,
            projection_dim= 64,
            num_heads=4,
            num_class_layers=2,
            layer_scale = True,
            layer_scale_init = 1e-3,
    ):

        
        class_tokens = layers.GroupNormalization(groups=1)(event_encoded)
        for _ in range(num_class_layers):
            concatenated = tf.concat([class_tokens, encoded],1)

            x1 = layers.GroupNormalization(groups=1)(concatenated)            
            updates = layers.MultiHeadAttention(num_heads=num_heads,
                                                key_dim=projection_dim//num_heads)(
                                                    query=x1[:,:1], value=x1, key=x1)
            updates = layers.GroupNormalization(groups=1)(updates)
            if layer_scale:
                updates = LayerScale(layer_scale_init, projection_dim)(updates)

            x2 = layers.Add()([updates,class_tokens])
            x3 = layers.GroupNormalization(groups=1)(x2)
            x3 = layers.Dense(2*projection_dim,activation="gelu")(x3)
            x3 = layers.Dense(projection_dim)(x3)
            if layer_scale:
                x3 = LayerScale(layer_scale_init, projection_dim)(x3)
            class_tokens = layers.Add()([x3,x2])

        class_tokens = layers.GroupNormalization(groups=1)(class_tokens)
        outputs_pred = layers.Dense(1,activation=None)(class_tokens[:,0])
        outputs_evt = layers.Dense(self.num_evt)(class_tokens[:,0])
                
        return [outputs_pred,outputs_evt]


def get_interaction(points,projection_dim,num_heads,num_parts,mask,nreal):
    pi = points[:,None,:nreal]
    pj = tf.transpose(pi, perm=[0, 2, 1, 3])
    drij = pairwise_distance(points[:,:nreal,:2])
    kt = tf.minimum(pi[:,:,:,2], pj[:,:,:,2])
    m2 = 2*pi[:,:,:,2]*pj[:,:,:,2]*(tf.cosh(pi[:,:,:,0]-pj[:,:,:,0])-tf.cos(pi[:,:,:,1]-pj[:,:,:,1]))
    
    inter = tf.math.log(1e-6+tf.stack([drij,kt*drij,kt/(pi[:,:,:,2]+ pj[:,:,:,2]+1e-6),m2],-1))*mask[:,:nreal,:nreal]
    
    inter = layers.Dense(projection_dim,activation='gelu')(inter)
    inter = layers.Dense(projection_dim,activation='gelu')(inter)
    inter = layers.Dense(num_heads)(inter)


    inter = layers.ZeroPadding2D(((1,num_parts- nreal),(1,num_parts- nreal)))(inter)

    
    # padding_horiz = tf.zeros([tf.shape(points)[0], nreal,tf.shape(points)[1]- nreal, num_heads])
    # padding_vert = tf.zeros([tf.shape(points)[0], tf.shape(points)[1] - nreal, tf.shape(points)[1], num_heads])
    # inter = tf.concat([inter, padding_horiz], axis=2)
    # inter = tf.concat([inter, padding_vert], axis=1)


    inter = layers.GroupNormalization(groups=1)(inter)
    return tf.transpose(inter, perm=[0, 3, 1, 2])

def get_neighbors(points,features,projection_dim,K):
    drij = pairwise_distance(points)  # (N, P, P)
    _, indices = tf.nn.top_k(-drij, k=K + 1)  # (N, P, K+1)
    indices = indices[:, :, 1:]  # (N, P, K)
    knn_fts = knn(tf.shape(points)[1], K, indices, features)  # (N, P, K, C)
    knn_fts_center = tf.broadcast_to(tf.expand_dims(features, 2), tf.shape(knn_fts))
    local = tf.concat([knn_fts-knn_fts_center,knn_fts_center],-1)
    local = layers.Dense(2*projection_dim,activation='gelu')(local)
    local = layers.Dense(projection_dim,activation='gelu')(local)
    local = tf.reduce_mean(local,-2)
    
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


def get_encodding(x,projection_dim):
    x = layers.Dense(projection_dim,activation='gelu')(x)
    x = layers.Dense(4*projection_dim,activation='gelu')(x)
    x = layers.Dense(projection_dim,activation='gelu')(x)
    return x


def get_reduction(encoded,num_part,projection_dim,
                  num_layer = 4,num_heads=4):
    embedding = tf.Variable(tf.zeros(shape=(num_part, projection_dim)),trainable = True)
    embedding = tf.tile(embedding[None, :, :], [tf.shape(encoded)[0], 1, 1])
    
    for _ in range(num_layer):
        x1 = layers.GroupNormalization(groups=1)(encoded)
        updates = layers.MultiHeadAttention(num_heads=num_heads,key_dim=projection_dim//num_heads)(
            query=embedding, value=x1, key=x1)
        updates = layers.GroupNormalization(groups=1)(updates)
                        
        x2 = layers.Add()([updates,embedding])
        x3 = layers.GroupNormalization(groups=1)(x2)
        x3 = layers.Dense(2*projection_dim,activation="gelu")(x3)
        x3 = layers.Dense(projection_dim)(x3)

        embedding = layers.Add()([x3,x2])
        
    embedding = layers.GroupNormalization(groups=1)(embedding)
    return embedding
