import numpy as np
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.layers import  Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint, ReduceLROnPlateau
import os
import horovod.tensorflow.keras as hvd
#from sklearn.model_selection import train_test_split
import logging
from architecture import DeepSetsAtt, weighted_binary_crossentropy
import utils

def label_smoothing(y_true,alpha=0):
    '''Regularization through binary label smoothing'''
    new_label = y_true*(1-alpha) + (1-y_true)*alpha
    return new_label

class Multifold():
    def __init__(self,
                 nstrap=0,
                 version = 'Closure',
                 config_file='config_omnifold.json',
                 verbose=1,
                 ):
        
        self.version = version
        self.verbose = verbose
        self.log_file =  open('log_{}.txt'.format(self.version),'w')
        self.opt = utils.LoadJson(config_file)
        self.niter = self.opt['NITER']
        self.nstrap=nstrap
        self.mc = None
        self.data=None

        self.weights_folder = '../weights'
        if self.nstrap>0:
            self.weights_folder = '../weights_strap'
        if not os.path.exists(self.weights_folder):
            os.makedirs(self.weights_folder)
            
    def Unfold(self):
        self.BATCH_SIZE=self.opt['BATCH_SIZE']
        self.EPOCHS=self.opt['EPOCHS']
        self.CompileModel(float(self.opt['LR']))
                                        
        self.weights_pull = np.ones(self.mc.weight.shape[0])
        self.weights_push = np.ones(self.mc.weight.shape[0])
            
        for i in range(self.niter):
            if hvd.rank()==0:print("ITERATION: {}".format(i + 1))            
            self.RunStep1(i)        
            self.RunStep2(i)            
            self.CompileModel(float(self.opt['LR']))

    def RunStep1(self,i):
        '''Data versus reco MC reweighting'''
        if hvd.rank()==0:print("RUNNING STEP 1")
        
        self.RunModel(
            np.concatenate((self.mc.reco[0], self.data.reco[0])),
            np.concatenate((self.mc.reco[1], self.data.reco[1])),
            np.concatenate((self.labels_mc , self.labels_data )),
            np.concatenate((self.weights_push*self.mc.pass_reco*self.mc.weight,self.data.weight*self.data.pass_reco)),
            i,self.model1,stepn=1,
            NTRAIN = (self.mc.nmax + self.data.nmax)//hvd.size(),
            cached = i>0 #after first training cache the training data
        )
                        
        new_weights=self.reweight(self.mc.reco,self.model1)            
        new_weights[self.mc.pass_reco==0]=1.0 #Don't update weights where there's no reco events
        self.weights_pull = self.weights_push *new_weights
        # self.weights_pull = self.weights_pull/np.average(self.weights_pull)

    def RunStep2(self,i):
        '''Gen to Gen reweighing'''        
        if hvd.rank()==0:print("RUNNING STEP 2")
        
        self.RunModel(
            np.concatenate((self.mc.gen[0], self.mc.gen[0])),
            np.concatenate((self.mc.gen[1], self.mc.gen[1])),
            np.concatenate((self.labels_mc, self.labels_gen)),
            np.concatenate((self.mc.weight, self.mc.weight*self.weights_pull)),
            i,self.model2,stepn=2,
            NTRAIN = (self.mc.nmax + self.mc.nmax)//hvd.size(),
            cached = i>0 #after first training cache the training data
        )

        new_weights=self.reweight(self.mc.gen,self.model2)
        self.weights_push = new_weights
        # self.weights_push = self.weights_push/np.average(self.weights_push)

    def RunModel(self,
                 part,
                 evt,
                 labels,
                 weights,
                 iteration,
                 model,
                 stepn,
                 NTRAIN=1000,
                 cached = False,
                 ):

        NTEST = int(0.2*NTRAIN)
        train_data, test_data = self.cache(part,evt,labels,weights,stepn,cached,NTRAIN-NTEST)
        
        #train_part,test_part,train_evt,test_evt,train_labels,test_labels,train_weights,test_weights = train_test_split(part,evt,labels,weights,test_size=NTEST,shuffle=True)
        
        
        if self.verbose and hvd.rank()==0:
            print(80*'#')
            self.log_string("Train events used: {}, Test events used: {}".format(NTRAIN,NTEST))
            print(80*'#')

        verbose = 1 if hvd.rank() == 0 else 0
        
        callbacks = [
            hvd.callbacks.BroadcastGlobalVariablesCallback(0),
            hvd.callbacks.MetricAverageCallback(),
            ReduceLROnPlateau(patience=self.opt['NPATIENCE']//2, min_lr=1e-7,verbose=verbose),
            EarlyStopping(patience=self.opt['NPATIENCE'],restore_best_weights=True)
        ]
        
        base_name = "Omnifold"
        
        if hvd.rank() ==0:
            if self.nstrap>0:                
                callbacks.append(
                    ModelCheckpoint('{}/{}_{}_iter{}_step{}_strap{}.h5'.format(self.weights_folder,base_name,self.version,iteration,stepn,self.nstrap),
                                    save_best_only=True,mode='auto',period=1,save_weights_only=True))
            else:
                callbacks.append(
                    ModelCheckpoint('{}/{}_{}_iter{}_step{}.h5'.format(self.weights_folder,base_name,self.version,iteration,stepn),
                                    save_best_only=True,mode='auto',period=1,save_weights_only=True))
                
        _ =  model.fit(
            train_data,
            #[train_part,train_evt,train_labels,train_weights],
            epochs=self.EPOCHS,
            #batch_size = self.BATCH_SIZE,
            steps_per_epoch=int(0.8*NTRAIN/self.BATCH_SIZE),
            validation_data= test_data,
            #validation_data=([[test_part,test_evt,test_labels,test_weights]]),
            validation_steps=int(NTEST/self.BATCH_SIZE),
            #sample_weight = train_weights,
            verbose=verbose,
            callbacks=callbacks)


    def cache(self,
              part,
              evt,
              label,
              weights,
              stepn,
              cached,
              NTRAIN
              ):

        if cached:
            if self.verbose:
                self.log_string("Loading cached data from step {}".format(stepn))
            if stepn ==1:
                train_data = self.train_cached_step1
                test_data = self.test_cached_step1
            elif stepn==2:
                train_data = self.train_cached_step2
                test_data = self.test_cached_step2
            else:
                logging.error("ERROR: STEPN not recognized")
        else:
            if self.verbose:
                self.log_string("Creating cached data from step {}".format(stepn))
                    
            if stepn ==1:
                tf_data = tf.data.Dataset.from_tensor_slices((
                    {'input_1':part,
                     'input_2':evt}, 
                    np.stack((label,weights),axis=1))).shuffle(NTRAIN)
                self.train_cached_step1 = tf_data.take(NTRAIN).cache().repeat().batch(self.BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
                self.test_cached_step1 = tf_data.skip(NTRAIN).cache().repeat().batch(self.BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
                train_data, test_data = self.train_cached_step1, self.test_cached_step1
            elif stepn ==2:
                tf_data = tf.data.Dataset.from_tensor_slices((
                    {'input_3':part,
                     'input_4':evt}, 
                    np.stack((label,weights),axis=1))).shuffle(NTRAIN)
                self.train_cached_step2 = tf_data.take(NTRAIN).cache().repeat().batch(self.BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
                self.test_cached_step2 = tf_data.skip(NTRAIN).cache().repeat().batch(self.BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
                train_data, test_data = self.train_cached_step2, self.test_cached_step2
            else:
                logging.error("ERROR: STEPN not recognized")

        return train_data, test_data

    def Preprocessing(self):
        self.PrepareInputs()
        self.PrepareModel()


    def CompileModel(self,lr):
        opt1 = tensorflow.keras.optimizers.legacy.Adam(learning_rate=lr*np.sqrt(hvd.size()))
        opt1 = hvd.DistributedOptimizer(opt1)
        opt2 = tensorflow.keras.optimizers.legacy.Adam(learning_rate=lr*np.sqrt(hvd.size()))
        opt2 = hvd.DistributedOptimizer(opt2)

        self.model1.compile(loss=weighted_binary_crossentropy,weighted_metrics=[],
                            optimizer=opt1,experimental_run_tf_function=False)

        self.model2.compile(loss=weighted_binary_crossentropy,weighted_metrics=[],
                            optimizer=opt2,experimental_run_tf_function=False)


    def PrepareInputs(self):
        self.labels_mc = np.zeros(len(self.mc.pass_reco),dtype=np.float32)
        self.labels_data = np.ones(len(self.data.pass_reco),dtype=np.float32)
        self.labels_gen = np.ones(len(self.mc.pass_gen),dtype=np.float32)

        # Label smoothing to avoid overtraining, experimental feature
        # self.labels_mc = label_smoothing(np.zeros(len(self.mc_reco)),0.01)
        # self.labels_data = label_smoothing(np.ones(len(self.data)),0.01)
        # self.labels_gen = label_smoothing(np.ones(len(self.mc_gen)),0.01)


    def PrepareModel(self):
        #Will assume same number of features for simplicity
        num_feat = self.data.reco[0].shape[2]
        num_event = self.data.reco[1].shape[1]
        if self.verbose:            
            self.log_string("Preparing model architecture with: {} particle features and {} event features".format(num_feat,num_event))


        input_part1,input_evt1,outputs1 = DeepSetsAtt(num_feat,
                                                      num_event,
                                                      num_heads=self.opt['NHEADS'],
                                                      num_transformer= self.opt['NTRANSF'],
                                                      projection_dim= self.opt['NDIM'],
                                                   )

        input_part2,input_evt2,outputs2 = DeepSetsAtt(num_feat,
                                                      num_event,
                                                      num_heads=self.opt['NHEADS'],
                                                      num_transformer= self.opt['NTRANSF'],
                                                      projection_dim= self.opt['NDIM'],
                                                   )

        
        self.model1 = Model(inputs=[input_part1,input_evt1], outputs=outputs1)
        self.model2 = Model(inputs=[input_part2,input_evt2], outputs=outputs2)
        
        if self.verbose and hvd.rank()==0:
            print(self.model2.summary())


    def reweight(self,events,model):
        f = np.nan_to_num(model.predict(events, batch_size=1000),posinf=1,neginf=0)
        weights = f / (1. - f)
        #weights = np.clip(weights,0,10)
        weights = weights[:,0]
        return np.squeeze(np.nan_to_num(weights,posinf=1))

    def CompareDistance(self,patience,min_distance,weights1,weights2):
        distance = np.mean(
            (np.sort(weights1) - np.sort(weights2))**2)
                
        print(80*'#')
        self.log_string("Distance between weights: {}".format(distance))
        print(80*'#')
                
        if distance<min_distance:
            min_distance = distance
            patience = 0
        else:
            print(80*'#')
            print("Distance increased! before {} now {}".format(min_distance,distance))
            print(80*'#')
            patience+=1
        return patience, min_distance
        
    def log_string(self,out_str):
        self.log_file.write(out_str+'\n')
        self.log_file.flush()
        print(out_str)
