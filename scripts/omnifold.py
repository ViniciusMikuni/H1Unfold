import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import  Input, Dropout
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint, ReduceLROnPlateau
import os, gc
import horovod.tensorflow.keras as hvd
from sklearn.model_selection import train_test_split
from scipy.special import expit
import logging
from architecture import Classifier,weighted_binary_crossentropy
import utils
import pickle

def label_smoothing(y_true,alpha=0):
    '''Regularization through binary label smoothing'''
    new_label = y_true*(1-alpha) + (1-y_true)*alpha
    return new_label


def assign_random_weights(model):

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Layer) and layer.weights:
            for weight in layer.weights:

                weight_shape = weight.shape 
                random_weights = tf.random.normal(weight_shape)

                layer.set_weights([random_weights if w.shape == random_weights.shape else w 
                                   for w in layer.get_weights()])

class Multifold():
    def __init__(self,
                 nstrap=0,
                 version = 'Closure',
                 config_file='config_omnifold.json',
                 pretrain = False,
                 load_pretrain = False,
                 verbose=1,
                 start = 0,
                 ):
        
        self.version = version
        self.verbose = verbose
        self.log_file =  open('log_{}.txt'.format(self.version),'w')
        self.opt = utils.LoadJson(config_file)
        self.start = start
        self.train_frac = 0.8
        self.niter = self.opt['NITER']        
        self.num_feat = self.opt['NFEAT'] 
        self.num_event = self.opt['NEVT']
        self.lr = float(self.opt['LR'])
        self.n_ensemble=self.opt['NENSEMBLE']
        self.size = hvd.size()
        self.nstrap=nstrap
        self.pretrain = pretrain
        self.load_pretrain = load_pretrain        
        
        if self.pretrain:
            self.niter = 1 #Skip iterative procedure when pretraining the model
        if self.load_pretrain:
            self.version += '_pretrained'
            self.lr_factor = 5.
        else:
            self.lr_factor = 1.

        self.num_steps_reco = None
        self.num_steps_gen = None
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
                                        
        self.weights_pull = np.ones(self.mc.weight.shape[0])
        if self.start>0:
            if hvd.rank()==0:
                print("Loading step 2 weights from iteration {}".format(self.start-1))
            model_name = '{}/OmniFold_{}_iter{}_step2/checkpoint'.format(self.weights_folder,self.version,self.start-1)
            self.model2.load_weights(model_name).expect_partial()
            self.weights_push = self.reweight(self.mc.gen,self.model2_ema,batch_size=1000)
            #Also load model 1 to have a better starting point
            model_name = '{}/OmniFold_{}_iter{}_step1/checkpoint'.format(self.weights_folder,self.version,self.start-1)
            self.model1.load_weights(model_name).expect_partial()
        else:
            self.weights_push = np.ones(self.mc.weight.shape[0])
            if self.load_pretrain:
                if hvd.rank()==0:
                    print("Loading pretrained weights")                
                model_name = '{}/OmniFold_pretrained_step1/checkpoint'.format(self.weights_folder)
                self.model1.load_weights(model_name).expect_partial()
                model_name = '{}/OmniFold_pretrained_step2/checkpoint'.format(self.weights_folder)
                self.model2.load_weights(model_name).expect_partial()

        self.CompileModel(self.lr)
        for i in range(self.start,self.niter):
            if hvd.rank()==0:print(f"ITERATION: {i} / {self.niter-self.start}")
            self.RunStep1(i)        
            self.RunStep2(i)
            self.CompileModel(self.lr,fixed=True)

            
    def RunStep1(self,i):
        '''Data versus reco MC reweighting'''
        if hvd.rank()==0:print("RUNNING STEP 1")

        if self.verbose:
            print("Estimated total number of events: {}".format(int((self.mc.nmax + self.data.nmax))//hvd.size()))
            print("Full number of events {}".format(np.concatenate((self.labels_mc[self.mc.pass_reco] , self.labels_data[self.data.pass_reco])).shape[0]))

        ensemble_avg_weights = np.zeros_like(self.weights_pull)

        for e in range(self.n_ensemble):
            if hvd.rank()==0:
                print(f"RUNNING ENSEMBlE {e} / {self.n_ensemble} \n")

            # Load pre-trained model weights in ensemble loop
            if self.load_pretrain:
                assert self.load_pretrain != (self.start > 0), \
                "Pretrain cannot be set if start >= 1"
                
                if hvd.rank()==0:
                    print("Loading pretrained weights for step 1")                

                model_name = '{}/OmniFold_pretrained_step1/checkpoint'.format(self.weights_folder)
                self.model1.load_weights(model_name).expect_partial()

            # Assign Random weights, if not loading pre-trained weights
            else:
                if hvd.rank()==0:
                    print("Randomly Assigning Random weights for step 1")
                assign_random_weights(self.model1)

            self.RunModel(
                np.concatenate((self.labels_mc[self.mc.pass_reco],
                                self.labels_data[self.data.pass_reco])),
                np.concatenate((self.weights_push[self.mc.pass_reco]*\
                    self.mc.weight[self.mc.pass_reco],
                                self.data.weight[self.data.pass_reco])),
                i,e,self.model1,stepn=1,
                NTRAIN = self.num_steps_reco*self.BATCH_SIZE,
                # cached = (i>self.start) and (e > 0)
                cached = False
                # ^ after first training cache the training data
            )

            #Don't update weights where there's no reco events
            new_weights = np.ones_like(self.weights_pull)
            new_weights[self.mc.pass_reco] = self.reweight(self.mc.reco,self.model1_ema,
                                                           batch_size=1000)[self.mc.pass_reco]

            ensemble_avg_weights += new_weights/self.n_ensemble  # running average

            tf.keras.backend.clear_session() # double check weights are reset, random initialization
            del new_weights
            gc.collect()
            self.CompileModel(self.lr,fixed=(i>0))

        # self.weights_pull = self.weights_push *new_weights
        self.weights_pull = self.weights_push *ensemble_avg_weights

    def RunStep2(self,i):
        '''Gen to Gen reweighing'''        
        if hvd.rank()==0:print("RUNNING STEP 2")
        
        ensemble_avg_weights = np.zeros_like(self.weights_pull)

        for e in range(self.n_ensemble):
            if hvd.rank()==0:
                print(f"RUNNING ENSEMBlE {e} / {self.n_ensemble} \n")

            if self.load_pretrain:
                if hvd.rank()==0:
                    print("Loading pretrained weights for Step 2")                
                model_name = '{}/OmniFold_pretrained_step2/checkpoint'.format(self.weights_folder)
                self.model2.load_weights(model_name).expect_partial()

            else:
                if hvd.rank()==0:
                    print("Randomly Assigning Random weights for step 2")
                assign_random_weights(self.model2)

            self.RunModel(
                np.concatenate((self.labels_mc, self.labels_gen)),
                np.concatenate((self.mc.weight, self.mc.weight*self.weights_pull)),
                i,e,self.model2,stepn=2,
                NTRAIN = self.num_steps_gen*self.BATCH_SIZE,
                cached = False
                # cached = (i>self.start) and (e > 0) #after first training cache the training data
            )

            new_weights=self.reweight(self.mc.gen,self.model2_ema)

            ensemble_avg_weights += new_weights/self.n_ensemble  # running average

            tf.keras.backend.clear_session()
            del new_weights
            gc.collect()
            self.CompileModel(self.lr,fixed=(i>0))

        # self.weights_push = new_weights
        self.weights_push = ensemble_avg_weights
        np.save(f'{self.weights_folder}/step2_iter{i}_ensembleAvg_EventWeights.npy', ensemble_avg_weights)


    def RunModel(self,
                 labels,
                 weights,
                 iteration,
                 ensemble,
                 model,
                 stepn,
                 NTRAIN=1000,
                 cached = False,
                 ):

        test_frac = 1.-self.train_frac
        NTEST = int(test_frac*NTRAIN)
        train_data, test_data = self.cache(labels,weights,stepn,cached,NTRAIN-NTEST)
        
        if self.verbose and hvd.rank()==0:
            print(80*'#')
            self.log_string("Train events used: {}, Test events used: {}".format(NTRAIN,NTEST))
            print(80*'#')

        verbose = 1 if hvd.rank() == 0 else 0
        
        callbacks = [
            hvd.callbacks.BroadcastGlobalVariablesCallback(0),
            hvd.callbacks.MetricAverageCallback(),
            
            ReduceLROnPlateau(patience=1000, min_lr=1e-7,
                              verbose=verbose,monitor="val_loss"),
            EarlyStopping(patience=self.opt['NPATIENCE'],restore_best_weights=True,
                          monitor="val_loss"),
        ]
        
        
        if hvd.rank() ==0:
            if self.nstrap>0:
                model_name = '{}/OmniFold_{}_iter{}_ens{}_step{}_strap{}/checkpoint'.format(
                    self.weights_folder,self.version,iteration,ensemble,stepn,self.nstrap)
            else:
                if self.pretrain:
                    model_name = '{}/OmniFold_pretrained_step{}/checkpoint'.format(
                        self.weights_folder,stepn)
                else:
                    model_name = '{}/OmniFold_{}_iter{}_ens{}_step{}/checkpoint'.format(
                        self.weights_folder,self.version,iteration,ensemble,stepn)

            callbacks.append(ModelCheckpoint(model_name,save_best_only=True,
                                             mode='auto',period=1,save_weights_only=True))
                    
        hist =  model.fit(
            train_data,
            epochs=self.EPOCHS,
            steps_per_epoch=int(self.train_frac*NTRAIN//self.BATCH_SIZE),
            validation_data= test_data,
            validation_steps=NTEST//self.BATCH_SIZE,
            verbose= verbose,
            callbacks=callbacks)
        
        if hvd.rank() ==0:
            with open(model_name.replace("/checkpoint",".pkl"),"wb") as f:
                pickle.dump(hist.history, f)
        
        del train_data, test_data
        gc.collect()


    def cache(self,
              label,
              weights,
              stepn,
              cached,
              NTRAIN
              ):


        if not cached:
            if self.verbose:
                self.log_string("Creating cached data from step {}".format(stepn))
                    
            if stepn ==1:
                self.idx_1 = np.arange(label.shape[0])
                np.random.shuffle(self.idx_1)
                self.tf_data1 = tf.data.Dataset.from_tensor_slices(
                    {'inputs_particle_1':np.concatenate((self.mc.reco[0][self.mc.pass_reco], self.data.reco[0][self.data.pass_reco]))[self.idx_1],
                     'inputs_event_1':np.concatenate((self.mc.reco[1][self.mc.pass_reco], self.data.reco[1][self.data.pass_reco]))[self.idx_1],
                     'inputs_mask_1':np.concatenate((self.mc.reco[2][self.mc.pass_reco], self.data.reco[2][self.data.pass_reco]))[self.idx_1],
                     })
                #del self.mc.reco, self.data.reco
                #gc.collect()

            elif stepn ==2:
                self.idx_2 = np.arange(label.shape[0])
                np.random.shuffle(self.idx_2)
                self.tf_data2 = tf.data.Dataset.from_tensor_slices(
                    {'inputs_particle_2':np.concatenate((self.mc.gen[0], self.mc.gen[0]))[self.idx_2],
                     'inputs_event_2':np.concatenate((self.mc.gen[1], self.mc.gen[1]))[self.idx_2],
                     'inputs_mask_2':np.concatenate((self.mc.gen[2], self.mc.gen[2]))[self.idx_2],
                     })
                #del self.mc.gen, self.data.gen
                #gc.collect()

        idx = self.idx_1 if stepn==1 else self.idx_2

        if hvd.rank()==0:
            print(label[idx])
            print(NTRAIN,idx.shape[0])
        labels = tf.data.Dataset.from_tensor_slices(np.stack((label[idx],weights[idx]),axis=1))
        
        if stepn ==1:
            data = tf.data.Dataset.zip((self.tf_data1,labels))
        elif stepn==2:
            data = tf.data.Dataset.zip((self.tf_data2,labels))
        else:
            logging.error("ERROR: STEPN not recognized")

                
        train_data = data.take(NTRAIN).shuffle(NTRAIN).repeat().batch(self.BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
        test_data  = data.skip(NTRAIN).repeat().batch(self.BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
        del data
        gc.collect()
        return train_data, test_data

    def Preprocessing(self):
        self.PrepareInputs()
        self.PrepareModel()


    def CompileModel(self,lr,fixed=False):

        if self.num_steps_reco ==None:
            self.num_steps_reco = int(0.7*(self.mc.nmax + self.data.nmax))//hvd.size()//self.BATCH_SIZE
            self.num_steps_gen = 2*self.mc.nmax//hvd.size()//self.BATCH_SIZE
            if hvd.rank()==0:print(self.num_steps_reco,self.num_steps_gen)

        
        lr_schedule_body_reco = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=lr/self.lr_factor,
            warmup_target = lr*np.sqrt(self.size)/self.lr_factor,
            warmup_steps= 3*int(self.train_frac*self.num_steps_reco),
            decay_steps= self.EPOCHS*int(self.train_frac*self.num_steps_reco),
            alpha = 1e-2,
        )


        lr_schedule_head_reco = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=lr,
            warmup_target = lr*np.sqrt(self.size),
            warmup_steps= 3*int(self.train_frac*(self.num_steps_reco)),
            decay_steps= self.EPOCHS*int(self.train_frac*self.num_steps_reco),
            alpha = 1e-2,
        )


        lr_schedule_body_gen = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=lr/self.lr_factor,
            warmup_target = lr*np.sqrt(self.size)/self.lr_factor,
            warmup_steps= 3*int(self.train_frac*self.num_steps_gen),
            decay_steps= self.EPOCHS*int(self.train_frac*self.num_steps_reco),
            alpha = 1e-2,
        )


        lr_schedule_head_gen = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=lr,
            warmup_target = lr*np.sqrt(self.size),
            warmup_steps= 3*int(self.train_frac*self.num_steps_gen),
            decay_steps= self.EPOCHS*int(self.train_frac*self.num_steps_reco),
            alpha = 1e-2,
        )


        min_learning_rate = 1e-5
        opt_head1 = tf.keras.optimizers.Lion(
            learning_rate=min_learning_rate if fixed else lr_schedule_head_reco,
            weight_decay=1e-5,
            beta_1=0.95,
            beta_2=0.99)
        
        opt_head1 = hvd.DistributedOptimizer(opt_head1)
        
        opt_body1 = tf.keras.optimizers.Lion(
            learning_rate=min_learning_rate if fixed else lr_schedule_body_reco,
            weight_decay=1e-5,
            beta_1=0.95,
            beta_2=0.99)
        
        opt_body1 = hvd.DistributedOptimizer(opt_body1)


        opt_head2 = tf.keras.optimizers.Lion(
            learning_rate=min_learning_rate if fixed else lr_schedule_head_gen,
            weight_decay=1e-5,
            beta_1=0.95,
            beta_2=0.99)
        
        opt_head2 = hvd.DistributedOptimizer(opt_head2)
        
        opt_body2 = tf.keras.optimizers.Lion(
            learning_rate=min_learning_rate if fixed else lr_schedule_body_gen,
            weight_decay=1e-5,
            beta_1=0.95,
            beta_2=0.99)
        
        opt_body2 = hvd.DistributedOptimizer(opt_body2)


        self.model1.compile(opt_body1,opt_head1)
        self.model2.compile(opt_body2,opt_head2)


    def PrepareInputs(self):
        self.labels_mc = np.zeros(len(self.mc.pass_reco),dtype=np.float32)
        self.labels_data = np.ones(len(self.data.pass_reco),dtype=np.float32)
        self.labels_gen = np.ones(len(self.mc.pass_gen),dtype=np.float32)

        print(f"Length of MC = {len(self.labels_mc)}")
        print(f"Length of data = {len(self.labels_data)}")
        # # Label smoothing to avoid overtraining, experimental feature
        # self.labels_mc = label_smoothing(np.zeros(len(self.mc.pass_reco)),0.1)
        # self.labels_data = label_smoothing(np.ones(len(self.data.pass_reco)),0.1)
        # self.labels_gen = label_smoothing(np.ones(len(self.mc.pass_gen)),0.1)


    def PrepareModel(self):
        #Will assume same number of features for simplicity
        if self.verbose:            
            self.log_string("Preparing model architecture with: {} particle features and {} event features".format(self.num_feat,self.num_event))
        self.model1 = Classifier(self.num_feat,
                                 self.num_event,
                                 num_heads=self.opt['NHEADS'],
                                 num_transformer= self.opt['NTRANSF'],
                                 projection_dim= self.opt['NDIM'],
                                 step=1,
                                 )
        
        self.model2 = Classifier(self.num_feat,
                                 self.num_event,
                                 num_heads=self.opt['NHEADS'],
                                 num_transformer= self.opt['NTRANSF'],
                                 projection_dim= self.opt['NDIM'],
                                 step=2,
                                 )

        
        self.model1_ema = self.model1.model_ema
        self.model2_ema = self.model2.model_ema
        
        # if self.verbose:
        #     print(self.model2.classifier.summary())


    def reweight(self,events,model,batch_size=None):
        if batch_size is None:
           batch_size =  self.BATCH_SIZE

        f = expit(model.predict(events,batch_size=batch_size,verbose=self.verbose)[0])            
        weights = f / (1. - f)
        return np.nan_to_num(weights[:,0],posinf=1)

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
