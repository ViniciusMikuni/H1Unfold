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
import time

import warnings
warnings.filterwarnings(
    "ignore",
    message=r"Callback method `on_train_batch_end` is slow compared to the batch time",
    category=UserWarning,
)
def hvd_synchronize():
    # Create a dummy tensor
    sync_pause_start = 0

    if hvd.rank() == 0:
        print("\n\n==== Waiting for all horovod processes before reweight ====")
        sync_pause_start = time.time()

    dummy = tf.constant(0.0)
    hvd.allreduce(dummy)

    if hvd.rank() == 0:
            print(f"==== Syncing took {time.time() - sync_pause_start} Seconds ====\n\n")

    # Perform an allreduce operation which acts as a barrier


def label_smoothing(y_true,alpha=0):
    '''Regularization through binary label smoothing'''
    new_label = y_true*(1-alpha) + (1-y_true)*alpha
    return new_label

def reset_weights(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            reset_weights(layer)  #sorry for the recursion, works well here
            continue

        # Get the layer's configuration
        config = layer.get_config()

        # Reinitialize weights
        if hasattr(layer, 'kernel') and layer.kernel is not None:
            initializer = tf.keras.initializers.get(config.get('kernel_initializer',
                                                               'glorot_uniform'))
            layer.kernel.assign(initializer(shape=layer.kernel.shape,
                                            dtype=layer.kernel.dtype))

        # Reinitialize biases
        if hasattr(layer, 'bias') and layer.bias is not None:
            initializer = tf.keras.initializers.get(config.get('bias_initializer',
                                                               'zeros'))
            layer.bias.assign(initializer(shape=layer.bias.shape,
                                          dtype=layer.bias.dtype))


class Multifold():
    def __init__(self,
                 nstrap=0,
                 version = 'Closure',
                 config_file='config_omnifold.json',
                 pretrain = False,
                 load_pretrain = False,
                 fine_tune = False,
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
        self.fine_tune = fine_tune        

        #runs the pretraining (uses same inputs as closure)
        if self.pretrain:
            self.niter = 1  # Skip iterative procedure when pretraining the model

        if self.load_pretrain: 
            self.version += '_pretrained'
            self.lr_factor = 5.  # default 5

        elif self.fine_tune:
            self.version += '_finetuned'
            self.lr_factor = 5.  # default 5
        else:
            self.version += '_fromscratch'
            self.lr_factor = 5.  # default 1

        self.num_steps_reco = None
        self.num_steps_gen = None
        self.mc = None
        self.data=None

        self.weights_folder = '../weights'

        if self.nstrap>0:
            self.weights_folder = '../weights_strap'
        if not os.path.exists(self.weights_folder):
            os.makedirs(self.weights_folder)

        self.pre_train_name1 = '{}/OmniFold_pretrained_step1/checkpoint'.format(self.weights_folder)
        self.pre_train_name2 = '{}/OmniFold_pretrained_step2/checkpoint'.format(self.weights_folder)

        # list of models, used in unfolding. Added to support
        # ensembling. Will deep-copy self.model1 or self.model2
        self.step1_models = []
        self.step2_models = []

    def Unfold(self):
        self.BATCH_SIZE=self.opt['BATCH_SIZE']
        self.EPOCHS=self.opt['EPOCHS']

        self.weights_pull = np.ones(self.mc.weight.shape[0])
        if self.start>0:

            if hvd.rank()==0:
                print("Loading step 2 weights from iteration {}".format(self.start-1))

            # Load step1 starting point
            model_name = '{}/OmniFold_{}_iter{}_step1/checkpoint'.format(self.weights_folder,self.version,self.start-1)
            self.model1.load_weights(model_name).expect_partial()

            #Also load step1 starting point for model2
            self.model2.load_weights(model_name).expect_partial()

            self.weights_push = self.reweight(self.mc.gen, self.model2, batch_size=1000)

        else:
            self.weights_push = np.ones(self.mc.weight.shape[0])
            if self.load_pretrain:
                if hvd.rank()==0:
                    print(f"Loading pretrained weights from {self.pre_train_name1}")                
                self.model1.load_weights(self.pre_train_name1).expect_partial()
                self.model2.load_weights(self.pre_train_name1).expect_partial()
                # model2 loads pre_train1 b/c closure is often run, 
                # and pre-train/closure use same rapgap + django for step 2

        self.CompileModels(self.lr)

        start_time = time.time()
        for i in range(self.start,self.niter):
            if hvd.rank()==0:print(f"ITERATION: {i} / {self.niter-self.start}")
            self.RunStep1(i)        
            if hvd.rank()==0:
                print(f"\n----- Step 1 Iter {i+1}/ {self.niter} took {time.time() -  start_time} Seconds -----\n")
            self.RunStep2(i)
            if hvd.rank()==0:
                print(f"\n----- Step 2 Iter {i+1} / {self.niter} took {time.time() - start_time} Seconds -----\n")
            self.CompileModels(self.lr,fixed=True)

        total_time = time.time() - start_time
        print(f"\n----- OmniFold Took {total_time} Seconds -----\n")

    def RunStep1(self,i):
        '''Data versus reco MC reweighting'''
        if hvd.rank()==0:print("RUNNING STEP 1")

        if self.verbose:
            print("Estimated total number of events: {}".format(
                int((self.mc.nmax + self.data.nmax))//hvd.size()))
            print("Full number of events {}".format(
                np.concatenate((self.labels_mc[self.mc.pass_reco],
                                self.labels_data[self.data.pass_reco])).shape[0]))

        # Model Weights: Pre-train, Fine Tune, or From Scratch
        if self.load_pretrain:
            assert self.load_pretrain != (self.start > 0), \
            "Pretrain cannot be set if start >= 1"

            if hvd.rank()==0:
                print(f"Loading pretrained weights for Step 1 from {self.pre_train_name1}")                
            self.model1.load_weights(self.pre_train_name1).expect_partial()

        elif self.fine_tune:
            if hvd.rank()==0:
                print(f"Loading pretrained weights for Step 1 from {self.pre_train_name1}")
                print(f"and resetting Classifier HEAD (FineTune)")
            self.model1.load_weights(self.pre_train_name1).expect_partial()
            reset_weights(self.model1.head)

        else:  # BaseLine
            reset_weights(self.model1)

        if hvd.rank()==0:
            print("Training From Scratch")


        self.RunModel(
            np.concatenate((self.labels_mc[self.mc.pass_reco],
                            self.labels_data[self.data.pass_reco])),
            np.concatenate((self.weights_push[self.mc.pass_reco]*\
                self.mc.weight[self.mc.pass_reco],
                            self.data.weight[self.data.pass_reco])),
            i, self.model1, stepn=1,
            NTRAIN = self.num_steps_reco*self.BATCH_SIZE,
            cached = (i>self.start)# cache the training data after 1st iter
        )

        #Don't update weights where there's no reco events
        new_weights = np.ones_like(self.weights_pull)
        new_weights[self.mc.pass_reco] = self.reweight(self.mc.reco,self.model1,
                                                       batch_size=1000)[self.mc.pass_reco]

        self.weights_pull = self.weights_push *new_weights


    def RunStep2(self,i):
        '''Gen to Gen reweighing'''        
        if hvd.rank()==0:print("RUNNING STEP 2")

        # model2 loads pre_train 1 b/c closure is often run, where django
        # is used as 'data'. If pre-training used step 2, then the closure
        # test would be useless: we would test on the same dataset as pre-loaded

        if self.load_pretrain:
            if hvd.rank()==0:
                print(f"Loading pretrained weights for Step 2 from {self.pre_train_name1}")                
            self.model2.load_weights(self.pre_train_name1).expect_partial()

        if self.fine_tune:
            if hvd.rank()==0:
                print(f"Loading pretrained weights for Step 2 from {self.pre_train_name1}")                
                print(f"And resetting Classifier HEAD")
            self.model2.load_weights(self.pre_train_name1).expect_partial()
            reset_weights(self.model2.head)

        if hvd.rank()==0:
            print("Training Model 2 From Scratch")

        self.RunModel(
            np.concatenate((self.labels_mc, self.labels_gen)),
            np.concatenate((self.mc.weight, self.mc.weight*self.weights_pull)),
            i, self.model2, stepn=2,
            NTRAIN = self.num_steps_gen*self.BATCH_SIZE,
            cached = (i>self.start)  # cache training data after 1st iter
        )

        new_weights=self.reweight(self.mc.gen,self.model2)
        self.weights_push = new_weights
        # np.save(f'{self.weights_folder}/step2_iter{i}_ensembleAvg_EventWeights.npy', ensemble_avg_weights)


    def RunModel(self,
                 labels,
                 weights,
                 iteration,
                 model,
                 stepn,
                 NTRAIN=1000,
                 cached = False,
                 ):

        test_frac = 1.-self.train_frac
        NTEST = int(test_frac*NTRAIN)
        train_data, test_data = self.cache(labels,weights,stepn,cached,NTRAIN-NTEST)
        num_steps = self.num_steps_reco if stepn==1 else self.num_steps_gen

        if self.verbose and hvd.rank()==0:
            print(80*'#')
            self.log_string(f"Train events used: {NTRAIN}")
            self.log_string(f"Test events used: {NTEST}")
            print(80*'#')

        verbose = 1 if hvd.rank() == 0 else 0

        # Model Checkpoint Name
        model_name = '{}/OmniFold_{}_iter{}_step{}'.format(
            self.weights_folder, self.version, iteration, stepn)
        if self.n_ensemble > 1: model_name += '_ensembleE'  #replace E in loop
        if self.nstrap > 0: model_name += f'_strap{self.nstrap}' #FIXME: strap iter?
        model_name += '/checkpoint'

        if hvd.rank() == 0:
            print('Saving model', model_name)


        callbacks = [
            hvd.callbacks.BroadcastGlobalVariablesCallback(0),
            hvd.callbacks.MetricAverageCallback(),

            ReduceLROnPlateau(patience=1000, min_lr=1e-7,
                              verbose=verbose,monitor="val_loss"),
            EarlyStopping(patience=self.opt['NPATIENCE'],restore_best_weights=True,
                          monitor="val_loss"),
            # monitor="val_loss", min_delta=0.001),
        ]  # will append checkpoint in ensemble loop


        for ensemble in range(self.n_ensemble):
            ''' ensembling implemented here, in RunModel. This reduces parallelization''' 
            ''' but results in overall less variance. Called 'step ensembling' since  '''
            ''' the ensembling is done within each step, before passing onto the next '''
            ''' step or iteration. Alternative would be to call a script and run the  '''
            ''' OmniFold procedure as a whole (for all iterations), [n_ensemble] times'''


            ens_name = model_name
            if self.n_ensemble > 1: ens_name = model_name.replace('E',f'{ensemble}')

            callbacks = [
                hvd.callbacks.BroadcastGlobalVariablesCallback(0),
                hvd.callbacks.MetricAverageCallback(),

                ReduceLROnPlateau(patience=1000, min_lr=1e-7,
                                  verbose=verbose,monitor="val_loss"),
                EarlyStopping(patience=self.opt['NPATIENCE'],restore_best_weights=True,
                              monitor="val_loss"),
                # monitor="val_loss", min_delta=0.001),
            ]  # will append checkpoint in ensemble loop

            if hvd.rank() == 0:
                #avoid simoultanous writes, so only rank0 has checkpoint
                callbacks.append(ModelCheckpoint(ens_name, save_best_only=True,
                                                 mode='auto', period=1,
                                                 save_weights_only=True))

                self.log_string("Ensemble: {} / {}".format(ensemble + 1, self.n_ensemble))


            # Instantiate new model_e, then load from previous iteration
            if iteration < 1:
                model_e = tf.keras.models.clone_model(model)  #clones layers and architecture
                model_e.set_weights(self.model1.get_weights())  #actually clones weights

                if stepn == 1:
                    self.step1_models.append(model_e)
                if stepn == 2:
                    self.step2_models.append(model_e)

            else:
                model_e = self.step1_models[ensemble] if stepn == 1 else self.step2_models[ensemble]

            # model_e = model  # to test iterations passing models

            self.CompileModel(model_e, self.lr ,num_steps)

            if hvd.rank() == 0:
                print(f"\n\nEnsemble {ensemble+1}/{self.n_ensemble} Iter {iteration}/{self.niter-self.start}: Model Weights summary:")
                model_weights = model.get_weights()
                all_weights = tf.concat([tf.reshape(w, [-1]) for w in model_weights if w.size > 0], axis=0)
                mean_weights = tf.reduce_mean([tf.reduce_mean(w) for w in model_weights if w.size > 0])
                stdv_weights = tf.math.reduce_std(all_weights)
                print("Mean of weights:", mean_weights)
                print("Stdv of weights:", stdv_weights)
                print("\n\n")
            # continue

            hist =  model_e.fit(
                train_data,
                epochs=self.EPOCHS,
                steps_per_epoch=int(self.train_frac*NTRAIN//self.BATCH_SIZE),
                validation_data= test_data,
                validation_steps=NTEST//self.BATCH_SIZE,
                verbose= verbose,
                callbacks=callbacks)

            if hvd.rank() ==0:
                with open(ens_name.replace("/checkpoint",".pkl"),"wb") as f:
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


    def CompileModels(self, lr, fixed=False):

        if self.num_steps_reco ==None:
            self.num_steps_reco = int(0.7*(self.mc.nmax + self.data.nmax))\
                //hvd.size()//self.BATCH_SIZE
            self.num_steps_gen = 2*self.mc.nmax//hvd.size()//self.BATCH_SIZE

        self.CompileModel(self.model1, lr, self.num_steps_reco,False)
        self.CompileModel(self.model2, lr, self.num_steps_gen, fixed)

        # loop over ensembles
        if self.n_ensemble > 1 and len(self.step1_models) > 0:
            for model in self.step1_models:
                self.CompileModel(model,lr,self.num_steps_reco,fixed)
            for model in self.step2_models:
                self.CompileModel(model,lr,self.num_steps_gen, fixed)

    def CompileModel(self,model,lr,num_steps,fixed=False):

        min_learning_rate = 1e-5

        lr_schedule_body = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=lr/self.lr_factor,
            warmup_target = lr*np.sqrt(self.size)/self.lr_factor,
            warmup_steps= int(3*self.train_frac*(num_steps)),
            decay_steps= int(self.EPOCHS*self.train_frac*self.num_steps_reco),
            alpha = 1e-2,)

        lr_schedule_head = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=lr,
            warmup_target = lr*np.sqrt(self.size),
            warmup_steps= int(3*self.train_frac*(num_steps)),
            decay_steps= int(self.EPOCHS*self.train_frac*self.num_steps_reco),
            alpha = 1e-2,)  # ^reco and gen had n_steps_reco in decay


        opt_head = tf.keras.optimizers.Lion(
            learning_rate=min_learning_rate if fixed else lr_schedule_head,
            weight_decay=1e-5,
            beta_1=0.95,
            beta_2=0.99)

        opt_body = tf.keras.optimizers.Lion(
            learning_rate=min_learning_rate if fixed else lr_schedule_body,
            weight_decay=1e-5,
            beta_1=0.95,
            beta_2=0.99)

        opt_head = hvd.DistributedOptimizer(opt_head)
        opt_body = hvd.DistributedOptimizer(opt_body)

        model.compile(opt_body, opt_head)


    def PrepareInputs(self):
        self.labels_mc = np.zeros(len(self.mc.pass_reco),dtype=np.float32)
        self.labels_data = np.ones(len(self.data.pass_reco),dtype=np.float32)
        self.labels_gen = np.ones(len(self.mc.pass_gen),dtype=np.float32)

        if hvd.rank() == 0:
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

        # if self.verbose:
        #     print(self.model2.classifier.summary())


    def reweight(self,events,model,batch_size=None):
        if batch_size is None:
            batch_size =  self.BATCH_SIZE

        # normally, would pass a flag instead of actual model
        # but by passing self.model1 or 2, we can keep functionality
        models = self.step1_models if model == self.model1 else self.step2_models
        if hvd.rank() == 0:
            print("In Rewight Function, mfold.models = ", models)

        # Make sure all ranks finish before averaging model outputs
        if hvd.size() > 1:
            hvd_synchronize()

        avg_weights = np.zeros((len(events[0])))
        for model in models:
            # self.model1_ema = self.model1.model_ema
            f = expit(model.model_ema.predict(events,batch_size=batch_size,verbose=self.verbose)[0])
            # f = expit(model.predict(events,batch_size=batch_size,verbose=self.verbose)[0])
            weights = f / (1. - f)  # approximates likelihood ratio
            weights = np.nan_to_num(weights[:,0],posinf=1)
            if hvd.rank() == 0:
                print("Avg f = ",np.mean(f))
                print("Avg Weights after reweight = ",np.mean(weights))
            avg_weights += weights / len(models)
        if hvd.rank() == 0:
            print("Avg Weights after averaging reweight = ",np.mean(avg_weights))
        return avg_weights

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
