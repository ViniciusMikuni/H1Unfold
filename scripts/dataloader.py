import numpy as np
import tensorflow as tf
import uproot3 as uproot
import os
import h5py as h5

def convert_to_np(file_list,base_path,name,is_data = False,max_part = 191,max_nonzero=132,nevts=30000000):
    reco_dict = {
        'event_features':[],
        'particle_features':[],
    }
    gen_dict = {
            'event_features':[],
            'particle_features':[],
        }
    var_list = ['Q2','e_px','e_py','e_pz','wgt']
    mask_list = ['y','ptmiss','Empz']
    particle_list = ['part_pt','part_eta','part_phi','part_charge']
    
    for ifile,f in enumerate(file_list):
        print("evaluating file {}".format(f))
        tmp_file = uproot.open(os.path.join(base_path,f))['{}/minitree'.format(name)]
        
        reco_dict['event_features'].append(np.stack([tmp_file[feat].array()[:nevts] for feat in var_list],-1))
        reco_dict['particle_features'].append(np.stack([tmp_file[feat].array()[:nevts].pad(max_part).fillna(0).regular() for feat in particle_list],-1))
        mask_reco = np.stack([tmp_file[feat].array()[:nevts] for feat in mask_list],-1)
        if 'Data' not in name:            
            gen_dict['event_features'].append(np.stack([tmp_file['gen_'+feat].array()[:nevts] for feat in var_list if 'wgt' not in feat],-1))
            mask_gen = tmp_file['gen_y'].array()[:nevts]
            mask_evt = (gen_dict['event_features'][ifile][:,0] > 100)
            reco_dict['particle_features'][ifile] = reco_dict['particle_features'][ifile][mask_evt]
            reco_dict['event_features'][ifile] = reco_dict['event_features'][ifile][mask_evt]
            
            gen_dict['particle_features'].append(np.stack([tmp_file['gen_'+feat].array()[:nevts].pad(max_part).fillna(0).regular() for feat in particle_list],-1))
            
            #print("Removing events")
            #Remove events not passing Q2> 100
            gen_dict['particle_features'][ifile] = gen_dict['particle_features'][ifile][mask_evt]
            gen_dict['event_features'][ifile] = gen_dict['event_features'][ifile][mask_evt]
            mask_reco = mask_reco[mask_evt]
            mask_gen = mask_gen[mask_evt]

        print("Number of events: {}".format(mask_reco.shape[0]))
        # 0.08 < y < 0.7, ptmiss < 10, 45 < empz < 65 and Q2 > 150
        pass_reco = (mask_reco[:,0] > 0.08) & (mask_reco[:,0] < 0.7) & (mask_reco[:,1]<10.0) & (mask_reco[:,2] > 45.) & (mask_reco[:,2] < 65) & (reco_dict['event_features'][ifile][:,0] > 150)
        reco_dict['event_features'][ifile] = np.concatenate((reco_dict['event_features'][ifile],pass_reco[:,None]),-1)        
        del mask_reco, pass_reco
        #Particle dataset

        # part pT > 0.1 GeV, -1.5 < part eta < 2.75
        mask_part = (reco_dict['particle_features'][ifile][:,:,0] > 0.1) & (reco_dict['particle_features'][ifile][:,:,1] > -1.5) & (reco_dict['particle_features'][ifile][:,:,1] < 2.75)
        reco_dict['particle_features'][ifile] = reco_dict['particle_features'][ifile]*mask_part[:,:,None]
        mask_evt = np.sum(reco_dict['particle_features'][ifile][:,:,0],1) > 0
        
        del mask_part
        if 'Data' not in name:
            print("Adding Gen info")
            
            # 0.2 < y < 0.7 and Q2 > 150
            pass_gen = (mask_gen > 0.2) & (mask_gen < 0.7) & (gen_dict['event_features'][ifile][:,0] > 150)
            gen_dict['event_features'][ifile] = np.concatenate((gen_dict['event_features'][ifile],pass_gen[:,None]),-1)
            # part pT > 0.5 GeV, -1.5 < part eta < 2.75            
            mask_part = (gen_dict['particle_features'][ifile][:,:,0] > 0.1) & (gen_dict['particle_features'][ifile][:,:,1] > -1.5) & (gen_dict['particle_features'][ifile][:,:,1] < 2.75)
            gen_dict['particle_features'][ifile] = gen_dict['particle_features'][ifile]*mask_part[:,:,None]
            
            #Keep only Q2>100 GeV^2
            mask_evt_gen = (np.sum(gen_dict['particle_features'][ifile][:,:,0],1) > 0) 
            mask_evt*=mask_evt_gen

            gen_dict['particle_features'][ifile] = gen_dict['particle_features'][ifile][mask_evt]
            gen_dict['event_features'][ifile] = gen_dict['event_features'][ifile][mask_evt]
            
            del mask_gen, pass_gen

        print("Rejecting {}".format(1.0 - 1.0*np.sum(mask_evt)/mask_evt.shape[0]))
        reco_dict['particle_features'][ifile] = reco_dict['particle_features'][ifile][mask_evt]
        reco_dict['event_features'][ifile] = reco_dict['event_features'][ifile][mask_evt]

                
    reco_dict['event_features'] = np.concatenate(reco_dict['event_features'])
    reco_dict['particle_features'] = np.concatenate(reco_dict['particle_features'])
    
    # Make sure reco particles that do not pass reco cuts are indeed zero padded
    reco_dict['particle_features'] *= reco_dict['event_features'][:,-1,None,None]
    order = np.argsort(-reco_dict['particle_features'][:,:,0],1)
    reco_dict['particle_features'] = np.take_along_axis(reco_dict['particle_features'],order[:,:,None],1)
    max_nonzero_reco = np.max(np.sum(reco_dict['particle_features'][:,:,0]>0,1))
    reco_dict['particle_features'] = reco_dict['particle_features'][:,:max_nonzero]
    
    print("Maximum reco particle multiplicity",max_nonzero_reco)
    if 'Data' not in name:
        gen_dict['event_features'] = np.concatenate(gen_dict['event_features'])
        gen_dict['particle_features'] = np.concatenate(gen_dict['particle_features'])
        order = np.argsort(-gen_dict['particle_features'][:,:,0],1)
        gen_dict['particle_features'] = np.take_along_axis(gen_dict['particle_features'],order[:,:,None],1)
        max_nonzero_gen = np.max(np.sum(gen_dict['particle_features'][:,:,0]>0,1))
        gen_dict['particle_features'] = gen_dict['particle_features'][:,:max_nonzero]
        print("Maximum gen particle multiplicity",max_nonzero_gen)
    del tmp_file
    return reco_dict,gen_dict

class TFDataset():
    def __init__(self,
                 file_names,
                 base_path,
                 rank=0,
                 size=1,
                 is_mc = False,
                 nmax = None,
                 ):
        
        self.rank = rank
        self.size = size
        self.base_path = base_path
        self.is_mc = is_mc
        self.nmax = nmax
        self.prepare_dataset(file_names)
        self.normalize_weights()


    def normalize_weights(self):
        #print("Total number of reco events {}".format(self.num_pass_reco))
        self.weight= 1e6*self.weight/self.num_pass_reco

        
    def preprocess(self,data):
        p,e = data
        #return (p,e)
        mask = p[:,:,0]!=0
        
        
        #use log(pt/Q), delta_eta, delta_phi        
        log_pt = np.log(1.0+np.ma.divide(p[:,:,0],np.sqrt(e[:,None,0])).filled(0))
        delta_eta = p[:,:,1] + np.ma.arctanh(e[:,None,3]/np.sqrt(e[:,None,1]**2 + e[:,None,2]**2+ e[:,None,3]**2)).filled(0)

        delta_phi = p[:,:,2] -np.pi - np.arctan2(e[:,None,2],e[:,None,3])
        delta_phi[delta_phi>np.pi] -= 2*np.pi
        delta_phi[delta_phi<-np.pi] += 2*np.pi
        new_p = np.stack([log_pt,delta_eta,delta_phi,p[:,:,3]],-1)*mask[:,:,None]

        log_q = np.ma.log(e[:,0]).filled(0)/5.0 -1.0
        new_e = np.stack([log_q,
                          e[:,1]/np.sqrt(e[:,0]),
                          e[:,2]/np.sqrt(e[:,0]),
                          1.0+e[:,3]/np.sqrt(e[:,0])],-1)

        return (np.nan_to_num(new_p,posinf=1,neginf=0),np.nan_to_num(new_e,posinf=1,neginf=0))
        # return tf.data.Dataset.zip((
        #     tf.data.Dataset.from_tensor_slices(new_p),
        #     tf.data.Dataset.from_tensor_slices(new_e)
        # )) 
        #return new_p, new_e

            
    def prepare_dataset(self,file_names):
        ''' Load h5 files containing the data. The structure of the h5 file should be
            reco_particle_features: p_pt,p_eta,p_phi,p_charge (B,N,4)
            reco_event_features   : Q2, e_px, e_py, e_pz, wgt, pass_reco (B,6)
            if MC should also contain
            gen_particle_features : p_pt,p_eta,p_phi,p_charge (B,N,4)
            gen_event_features    : Q2, e_px, e_py, e_pz, pass_gen (B,5)

        '''
        self.num_pass_reco = 0
        for ifile, f in enumerate(file_names):
            if self.rank==0:print("Loading file {}".format(f))
            #Determine the total number of event passing reco for normalization of the weights
            
            
            if self.nmax is None:
                self.nmax = h5.File(os.path.join(self.base_path,f),'r')['reco_event_features'].shape[0]
                
            self.num_pass_reco += np.sum(h5.File(os.path.join(self.base_path,f),'r')['reco_event_features'][:self.nmax,-2][h5.File(os.path.join(self.base_path,f),'r')['reco_event_features'][:self.nmax,-1]==1])
                
            reco_p =  h5.File(os.path.join(self.base_path,f),'r')['reco_particle_features'][self.rank:self.nmax:self.size].astype(np.float32)
            reco_e = h5.File(os.path.join(self.base_path,f),'r')['reco_event_features'][self.rank:self.nmax:self.size].astype(np.float32)


            if ifile ==0:
                self.weight = reco_e[:,-2].astype(np.float32)
                self.pass_reco = reco_e[:,-1] ==1

                reco = (reco_p,reco_e[:,:-2])
                # reco = tf.data.Dataset.zip((
                #     tf.data.Dataset.from_tensor_slices(reco_p),
                #     tf.data.Dataset.from_tensor_slices(reco_e[:,:-2])
                # ))                
                #reco = tf.data.Dataset.from_tensor_slices((reco_p,reco_e[:,:-2]))                
            else:
                self.weight = np.concatenate((self.weight,reco_e[:,-2].astype(np.float32)))
                self.pass_reco = np.concatenate((self.pass_reco==1,reco_e[:,-1]))

                reco = np.concatenate([reco,(reco_p,reco_e[:,:-2])],0)

                # reco.concatenate(tf.data.Dataset.zip((
                #     tf.data.Dataset.from_tensor_slices(reco_p),
                #     tf.data.Dataset.from_tensor_slices(reco_e[:,:-2])
                # )))
                
                #reco.concatenate(tf.data.Dataset.from_tensor_slices((reco_p,reco_e)))
                
            if self.is_mc:
                gen_p = h5.File(os.path.join(self.base_path,f),'r')['gen_particle_features'][self.rank:self.nmax:self.size].astype(np.float32)
                gen_e = h5.File(os.path.join(self.base_path,f),'r')['gen_event_features'][self.rank:self.nmax:self.size].astype(np.float32)
                if ifile ==0:
                    self.pass_gen = gen_e[:,-1] ==1
                    gen = (gen_p,gen_e[:,:-1])
                    #gen = tf.data.Dataset.from_tensor_slices((gen_p,gen_e[:,:-1]))
                    # gen = tf.data.Dataset.zip((
                    #     tf.data.Dataset.from_tensor_slices(gen_p),
                    #     tf.data.Dataset.from_tensor_slices(gen_e[:,:-1])
                    # ))                
                else:
                    self.pass_gen = np.concatenate((self.pass_gen==1,gen_e[:,-1]))
                    gen = np.concatenate([gen,(gen_p,gen_e[:,:-1])],0)
                    #gen.concatenate(tf.data.Dataset.from_tensor_slices((gen_p,gen_e)))
                    # gen.concatenate(tf.data.Dataset.zip((
                    #     tf.data.Dataset.from_tensor_slices(gen_p),
                    #     tf.data.Dataset.from_tensor_slices(gen_e[:,:-1])
                    # )))
            else:
                self.pass_gen = None

        self.reco = self.preprocess(reco)
        assert np.any(np.isnan(reco[0])) == False, "ERROR: NAN in particle dataset"
        assert np.any(np.isnan(reco[1])) == False, "ERROR: NAN in event dataset"

        #self.reco =  self.return_dataset(reco)
        if self.is_mc:
            self.gen = self.preprocess(gen)
            assert np.any(np.isnan(gen[0])) == False, "ERROR: NAN in particle dataset"
            assert np.any(np.isnan(gen[1])) == False, "ERROR: NAN in event dataset"
            #self.gen  =  self.return_dataset(gen)
        else:                
            self.gen = None


def create_toy(data_output):

    nevts = 1000000
    nfeat = 4
    npart = 100
    
    mean1 = 1.0
    std1 = 1.0
    mean2 = 0.8
    std2 = 1.0

    std_smear = 0.1

    def create_gen(nevts,nfeat,npart,mean,std):
        evt = np.random.normal(size=(nevts,nfeat),
                               loc=mean*np.ones((nevts,nfeat)),
                               scale=std*np.ones((nevts,nfeat)))
        part = np.random.normal(size=(nevts,npart,nfeat),
                                loc=mean*np.ones((nevts,npart,nfeat)),
                                scale=std*np.ones((nevts,npart,nfeat)))
        return evt,part

    gen1_evt, gen1_part = create_gen(nevts,nfeat,npart,mean1,std1)
    gen2_evt, gen2_part = create_gen(nevts,nfeat,npart,mean2,std2)

    def smear(sample,std):        
        return std*np.random.normal(size=sample.shape) + sample
    reco1_evt = smear(gen1_evt,std_smear)
    reco1_part = smear(gen1_part,std_smear)

    reco2_evt = smear(gen2_evt,std_smear)
    reco2_part = smear(gen2_part,std_smear)

    #Add mock weights and pass reco flags
    pass_reco1 = np.random.randint(2,size = (nevts,1))
    reco1_evt *= pass_reco1
    pass_reco2 = np.random.randint(2,size = (nevts,1))
    reco2_evt *= pass_reco2
    weights = np.ones((nevts,1))
    reco1_evt = np.concatenate((reco1_evt,weights,pass_reco1),-1)
    reco2_evt = np.concatenate((reco2_evt,weights,pass_reco2),-1)
    gen1_evt = np.concatenate((gen1_evt,weights),-1)
    gen2_evt = np.concatenate((gen2_evt,weights),-1)

    

    with h5.File(os.path.join(data_output,"toy1.h5"),'w') as fh5:
        dset = fh5.create_dataset('reco_particle_features', data=reco1_part)
        dset = fh5.create_dataset('reco_event_features', data=reco1_evt)
        dset = fh5.create_dataset('gen_particle_features', data=gen1_part)
        dset = fh5.create_dataset('gen_event_features', data=gen1_evt)

    with h5.File(os.path.join(data_output,"toy2.h5"),'w') as fh5:
        dset = fh5.create_dataset('reco_particle_features', data=reco2_part)
        dset = fh5.create_dataset('reco_event_features', data=reco2_evt)
        dset = fh5.create_dataset('gen_particle_features', data=gen2_part)
        dset = fh5.create_dataset('gen_event_features', data=gen2_evt)


            

if __name__ == "__main__":
    import argparse
    import logging
    #Convert root files to h5 inputs that are easier to load when training OmniFold
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-input', default='/global/cfs/cdirs/m3246/vmikuni/H1v2/root/', help='Folder containing data and MC files in the root format')
    parser.add_argument('--data-output', default='/pscratch/sd/v/vmikuni/H1v2/h5', help='Output folder containing data and MC files')
    flags = parser.parse_args()

    #create_toy(flags.data_output)
    

    # print("Processing Data")
    # file_list = ['out_em06/Data_Eminus06.root']
    # data,_ = convert_to_np(file_list,flags.data_input,name='Data')
    # with h5.File(os.path.join(flags.data_output,"test_data.h5"),'w') as fh5:
    #     dset = fh5.create_dataset('reco_particle_features', data=data['particle_features'])
    #     dset = fh5.create_dataset('reco_event_features', data=data['event_features'])

    
    # print("Processing Data")
    # file_list = ['out_ep0607/Data_Eplus0607.root','out_em06/Data_Eminus06.root']
    # data,_ = convert_to_np(file_list,flags.data_input,name='Data')
    # with h5.File(os.path.join(flags.data_output,"data.h5"),'w') as fh5:
    #     dset = fh5.create_dataset('reco_particle_features', data=data['particle_features'])
    #     dset = fh5.create_dataset('reco_event_features', data=data['event_features'])
        
    
    # print("Processing Rapgap")    
    # file_list = [
    #     #'out_ep0607/Rapgap_Eplus0607.root',
    #     #'out_em06/Rapgap_Eminus06.root'
    #     'out_ep0607/Rapgap_Eplus0607_101.nominal.root',
    #     'out_ep0607/Rapgap_Eplus0607_102.nominal.root',
    #     'out_ep0607/Rapgap_Eplus0607_103.nominal.root',
    #     'out_ep0607/Rapgap_Eplus0607_104.nominal.root',
    #     'out_ep0607/Rapgap_Eplus0607_105.nominal.root',
    #     'out_ep0607/Rapgap_Eplus0607_106.nominal.root',
    #     'out_ep0607/Rapgap_Eplus0607_107.nominal.root',
    #     'out_ep0607/Rapgap_Eplus0607_108.nominal.root',
    #     'out_ep0607/Rapgap_Eplus0607_109.nominal.root',
    #     'out_ep0607/Rapgap_Eplus0607_10.nominal.root',
    #     'out_ep0607/Rapgap_Eplus0607_110.nominal.root',
    #     'out_ep0607/Rapgap_Eplus0607_111.nominal.root',
    #     'out_ep0607/Rapgap_Eplus0607_112.nominal.root',
    #     'out_ep0607/Rapgap_Eplus0607_113.nominal.root',
    #     'out_ep0607/Rapgap_Eplus0607_114.nominal.root',
    #     'out_ep0607/Rapgap_Eplus0607_115.nominal.root',
    #     'out_ep0607/Rapgap_Eplus0607_116.nominal.root',
    #     'out_ep0607/Rapgap_Eplus0607_117.nominal.root',
    #     'out_ep0607/Rapgap_Eplus0607_118.nominal.root',
    #     'out_ep0607/Rapgap_Eplus0607_119.nominal.root',
    #     'out_ep0607/Rapgap_Eplus0607_11.nominal.root',
    #     'out_ep0607/Rapgap_Eplus0607_120.nominal.root',
    #     'out_ep0607/Rapgap_Eplus0607_121.nominal.root',
    #     'out_ep0607/Rapgap_Eplus0607_122.nominal.root',
    #     'out_ep0607/Rapgap_Eplus0607_123.nominal.root',
    #     'out_ep0607/Rapgap_Eplus0607_124.nominal.root',
    #     'out_ep0607/Rapgap_Eplus0607_125.nominal.root',
    #     'out_ep0607/Rapgap_Eplus0607_126.nominal.root',
    #     'out_ep0607/Rapgap_Eplus0607_127.nominal.root',
    #     'out_ep0607/Rapgap_Eplus0607_128.nominal.root',
    #     'out_ep0607/Rapgap_Eplus0607_129.nominal.root',
    #     'out_ep0607/Rapgap_Eplus0607_12.nominal.root',
    #     'out_ep0607/Rapgap_Eplus0607_130.nominal.root',
    #     'out_ep0607/Rapgap_Eplus0607_131.nominal.root',
    #     'out_ep0607/Rapgap_Eplus0607_132.nominal.root',
    #     'out_ep0607/Rapgap_Eplus0607_133.nominal.root',
    #     'out_ep0607/Rapgap_Eplus0607_134.nominal.root',
    #     'out_ep0607/Rapgap_Eplus0607_135.nominal.root',
    #     'out_ep0607/Rapgap_Eplus0607_136.nominal.root',
    #     'out_ep0607/Rapgap_Eplus0607_137.nominal.root',
    #     'out_ep0607/Rapgap_Eplus0607_138.nominal.root',
    #     'out_ep0607/Rapgap_Eplus0607_139.nominal.root',
    #     'out_ep0607/Rapgap_Eplus0607_140.nominal.root',
    #     'out_ep0607/Rapgap_Eplus0607_141.nominal.root',
    #     'out_ep0607/Rapgap_Eplus0607_142.nominal.root',
    #     'out_ep0607/Rapgap_Eplus0607_143.nominal.root',
    #     'out_ep0607/Rapgap_Eplus0607_144.nominal.root',
    #     'out_ep0607/Rapgap_Eplus0607_145.nominal.root',
    #     'out_ep0607/Rapgap_Eplus0607_146.nominal.root',
    #     'out_ep0607/Rapgap_Eplus0607_147.nominal.root',
    #     'out_ep0607/Rapgap_Eplus0607_148.nominal.root',
    #     'out_ep0607/Rapgap_Eplus0607_149.nominal.root',
    #     'out_ep0607/Rapgap_Eplus0607_150.nominal.root',
    #     'out_ep0607/Rapgap_Eplus0607_1.nominal.root',
    #     'out_ep0607/Rapgap_Eplus0607_2.nominal.root',
    #     'out_ep0607/Rapgap_Eplus0607_3.nominal.root',
    #     'out_ep0607/Rapgap_Eplus0607_4.nominal.root',
    #     'out_ep0607/Rapgap_Eplus0607_5.nominal.root',
    #     'out_ep0607/Rapgap_Eplus0607_6.nominal.root',
    #     'out_ep0607/Rapgap_Eplus0607_7.nominal.root',
    #     'out_ep0607/Rapgap_Eplus0607_8.nominal.root',
    #     'out_ep0607/Rapgap_Eplus0607_9.nominal.root',
    # ]
    # #file_list = ['out_em06/Rapgap_Eminus06_122.nominal.root'] #quick testing
    # reco,gen = convert_to_np(file_list,flags.data_input,name='Rapgap')
    # with h5.File(os.path.join(flags.data_output,"Rapgap_Eplus0607.h5"),'w') as fh5:
    # #with h5.File(os.path.join(flags.data_output,"test_sim.h5"),'w') as fh5:
    #     dset = fh5.create_dataset('reco_particle_features', data=reco['particle_features'])
    #     dset = fh5.create_dataset('reco_event_features', data=reco['event_features'])
    #     dset = fh5.create_dataset('gen_particle_features', data=gen['particle_features'])
    #     dset = fh5.create_dataset('gen_event_features', data=gen['event_features'])
    
    print("Processing Djangoh")
    file_list = [
        #'out_em06/Django_Eminus06.root',
        #'out_ep0607/Django_Eplus0607.root'
        'out_ep0607/Django_Eplus0607_101.nominal.root',
        'out_ep0607/Django_Eplus0607_102.nominal.root',
        'out_ep0607/Django_Eplus0607_103.nominal.root',
        'out_ep0607/Django_Eplus0607_104.nominal.root',
        'out_ep0607/Django_Eplus0607_105.nominal.root',
        'out_ep0607/Django_Eplus0607_106.nominal.root',
        'out_ep0607/Django_Eplus0607_107.nominal.root',
        'out_ep0607/Django_Eplus0607_108.nominal.root',
        'out_ep0607/Django_Eplus0607_109.nominal.root',
        'out_ep0607/Django_Eplus0607_10.nominal.root',
        'out_ep0607/Django_Eplus0607_110.nominal.root',
        'out_ep0607/Django_Eplus0607_111.nominal.root',
        'out_ep0607/Django_Eplus0607_112.nominal.root',
        'out_ep0607/Django_Eplus0607_113.nominal.root',
        'out_ep0607/Django_Eplus0607_114.nominal.root',
        'out_ep0607/Django_Eplus0607_115.nominal.root',
        'out_ep0607/Django_Eplus0607_116.nominal.root',
        'out_ep0607/Django_Eplus0607_117.nominal.root',
        'out_ep0607/Django_Eplus0607_118.nominal.root',
        'out_ep0607/Django_Eplus0607_119.nominal.root',
        'out_ep0607/Django_Eplus0607_11.nominal.root',
        'out_ep0607/Django_Eplus0607_120.nominal.root',
        'out_ep0607/Django_Eplus0607_121.nominal.root',
        'out_ep0607/Django_Eplus0607_122.nominal.root',
        'out_ep0607/Django_Eplus0607_123.nominal.root',
        'out_ep0607/Django_Eplus0607_124.nominal.root',
        'out_ep0607/Django_Eplus0607_125.nominal.root',
        'out_ep0607/Django_Eplus0607_126.nominal.root',
        'out_ep0607/Django_Eplus0607_127.nominal.root',
        'out_ep0607/Django_Eplus0607_128.nominal.root',
        'out_ep0607/Django_Eplus0607_129.nominal.root',
        'out_ep0607/Django_Eplus0607_12.nominal.root',
        'out_ep0607/Django_Eplus0607_130.nominal.root',
        'out_ep0607/Django_Eplus0607_131.nominal.root',
        'out_ep0607/Django_Eplus0607_132.nominal.root',
        'out_ep0607/Django_Eplus0607_133.nominal.root',
        'out_ep0607/Django_Eplus0607_134.nominal.root',
        'out_ep0607/Django_Eplus0607_135.nominal.root',
        'out_ep0607/Django_Eplus0607_136.nominal.root',
        'out_ep0607/Django_Eplus0607_137.nominal.root',
        'out_ep0607/Django_Eplus0607_138.nominal.root',
        'out_ep0607/Django_Eplus0607_139.nominal.root',
        'out_ep0607/Django_Eplus0607_13.nominal.root',
        'out_ep0607/Django_Eplus0607_141.nominal.root',
        'out_ep0607/Django_Eplus0607_142.nominal.root',
        'out_ep0607/Django_Eplus0607_143.nominal.root',
        'out_ep0607/Django_Eplus0607_144.nominal.root',
        'out_ep0607/Django_Eplus0607_145.nominal.root',
        'out_ep0607/Django_Eplus0607_146.nominal.root',
        'out_ep0607/Django_Eplus0607_147.nominal.root',
        'out_ep0607/Django_Eplus0607_148.nominal.root',
        'out_ep0607/Django_Eplus0607_149.nominal.root',
        'out_ep0607/Django_Eplus0607_14.nominal.root',
        'out_ep0607/Django_Eplus0607_151.nominal.root',
        'out_ep0607/Django_Eplus0607_152.nominal.root',
        'out_ep0607/Django_Eplus0607_153.nominal.root',
        'out_ep0607/Django_Eplus0607_154.nominal.root',
        'out_ep0607/Django_Eplus0607_15.nominal.root',
        'out_ep0607/Django_Eplus0607_16.nominal.root',
        'out_ep0607/Django_Eplus0607_17.nominal.root',
        'out_ep0607/Django_Eplus0607_18.nominal.root',
        'out_ep0607/Django_Eplus0607_19.nominal.root',
        'out_ep0607/Django_Eplus0607_1.nominal.root',
        'out_ep0607/Django_Eplus0607_20.nominal.root',
        'out_ep0607/Django_Eplus0607_2.nominal.root',
        'out_ep0607/Django_Eplus0607_3.nominal.root',
        'out_ep0607/Django_Eplus0607_4.nominal.root',
        'out_ep0607/Django_Eplus0607_5.nominal.root',
        'out_ep0607/Django_Eplus0607_6.nominal.root',
        'out_ep0607/Django_Eplus0607_7.nominal.root',
        'out_ep0607/Django_Eplus0607_8.nominal.root',
        'out_ep0607/Django_Eplus0607_9.nominal.root',


    ]
    reco,gen = convert_to_np(file_list,flags.data_input,name='Django')
    with h5.File(os.path.join(flags.data_output,"Djangoh_Eplus0607.h5"),'w') as fh5:
        dset = fh5.create_dataset('reco_particle_features', data=reco['particle_features'])
        dset = fh5.create_dataset('reco_event_features', data=reco['event_features'])
        dset = fh5.create_dataset('gen_particle_features', data=gen['particle_features'])
        dset = fh5.create_dataset('gen_event_features', data=gen['event_features'])
    
    
