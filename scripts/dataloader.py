import numpy as np
import tensorflow as tf

import os
import h5py as h5
import gc

def convert_to_np(file_list,base_path,name,is_data = False,
                  max_part = 191, #Overall maximum
                  max_nonzero=132,#Maximum number after applying the per-particle selection
                  nevts=30000000):
    reco_dict = {
        'event_features':[],
        'particle_features':[],
    }
    gen_dict = {
            'event_features':[],
            'particle_features':[],
        }
    var_list = ['Q2','y','e_px','e_py','e_pz','wgt']
    mask_list = ['y','ptmiss','Empz'] #Variables used only to determine the selection but not used during unfolding
    particle_list = ['part_pt','part_eta','part_phi','part_charge']
    
    for ifile,f in enumerate(file_list):
        print("evaluating file {}".format(f))
        tmp_file = uproot.open(os.path.join(base_path,f))['{}/minitree'.format(name)]
        print("loaded file")
        
        reco_dict['event_features'].append(np.stack([tmp_file[feat].array()[:nevts] for feat in var_list],-1))
        reco_dict['particle_features'].append(np.stack([tmp_file[feat].array()[:nevts].pad(max_part).fillna(0).regular() for feat in particle_list],-1))
        mask_reco = np.stack([tmp_file[feat].array()[:nevts] for feat in mask_list],-1)
        if 'Data' not in name:            
            gen_dict['event_features'].append(np.stack([tmp_file['gen_'+feat].array()[:nevts] for feat in var_list if 'wgt' not in feat],-1))
            mask_gen = tmp_file['gen_y'].array()[:nevts]
            #Keep only Q2>100 GeV^2
            mask_evt = (gen_dict['event_features'][ifile][:,0] > 100)
            reco_dict['particle_features'][ifile] = reco_dict['particle_features'][ifile][mask_evt]
            reco_dict['event_features'][ifile] = reco_dict['event_features'][ifile][mask_evt]
            
            gen_dict['particle_features'].append(np.stack([tmp_file['gen_'+feat].array()[:nevts].pad(max_part).fillna(0).regular() for feat in particle_list],-1))
            
            #Set charge to 0 for gen particles outside the tracker acceptance

            gen_dict['particle_features'][-1][:,:,-1] *= np.abs(gen_dict['particle_features'][-1][:,:,1])<2.0
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
            # part pT > 0.1 GeV, -1.5 < part eta < 2.75            
            mask_part = (gen_dict['particle_features'][ifile][:,:,0] > 0.1) & (gen_dict['particle_features'][ifile][:,:,1] > -1.5) & (gen_dict['particle_features'][ifile][:,:,1] < 2.75)
            gen_dict['particle_features'][ifile] = gen_dict['particle_features'][ifile]*mask_part[:,:,None]
            
            #Reject events in case there's no particle left
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

def find_files_with_string(directory, string):
    matching_files = []
    for filename in os.listdir(directory):
        if string in filename:
            matching_files.append(filename)
    return matching_files


class Dataset():
    def __init__(self,
                 file_names,
                 base_path,
                 rank=0,
                 size=1,
                 is_mc = False,
                 nmax = None,
                 norm = None,
                 pass_fiducial=False,
                 pass_reco = False,
                 ):
        
        self.rank = rank
        self.size = size
        self.base_path = base_path
        self.is_mc = is_mc
        self.nmax = nmax

        #Preprocessing parameters

        #2.1070838
        self.mean_part = [0.0 , 0.0, -0.761949, -3.663438,
                          -2.8690917,0.03239748, 3.9436243, 0.0]
        self.std_part =  [1.0, 1.0, 1.0133458, 1.03931,
                          1.0040112, 0.98908925, 1.2256976, 1.0] 

        self.mean_event =  [ 6.4188385 ,  0.3331013 ,  0.8914633 , -0.8352072,  -0.07296985]
        self.std_event  = [0.97656405, 0.1895471 , 0.14934653, 0.4191545 , 1.734126]
        

        self.prepare_dataset(file_names,pass_fiducial,pass_reco)
        self.normalize_weights(self.nmax if norm is None else norm)


    def normalize_weights(self,norm):
        #print("Total number of reco events {}".format(self.num_pass_reco))
        self.weight= (norm*self.weight/self.num_pass_reco).astype(np.float32)

    def standardize(self,data):
        new_p,new_e = data
        mask = new_p[:,:,2]!=0
        p = mask[:,:,None]*(new_p-self.mean_part)/self.std_part
        e = (new_e-self.mean_event)/self.std_event
        return p,e, mask

    def revert_standardize(self,new_p,new_e,mask):
        p = new_p*self.std_part + self.mean_part
        e = new_e*self.std_event + self.mean_event
        return p*mask[:,:,None],e
    
    def concatenate(self,data_list):
        data_part1 = [item[0] for item in data_list]  # Extracting all (M, P, Q) arrays
        data_part2 = [item[1] for item in data_list]  # Extracting all (M, F) arrays

        # Concatenate along the first axis (N * M)
        concatenated_part1 = np.concatenate(data_part1, axis=0)
        concatenated_part2 = np.concatenate(data_part2, axis=0)
        del data_list
        gc.collect()
        return concatenated_part1, concatenated_part2
            
    def prepare_dataset(self,file_names,pass_fiducial,pass_reco):
        ''' Load h5 files containing the data. The structure of the h5 file should be
            reco_particle_features: p_pt,p_eta,p_phi,p_charge (B,N,4)
            reco_event_features   : Q2, e_px, e_py, e_pz, wgt, pass_reco (B,6)
            if MC should also contain
            gen_particle_features : p_pt,p_eta,p_phi,p_charge (B,N,4)
            gen_event_features    : Q2, e_px, e_py, e_pz, pass_gen (B,5)

        '''
        self.num_pass_reco = 0
        self.weight = []
        self.pass_reco = []
        self.pass_gen = []
        reco = []
        gen = []
        for ifile, f in enumerate(file_names):
            if self.rank==0:print("Loading file {}".format(f))
            #Determine the total number of event passing reco for normalization of the weights
                        
            if self.nmax is None:
                self.nmax = h5.File(os.path.join(self.base_path,f),'r')['reco_event_features'].shape[0]

            #Sum of weighted events for collisions passing the reco cuts
            self.num_pass_reco += np.sum(h5.File(os.path.join(self.base_path,f),'r')['reco_event_features'][:self.nmax,-2][h5.File(os.path.join(self.base_path,f),'r')['reco_event_features'][:self.nmax,-1]==1])
            
            reco_p =  h5.File(os.path.join(self.base_path,f),'r')['reco_particle_features'][self.rank:self.nmax:self.size].astype(np.float32)
            reco_e = h5.File(os.path.join(self.base_path,f),'r')['reco_event_features'][self.rank:self.nmax:self.size].astype(np.float32)
            
            self.weight.append(reco_e[:,-2].astype(np.float32))
            self.pass_reco.append(reco_e[:,-1] ==1)
            if pass_reco:
                reco_p = reco_p[self.pass_reco[-1]]
                reco_e = reco_e[self.pass_reco[-1]]
                self.weight[-1] = self.weight[-1][self.pass_reco[-1]]
                self.pass_reco[-1] = self.pass_reco[-1][self.pass_reco[-1]]
            reco.append((reco_p,reco_e[:,:-2]))
                
            if self.is_mc:
                gen_p = h5.File(os.path.join(self.base_path,f),'r')['gen_particle_features'][self.rank:self.nmax:self.size].astype(np.float32)
                gen_e = h5.File(os.path.join(self.base_path,f),'r')['gen_event_features'][self.rank:self.nmax:self.size].astype(np.float32)

                self.pass_gen.append(gen_e[:,-1] ==1)
                if pass_fiducial:
                    gen_p = gen_p[self.pass_gen[-1]]
                    gen_e = gen_e[self.pass_gen[-1]]
                    self.weight[-1] = self.weight[-1][self.pass_gen[-1]]
                    self.pass_gen[-1] = self.pass_gen[-1][self.pass_gen[-1]]
                
                gen.append((gen_p,gen_e[:,:-1]))
            else:
                self.pass_gen = None

        self.weight = np.concatenate(self.weight)
        self.pass_reco = np.concatenate(self.pass_reco)

        #self.reco = self.preprocess(self.concatenate(reco))
        self.reco = self.standardize(self.concatenate(reco))
        del reco
        gc.collect()
        assert np.any(np.isnan(self.reco[0])) == False, "ERROR: NAN in particle dataset"
        assert np.any(np.isnan(self.reco[1])) == False, "ERROR: NAN in event dataset"

        #self.reco =  self.return_dataset(reco)
        if self.is_mc:
            self.pass_gen = np.concatenate(self.pass_gen)
            self.gen = self.standardize(self.concatenate(gen))
            del gen
            gc.collect()
            assert np.any(np.isnan(self.gen[0])) == False, "ERROR: NAN in particle dataset"
            assert np.any(np.isnan(self.gen[1])) == False, "ERROR: NAN in event dataset"
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
    #import uproot
    import uproot3 as uproot
    #Convert root files to h5 inputs that are easier to load when training OmniFold
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-input', default='/global/cfs/cdirs/m3246/vmikuni/H1v2/root/', help='Folder containing data and MC files in the root format')
    parser.add_argument('--data-output', default='/pscratch/sd/v/vmikuni/H1v2/h5', help='Output folder containing data and MC files')

    parser.add_argument('--sample', default='data', help='Sample to process. Options are: DjangohEp, DjangohEm,RapgapEp,RapgapEm,data')
    flags = parser.parse_args()

    #create_toy(flags.data_output)
    

    if flags.sample == 'data':
        print("Processing Data")        
        file_list = ['out_ep0607/Data_Eplus0607.root',
                     #'out_em06/Data_Eminus06_0.nominal.root',
                     #'out_em06/Data_Eminus06_1.nominal.root'
                     ]
        data,_ = convert_to_np(file_list,flags.data_input,name='Data')
        with h5.File(os.path.join(flags.data_output,"data.h5"),'w') as fh5:
            dset = fh5.create_dataset('reco_particle_features', data=data['particle_features'])
            dset = fh5.create_dataset('reco_event_features', data=data['event_features'])
            
    elif flags.sample == 'RapgapEp':
        print("Processing Rapgap")
        file_list = find_files_with_string(flags.data_input+'/out_ep0607', 'Rapgap_Eplus0607_')
        #file_list = ["/global/cfs/cdirs/m3246/vmikuni/H1v2/root/out_ep0607/Rapgap_Eplus0607_4.nominal.root"]
        reco,gen = convert_to_np(file_list,flags.data_input+'/out_ep0607',name='Rapgap')
        with h5.File(os.path.join(flags.data_output,"Rapgap_Eplus0607.h5"),'w') as fh5:
            dset = fh5.create_dataset('reco_particle_features', data=reco['particle_features'])
            dset = fh5.create_dataset('reco_event_features', data=reco['event_features'])
            dset = fh5.create_dataset('gen_particle_features', data=gen['particle_features'])
            dset = fh5.create_dataset('gen_event_features', data=gen['event_features'])
    elif flags.sample == 'DjangohEp':
        print("Processing Djangoh")
        file_list = find_files_with_string(flags.data_input+'/out_ep0607', 'Django_Eplus0607_')
        reco,gen = convert_to_np(file_list,flags.data_input+'/out_ep0607',name='Django')
        with h5.File(os.path.join(flags.data_output,"Djangoh_Eplus0607.h5"),'w') as fh5:
            dset = fh5.create_dataset('reco_particle_features', data=reco['particle_features'])
            dset = fh5.create_dataset('reco_event_features', data=reco['event_features'])
            dset = fh5.create_dataset('gen_particle_features', data=gen['particle_features'])
            dset = fh5.create_dataset('gen_event_features', data=gen['event_features'])

    elif flags.sample == 'DjangohEm':
        print("Processing Djangoh")
        file_list = find_files_with_string(flags.data_input+'/out_em06', 'Django_Eminus06_')
        reco,gen = convert_to_np(file_list,flags.data_input+'/out_em06',name='Django')
        with h5.File(os.path.join(flags.data_output,"Djangoh_Eminus06.h5"),'w') as fh5:
            dset = fh5.create_dataset('reco_particle_features', data=reco['particle_features'])
            dset = fh5.create_dataset('reco_event_features', data=reco['event_features'])
            dset = fh5.create_dataset('gen_particle_features', data=gen['particle_features'])
            dset = fh5.create_dataset('gen_event_features', data=gen['event_features'])
            

    elif flags.sample == 'RapgapEm':
        print("Processing Rapgap")    
        file_list = find_files_with_string(flags.data_input+'/out_em06', 'Rapgap_Eminus06_')
        ###file_list = ['Rapgap_Eminus06_122.nominal.root'] #quick testing
        reco,gen = convert_to_np(file_list,flags.data_input+'/out_em06',name='Rapgap')
        with h5.File(os.path.join(flags.data_output,"Rapgap_Eminus06.h5"),'w') as fh5:
            ###with h5.File(os.path.join(flags.data_output,"test_sim.h5"),'w') as fh5:
            dset = fh5.create_dataset('reco_particle_features', data=reco['particle_features'])
            dset = fh5.create_dataset('reco_event_features', data=reco['event_features'])
            dset = fh5.create_dataset('gen_particle_features', data=gen['particle_features'])
            dset = fh5.create_dataset('gen_event_features', data=gen['event_features'])

    else:
        raise Exception("ERROR: Sample not recognized") 
