import numpy as np
import argparse
import h5py as h5
import os

def preprocess(data):
    p,e = data
    #return (p,e)
    mask = p[:,:,0]!=0
    
    #use log(pt/Q), delta_eta, delta_phi        
    log_pt_rel = np.ma.log(np.ma.divide(p[:,:,0],np.sqrt(e[:,None,0])).filled(0)).filled(0)
    log_pt = np.ma.log(p[:,:,0]).filled(0)
    
    log_e_rel = np.ma.log(np.ma.divide(p[:,:,0]*np.cosh(p[:,:,1]),
                                       np.sqrt(e[:,None,0])).filled(0)).filled(0)
    log_e = np.ma.log(p[:,:,0]*np.cosh(p[:,:,1])).filled(0)
    
    delta_eta = p[:,:,1] - np.ma.arctanh(e[:,None,4]/np.sqrt(e[:,None,2]**2 + e[:,None,3]**2+ e[:,None,4]**2)).filled(0)        
    delta_phi = p[:,:,2] - np.pi - np.arctan2(e[:,None,3],e[:,None,2])
    delta_phi[delta_phi>np.pi] -= 2*np.pi
    delta_phi[delta_phi<-np.pi] += 2*np.pi
    delta_r = np.hypot(delta_eta,delta_phi + np.pi)
    new_p = np.stack([delta_eta,
                      delta_phi,
                      log_pt,
                      log_pt_rel,
                      log_e_rel,
                      log_e,
                      delta_r,
                      p[:,:,3]],-1)*mask[:,:,None]
    
    log_Q2 = np.ma.log(e[:,0]).filled(0)
    new_e = np.stack([log_Q2,
                      e[:,1],
                      np.sqrt(e[:,2]**2 + e[:,3]**2)/np.sqrt(e[:,0]),
                      np.ma.arctanh(e[:,4]/np.sqrt(e[:,2]**2 + e[:,3]**2+ e[:,4]**2)).filled(0),
                      np.arctan2(e[:,3],e[:,2])],-1)

    return new_e, new_p*mask[:,:,None]



parser = argparse.ArgumentParser()
parser.add_argument('--data_folder', default='/pscratch/sd/v/vmikuni/H1v2/h5', help='Folder containing data and MC files')
parser.add_argument('--file_name', default='data.h5', help='File to load')

flags = parser.parse_args()

is_mc = False if 'data' in flags.file_name else True

reco_p =  h5.File(os.path.join(flags.data_folder,flags.file_name),'r')['reco_particle_features'][:].astype(np.float32)
reco_e = h5.File(os.path.join(flags.data_folder,flags.file_name),'r')['reco_event_features'][:].astype(np.float32)
weight = reco_e[:,-2].astype(np.float32)
pass_reco = reco_e[:,-1]
print(f"Running {weight.shape[0]} Reco events")
reco_e,reco_p = preprocess((reco_p,reco_e[:,:-2]))
reco_e = np.concatenate([reco_e,weight[:,None],pass_reco[:,None]],-1)

if is_mc:
    print("Running Gen events")
    gen_p =  h5.File(os.path.join(flags.data_folder,flags.file_name),'r')['gen_particle_features'][:].astype(np.float32)
    gen_e = h5.File(os.path.join(flags.data_folder,flags.file_name),'r')['gen_event_features'][:].astype(np.float32)
    pass_gen = gen_e[:,-1]
    gen_e,gen_p = preprocess((gen_p,gen_e[:,:-1]))
    gen_e = np.concatenate([gen_e,pass_gen[:,None]],-1)
    

with h5.File(os.path.join(flags.data_folder,flags.file_name.replace(".h5","_prep.h5")),'w') as fh5:
    dset = fh5.create_dataset('reco_particle_features', data=reco_p)
    dset = fh5.create_dataset('reco_event_features', data=reco_e)
    if is_mc:
        dset = fh5.create_dataset('gen_particle_features', data=gen_p)
        dset = fh5.create_dataset('gen_event_features', data=gen_e)

    

