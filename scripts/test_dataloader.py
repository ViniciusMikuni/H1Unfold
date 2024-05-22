from dataloader import Dataset
import numpy as np
import tensorflow as tf
import utils
import os
utils.SetStyle()

if __name__ == "__main__":
    #base_path = '/global/cfs/cdirs/m3246/vmikuni/H1v2/h5/'
    base_path = '/pscratch/sd/v/vmikuni/H1v2/h5'
    #file_mc = ['Rapgap_Eminus06.h5']
    #file_mc = ['Rapgap_Eplus0607.h5']
    
    #file_data = ['Data.h5']
    file_mc = ['test_sim.h5']
    file_data = ['data.h5']
    # file_mc = ['toy1.h5']
    # file_data = ['toy2.h5']
    dataloader_mc = Dataset(file_mc,base_path,is_mc=True)
    
    dataloader_data = Dataset(file_data,base_path,is_mc=False,norm=dataloader_mc.nmax)
    print("Loaded {} data events".format(dataloader_data.reco[0].shape[0]))
    if not os.path.exists('../plots'):
        os.makedirs('../plots')
        
    #Let's make some plots
    particles_mc, events_mc,_,mask_mc = dataloader_mc.reco    
    particles_gen, events_gen,_,_ = dataloader_mc.gen
    #Undo the preprocessing
    particles_mc, events_mc = dataloader_mc.revert_standardize(particles_mc, events_mc,mask_mc)

    

    print("Using {} particles for reco mc".format(particles_mc.shape[1]))
    print("Using {} particles for gen mc".format(particles_gen.shape[1]))
    pass_reco_mc = dataloader_mc.pass_reco
    print("Sum of weights MC: {}".format(np.sum(dataloader_mc.weight[pass_reco_mc])))
    print("Pass reco MC: {}".format(1.0*np.sum(pass_reco_mc)/pass_reco_mc.shape[0]))
    print("Pass gen fid MC: {}".format(1.0*np.sum(dataloader_mc.pass_gen)/pass_reco_mc.shape[0]))
    print("Pass fid but not pass reco {}".format(1.0*np.sum(dataloader_mc.pass_gen[pass_reco_mc==0])/np.sum(dataloader_mc.pass_gen)))
    print("Pass fid  pass reco {}".format(1.0*np.sum(dataloader_mc.pass_gen[pass_reco_mc])/np.sum(dataloader_mc.pass_gen)))
    print("Not pass fid but pass reco {}".format(1.0*np.sum(dataloader_mc.pass_gen[pass_reco_mc]==0)/np.sum(dataloader_mc.pass_gen==0)))
    
    print("Pass reco to pass fiducial ratio: {}".format(1.0*np.sum(pass_reco_mc)/np.sum(dataloader_mc.pass_gen)))
    particles_mc = particles_mc[pass_reco_mc]
    events_mc = events_mc[pass_reco_mc]
    
    wgt = dataloader_mc.weight[pass_reco_mc]

    particles_data,events_data,_,mask_data  = dataloader_data.reco
    particles_data, events_data = dataloader_data.revert_standardize(particles_data, events_data,mask_data)
    
    print("Using {} particles for data".format(particles_data.shape[1]))
    pass_reco_data = dataloader_data.pass_reco
    print("Sum of weights data: {}".format(np.sum(dataloader_data.weight[pass_reco_data])))
    print("Total data: {}".format(pass_reco_data.shape[0]))
    print("Pass reco data: {}".format(1.0*np.sum(pass_reco_data)/pass_reco_data.shape[0]))
    particles_data = particles_data[pass_reco_data]
    events_data = events_data[pass_reco_data]

    
    for feature in range(events_data.shape[-1]):
        feed_dict = {
            'data': events_data[:,feature],
            'Rapgap reco': events_mc[:,feature],
            #'Rapgap gen': events_gen[:,feature],            
        }
        weights = {
            'data':np.ones(events_data.shape[0]),
            'Rapgap reco': wgt,
            #'Rapgap gen': np.array(wgt),
            }
        fig,ax = utils.HistRoutine(feed_dict,
                                   xlabel=utils.event_names[str(feature)],
                                   weights = weights,
                                   label_loc='upper left',
                                   )
        fig.savefig('../plots/event_{}.pdf'.format(feature))
    #Flatten for plotting
    particles_data = particles_data.reshape((-1,particles_data.shape[-1]))
    particles_data = particles_data[particles_data[:,0]!=0]

    wgt = np.tile(wgt[:,None],(1,particles_mc.shape[1]))
    wgt = wgt.reshape(-1,1)
    particles_mc = particles_mc.reshape((-1,particles_mc.shape[-1]))
    mask_zero = particles_mc[:,0]!=0
    particles_mc = particles_mc[mask_zero]
    wgt = wgt[mask_zero]
    for feature in range(particles_data.shape[-1]):
        feed_dict = {
            'data': particles_data[:,feature],
            'Rapgap reco': particles_mc[:,feature],            
        }

        weights = {
            'data':np.ones(particles_data.shape[0]),
            'Rapgap reco': wgt,
            #'Rapgap gen': np.array(wgt),
        }
        
        fig,ax = utils.HistRoutine(feed_dict,
                                   xlabel=utils.particle_names[str(feature)],
                                   label_loc='upper left',
                                   #weights = weights,
                                   )
        fig.savefig('../plots/part_{}.pdf'.format(feature))
    
