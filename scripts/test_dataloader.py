from dataloader import TFDataset
import numpy as np
import tensorflow as tf
import utils
utils.SetStyle()

if __name__ == "__main__":
    base_path = '/global/cfs/cdirs/m3246/vmikuni/H1v2/h5/'
    file_mc = ['test_sim.h5']
    file_data = ['test_data.h5']
    dataloader_mc = TFDataset(file_mc,base_path,is_mc=True)
    dataloader_data = TFDataset(file_data,base_path,is_mc=False)

    
    #Let's make some plots
    particles_mc, events_mc = dataloader_mc.reco
    pass_reco_mc = dataloader_mc.pass_reco
    print("Acceptance MC: {}".format(1.0*np.sum(pass_reco_mc)/pass_reco_mc.shape[0]))
    particles_mc = particles_mc[pass_reco_mc]
    events_mc = events_mc[pass_reco_mc]
    
    wgt = dataloader_mc.weight

    particles_data, events_data  = dataloader_data.reco
    pass_reco_data = dataloader_data.pass_reco
    print("Acceptance data: {}".format(1.0*np.sum(pass_reco_data)/pass_reco_data.shape[0]))
    particles_data = particles_data[pass_reco_data]
    events_data = events_data[pass_reco_data]
    
    for feature in range(events_data.shape[-1]):
        feed_dict = {
            'data': events_data[:,feature],
            'mc': events_mc[:,feature],            
        }
        weights = {
            'data':np.ones(events_data.shape[0]),
            'mc': np.array(wgt)[pass_reco_mc],
            }
        fig,ax = utils.HistRoutine(feed_dict,
                                   xlabel=utils.event_names[str(feature)],
                                   weights = weights,
                                   )
        fig.savefig('../plots/event_{}.pdf'.format(feature))
    #Flatten for plotting
    particles_data = particles_data.reshape((-1,particles_data.shape[-1]))
    particles_data = particles_data[particles_data[:,0]!=0]
    
    particles_mc = particles_mc.reshape((-1,particles_mc.shape[-1]))
    particles_mc = particles_mc[particles_mc[:,0]!=0]
    
    for feature in range(particles_data.shape[-1]):
        feed_dict = {
            'data': particles_data[:,feature],
            'mc': particles_mc[:,feature],            
        }
        # weights = {
        #     'data':np.ones(events_data.shape[0]),
        #     'mc': np.array(wgt)[pass_reco_mc],
        #     }
        fig,ax = utils.HistRoutine(feed_dict,
                                   xlabel=utils.particle_names[str(feature)],
                                   #weights = weights,
                                   )
        fig.savefig('../plots/part_{}.pdf'.format(feature))


        

    # for entry in dataloader.reco:
    #     print(entry[0].numpy())
    #     input()
    #print(list(dataloader.data.as_numpy_iterator()))
    
