import numpy as np

colors = {
    'data':'black',
    'mc':'darkorange',
    'LO':'b', 
    'NLO':'g',
    'NNLO':'r', 
    'Pythia_Vincia': '#9467bd',
    'Pythia_Dire': 'indigo',
    'Pythia_Dire_hadOff': 'indigo',
    'Pythia':'blueviolet',
    'Djangoh':'#8c564b',
    'Rapgap':'darkorange',
    'Rapgap_unfolded':'darkolivegreen',
    'Rapgap_closure':'darkolivegreen',
    'Herwig':'crimson',
    'Herwig_Matchbox':'#a50f15',
    'Herwig_Merging':'#fc9272',
    'Cascade':'b',
    'Sherpa2Lund':'#006837',
    'Sherpa2Cluster':'#addd8e',
    'Sherpa3NLO':'#31a354',
    'Rapgap reco':'#7570b3',
    'Rapgap gen':'darkorange',
}


markers = {
    'Djangoh':'P',
    'Rapgap':'X',
    'Rapgap_unfolded':'X',
    'Rapgap_closure':'X',
    'mc':'X',    

    'Pythia': '^',
    'Pythia_Vincia': '^',
    'Pythia_Dire': '^',
    'Pythia_Dire_hadOff': 'v',
    'Herwig':'D',
    'Herwig_Matchbox':'D',
    'Herwig_Merging':'D',
    'Sherpa2Cluster':'o',
    'Sherpa2Lund':'o',
    'Sherpa3NLO':'o',

    'Rapgap reco':'X',
    'Rapgap gen':'o',
}

#Shift in x-axis for visualization
xaxis_disp = {
    'Pythia_Vincia': -0.7,
    'Pythia_Dire': -0.6,
    'Pythia_Dire_hadOff': -0.6,
    'Pythia': -0.5,
    'Herwig':-0.3,
    'Herwig_Matchbox':-0.2,
    'Herwig_Merging':-0.1,
    'Djangoh':0.1,
    'Rapgap':0.2,
    'Rapgap_unfolded':0.3,
    'Sherpa2Lund':0.5,
    'Sherpa2Cluster':0.6,
    'Sherpa3NLO':0.7,
}
    

dedicated_binning = {
    
    'genjet_pt': np.logspace(np.log10(10),np.log10(100),7),
    'genjet_eta':np.linspace(-1,2.5,6),
    'genjet_phi':np.linspace(-3.14,3.14,8),

    'jet_pt': np.logspace(np.log10(10),np.log10(100),7),
    'jet_eta':np.linspace(-1,2.5,8),
    'jet_phi':np.linspace(-3.14,3.14,8),

    'gen_eec':np.linspace(-1,1,20),
    'eec':np.linspace(-1,1,20),


    'gen_jet_charge':np.array([-1.00,-0.88,-0.76,-0.62,-0.47,-0.33,-0.21,-0.11,-0.03,0.03,0.11,0.19,0.29,0.41,0.56,0.70,0.84,1.00]),
    'gen_jet_tau10':np.array([-4.00,-3.15,-2.59,-2.18,-1.86,-1.58,-1.29,-1.05,-0.81,-0.61,0.00]),
    'gen_jet_tau15':np.array([-5.00,-3.99,-3.28,-2.78,-2.32,-1.92,-1.57,-1.21,-0.91,0.00]),
    'gen_jet_tau20':np.array([-6.00,-4.61,-3.76,-3.09,-2.55,-2.06,-1.58,-1.15,0.00]),    
    'gen_jet_ptD':np.array([0.00,0.32,0.39,0.46,0.55,0.64,0.74,0.85,0.94,1.00,]),
    'gen_jet_ncharged':np.linspace(1,15-1e-8,15),    
    'gen_Q2':np.logspace(np.log10(150),np.log10(5000), 5),
    'jet_ncharged':np.linspace(1,15-1e-8,15),
    
    'jet_charge':np.array([-1.00,-0.88,-0.76,-0.62,-0.47,-0.33,-0.21,-0.11,-0.03,0.03,0.11,0.19,0.29,0.41,0.56,0.70,0.84,1.00]),
    'jet_tau10':np.array([-4.00,-3.15,-2.59,-2.18,-1.86,-1.58,-1.29,-1.05,-0.81,-0.61,0.00]),
    'jet_tau15':np.array([-5.00,-3.99,-3.28,-2.78,-2.32,-1.92,-1.57,-1.21,-0.91,0.00]),
    'jet_tau20':np.array([-6.00,-4.61,-3.76,-3.09,-2.55,-2.06,-1.58,-1.15,0.00]),
    'jet_ptD':np.array([0.00,0.32,0.39,0.46,0.55,0.64,0.74,0.85,0.94,1.00,]),
    
    'npart':np.linspace(0,30,5),
    'e_pt':np.linspace(5,50,5),
    'e_theta':np.linspace(0.4,2.7,8),

    'Q2':np.logspace(np.log10(150),np.log10(5000), 5),
}

fixed_yaxis = {
    'gen_jet_ncharged':0.3, 
    'gen_jet_charge':3.0,
    'gen_jet_tau10':1.3,
    'gen_jet_tau15':0.9,
    'gen_jet_tau20':0.7,
    'gen_jet_ptD':6.5,
    'jet_ncharged':0.3, 
    'jet_charge':3.5,
    'jet_tau10':0.9,
    'jet_tau15':0.8,
    'jet_tau20':0.7,
    'jet_ptD':6.5,
    'npart':0.15,
    'e_pt':0.15,
    'e_theta':2.15,

    'jet_eta':0.7,
    'genjet_pt':12,
    'gen_Q2':12,
    }

sys_sources = {
    'sys_0':'#66c2a5',
    'sys_1':'#fc8d62',
    'sys_5':'#a6d854',
    'sys_7':'#ffd92f',
    'sys_11':'#8da0cb',
    # 'QED': '#8c564b',
    'model':'#e78ac3',
    'closure': '#e5c494',
    'stat':'#808000'

}

sys_translate = {
    'sys_0':"HFS scale (in jet)",
    'sys_1':"HFS scale (remainder)",
    'sys_5':"HFS $\phi$ angle",
    'sys_7':"Lepton energy scale",
    'sys_11':"Lepton $\phi$ angle",
    'model': 'Model',
    'QED':'QED correction',
    'closure': 'Non-closure',
    'stat':'Stat.',
}

name_translate = {
    'Herwig': "Herwig 7.2",
    'Herwig_Matchbox': "Herwig 7.2 + Matchbox",
    'Herwig_Merging': "Herwig 7.2 + Merging",
    'Pythia': 'Pythia 8.3',
    'Pythia_Vincia':'Pythia 8.3 + Vincia',
    'Pythia_Dire':'Pythia 8.3 + Dire',
    'Pythia_Dire_hadOff':'Pythia 8.3 + Dire had. off',
    'Sherpa2Cluster':'Sherpa 2',
    'Sherpa2Lund':'Sherpa 2 Lund string',
    'Sherpa3NLO': 'Sherpa 3.0 NLO',
    'Rapgap': 'RAPGAP',
    'Rapgap_unfolded': 'RAPGAP + Unfolding',
    'Rapgap_closure': 'RAPGAP Closure',
    'Djangoh': 'DJANGOH',
    'data': 'Data',

}

reco_vars = {
    'jet_ncharged':r'Charged hadron multiplicity $(\tilde{\lambda}_0^0)$', 
    'jet_charge':r'Jet Charge $(\tilde{\lambda}_0^1)$', 
    'jet_ptD':r'$p_\mathrm{T}\mathrm{D}$ $(\sqrt{\lambda_0^2})$',
    'jet_tau10':r'$\mathrm{ln}(\lambda_1^1)$', 
    'jet_tau15':r'$\mathrm{ln}(\lambda_{1.5}^1)$',
    'jet_tau20':r'$\mathrm{ln}(\lambda_2^1)$',
    'eec':r'$EEC$',
    # 'npart':'Jet particle multiplicity',
    # 'e_pt':r'electron p$_\mathrm{T}$ [GeV]',
    # 'e_theta':r'electron $\pi - \theta$',
    # 'jet_eta':r'Jet $\eta$',

}

gen_vars = {
    'gen_jet_ncharged':r'Charged hadron multiplicity N$_c$ $(\tilde{\lambda}_0^0)$', 
    'gen_jet_charge':r'Jet Charge Q$_1$ $(\tilde{\lambda}_0^1)$', 
    'gen_jet_ptD':r'$p_\mathrm{T}\mathrm{D}$ $(\sqrt{\lambda_0^2})$',
    'gen_jet_tau10':r'$\ln(\lambda_1^1)$', 
    'gen_jet_tau15':r'$\ln(\lambda_{1.5}^1)$',
    'gen_jet_tau20':r'$\ln(\lambda_2^1)$',
    'gen_eec':r'$EEC$',
    
    # 'genjet_pt':r'Jet $p_\mathrm{T}$',
    # 'gen_Q2':r'$Q^2$', 
}

