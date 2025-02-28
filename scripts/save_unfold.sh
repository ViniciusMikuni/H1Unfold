#srun python save_unfolded.py --niter 4 --load_pretrain --file Rapgap_Eplus0607_prep.h5
#srun python save_unfolded.py --niter 4 --load_pretrain --file Djangoh_Eplus0607_prep.h5
# srun python save_unfolded.py --niter 4 --load_pretrain --file Rapgap_Eplus0607_sys0_prep.h5
# srun python save_unfolded.py --data_folder /global/cfs/cdirs/m3246/H1/h5/ --weights /global/cfs/cdirs/m3246/H1/weights --nmax 100_000 --eec --niter 4 --load_pretrain
srun python save_unfolded.py --data_folder /global/cfs/cdirs/m3246/H1/h5/ --weights /global/cfs/cdirs/m3246/H1/weights --nmax 100_000 --eec --niter 4 --load_pretrain --file Rapgap_Eplus0607_sys1_prep.h5
srun python save_unfolded.py --data_folder /global/cfs/cdirs/m3246/H1/h5/ --weights /global/cfs/cdirs/m3246/H1/weights --nmax 100_000 --eec --niter 4 --load_pretrain --file Rapgap_Eplus0607_sys5_prep.h5
srun python save_unfolded.py --data_folder /global/cfs/cdirs/m3246/H1/h5/ --weights /global/cfs/cdirs/m3246/H1/weights --nmax 100_000 --eec --niter 4 --load_pretrain --file Rapgap_Eplus0607_sys7_prep.h5
srun python save_unfolded.py --data_folder /global/cfs/cdirs/m3246/H1/h5/ --weights /global/cfs/cdirs/m3246/H1/weights --nmax 100_000 --eec --niter 4 --load_pretrain --file Rapgap_Eplus0607_sys11_prep.h5