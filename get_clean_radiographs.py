#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import xarray as xr
import time
import matplotlib.pyplot as plt
# %matplotlib notebook


# In[2]:


hydro_path='/egr/research-slim/shared/hydro_simulations/data/'
file1 = open(hydro_path+'filenames.txt', 'r') 
Lines = file1.readlines() 
Lines = Lines[:-1]
# count=0; 
nFiles=len(Lines); # Total files to be read
# filenames=[]
rng = np.random.default_rng() # Random Number Generator
lno = rng.choice(np.arange(nFiles), size=nFiles, replace=False) # Randomly read files

print('Total number of files read:',nFiles)


# In[3]:


count = 0
sequences =[]

for n in lno:
    line=Lines[n]
    if line[48:50]!='01':
        sequences.append(line[:-1])


# In[5]:


len(sequences)


# ## Radiograph Generation
# 
# Areal Density ${{\rho_A}}$ is defined by
# 
# $${{\rho_A}}(r) = \int_{-\infty}^{\infty} \rho(r_x(t), r_y(t), r_z(t)) dt$$
# 
# Direct Radiograph is defined by:
# 
# $$D=exp(- \xi \, (C+{{\rho_A}}) \Delta l)$$
# 
# 
# where $\xi=4\times 10^{-2}\,cm^2\,g^{-1}$ is the mass attenuation coefficient of tantalum, $\Delta l = 0.025\,cm$ is the pixel spacing and $C$ is the areal mass of the collimator image. Unit of density $\rho$ is in $g\,cm^{-3}$.

# In[ ]:


frames = np.array([19,23,27,31])
Height = 650
Width = 650
num_views = 1
nFrames = len(frames)
air_threshold = 10

num_views=1
dl = 0.25
dso=1330
dsd = 5250

mac_ta = 4e-2
mac_air = 3e-2


# In[ ]:


# Reading collimator image
f = open('kernels/RMI_Collimator_ArealMass.dat', 'r')
collimator = np.genfromtxt(f)
collimator = np.reshape(collimator, (880,880))

# Cropping the collimator image
collimator=collimator[(880-Height)//2:(880+Height)//2,(880-Width)//2:(880+Width)//2]


# In[ ]:


line_number=0
get_ipython().run_line_magic('run', '../utils.ipynb')

get_ipython().run_line_magic('run', 'rotate_rho_3d.ipynb')

get_ipython().run_line_magic('run', 'generate_direct_rad.ipynb')
    
for line in sequences:

    line_number+=1
    
    sim_name=line[:-3]

    nc_loc = hydro_path+sim_name+'.nc' # location of .nc file

    print('\nFile number',line_number,'out of',len(Lines))
    print('Filename:',line)
        
#     rho_seq = get_rho(nc_loc, frames, Height = Height, Width = Width)
    
    sim = xr.open_dataarray(nc_loc)  # Reading .nc xarray file'
    
    # Using the function written by Dan (LANL) to spun the 2D density image
    rho_xyz = project_3d(sim,frames)

    direct_rad=np.zeros((nFrames,Height,num_views,Width))
    noisy_rad=np.zeros((nFrames,Height,num_views,Width))
    
    for idx in range(nFrames):

#         rho_clean = rho_seq[idx].detach().numpy() # Selecting a particular frame
#         rho_3d = rotate_rho_3d_interpolate(rho_clean) # Spinning to generate the 3D image

        rho_3d = rho_xyz[:,:,:,idx]
    
        # Cropping the density to 650 x 650
        rho_3d = rho_3d[(880-Height)//2:(880+Height)//2,(880-Width)//2:(880+Width)//2,(880-Width)//2:(880+Width)//2]
        
        # Generating Tantalum profile: Pixel values greater than or equal to 5
        rho_ta_3d = np.copy(rho_3d)
        rho_ta_3d[rho_ta_3d<air_threshold]=0
        
        # Generating Tantalum profile: Pixel values greater than or equal to 5
        rho_air_3d = np.copy(rho_3d)
        rho_air_3d[rho_air_3d>=air_threshold]=0
        
        # Generating areal density separately for air and tantalum using ASTRA toolbox
        areal_density_ta = get_areal_density_astra(rho_ta_3d, num_views=num_views, dl = dl, dso=dso, dsd = dsd)
        areal_density_air = get_areal_density_astra(rho_air_3d, num_views=num_views, dl = dl, dso=dso, dsd = dsd)

        # Generating direct radiograph
        direct_rad[idx] = generate_direct_rad(areal_density_ta, areal_density_air,collimator, mac_ta = mac_ta, mac_air = mac_air)
        
        # Adding noise to each view
        get_ipython().run_line_magic('run', 'add_noise_to_radiograph.ipynb')
        for view in range(num_views):
            noisy_rad[idx,:,view,:] = add_noise_to_radiograph(direct_rad[idx,:,view,:])

    # # Saving direct and noisy radiograph
    np.savez('/egr/research-slim/shared/hydro_simulations/radiographs-conebeam-all-files/direct/direct_'+sim_name+'.npz',\
             direct_rad=direct_rad, air_threshold=air_threshold,frames=frames, num_views=num_views, dl = dl, dso=dso, dsd = dsd,\
                mac_ta = mac_ta, mac_air = mac_air)
    np.savez('/egr/research-slim/shared/hydro_simulations/radiographs-conebeam-all-files/noisy/noisy_'+sim_name+'.npz',\
             noisy_rad=noisy_rad, air_threshold=air_threshold,frames=frames, num_views=num_views, dl = dl, dso=dso, dsd = dsd,\
             mac_ta = mac_ta, mac_air = mac_air)


# In[ ]:


stop


# ## Plotting clean and noisy radiographs for 4 frames and 8 views

# In[ ]:


for view in range(num_views):
    
    plt.figure(figsize=(10,6))

    for frame in range(4): 

        plt.subplot(2,4,frame+1)
        plt.imshow((direct_rad[frame,:,view,:]))#,vmax=1e-4)#,cmap='gray')
        plt.axis('off')
    #     plt.colorbar()#fraction=0.046, pad=0.04)
        plt.title('Direct: '+str(frames[frame]+1))

        plt.subplot(2,4,frame+5)
        plt.imshow((noisy_rad[frame,:,view,:]))#,vmax=1e-4)#,cmap='gray')
        plt.title('Noisy: '+str(frames[frame]+1))
        plt.axis('off')
        plt.tight_layout()
    #     plt.colorbar()#fraction=0.046, pad=0.04)

    #     plt.title('Masked radiograph\nFrame: '+str(frame+1))
    #     plt.savefig('masked_rad_frame_1234.png')


# In[ ]:


frame=0
view=0

direct2=-np.log(direct_rad[idx,:,view,:])
noisy2 = add_noise_to_radiograph(direct2)
    
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
# plt.imshow(direct2,vmax=None)
plt.imshow(direct_rad[frame,:,view,:],vmax=None)#1e-3)
# plt.axis('off')
plt.colorbar(fraction=0.05, pad=0.04)
plt.title('Direct: '+str(frames[frame]+1))

plt.subplot(1,2,2)
# plt.imshow(noisy2,vmax=None)
plt.imshow((noisy_rad[frame,:,view,:]),vmax=1e-3)
plt.title('Noisy: '+str(frames[frame]+1))
# plt.axis('off')
plt.colorbar(fraction=0.05, pad=0.04)
plt.tight_layout()

# plt.savefig('direct_and_noisy_rad.png')


# In[ ]:


direct_rad.max()


# In[ ]:


plt.figure()
plt.semilogy(direct_rad[frame,650//2,view,:]); plt.show()
plt.semilogy(noisy_rad[frame,650//2,view,:])
plt.grid('on')
plt.xlabel('x')
plt.legend(['Direct','Noisy'])
plt.ylabel('Radiograph Magnitude')
plt.title('Frame: '+str(frame+1)+' View: '+str(view+1))
plt.savefig('direct_noisy_rad_line_out.png')


# In[ ]:




