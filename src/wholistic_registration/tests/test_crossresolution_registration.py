# %% [markdown]
# get the data
# 

# %%
import os
os.chdir('/home/cyf/wbi/Virginia/code/wbi_0123/wholistic_registration/src/wholistic_registration')
import cupy as cp
cp.cuda.Device(1).use()
from wholistic_registration.utils import IO
from wholistic_registration.utils import calFlowCrossResolution, calFlow3d_Wei_v1
from wholistic_registration.utils import registration
from wholistic_registration.utils import option
from wholistic_registration.utils import preprocess as prep
from wholistic_registration.utils import mask
from wholistic_registration.utils import visualization
import numpy as np
gut_mov_Path     = "/home/cyf/wbi/Virginia/registrated_data/f260201/gut/raw/260201_test1_0_00002_TZCXY.ome.tif"
ventral_mov_Path = "/home/cyf/wbi/Virginia/registrated_data/f260201/ventral/raw/260201_test1_0_00003_TZCXY.ome.tif"
dorsal_mov_Path  = "/home/cyf/wbi/Virginia/registrated_data/f260201/dorsal/raw/260201_test1_0_00005_TZCXY.ome.tif"
gut_ref_Path     = "/home/cyf/wbi/Virginia/registrated_data/f260201/gut/anat/260201_test1_0_00002_TZCXY.ome.tif"
ventral_ref_Path = "/home/cyf/wbi/Virginia/registrated_data/f260201/ventral/anat/260201_test1_0_00002_TZCXY.ome.tif"
dorsal_ref_Path  = "/home/cyf/wbi/Virginia/registrated_data/f260201/dorsal/anat/260201_test1_0_00004_TZCXY.ome.tif"

# %% [markdown]
# gut exp
# ---

# %%
gut_ref,gut_ref_desc = IO.readTifff(gut_ref_Path)
gut_mov,gut_mov_desc = IO.readTifff(gut_mov_Path)

gut_ref = gut_ref.transpose(1,2,0)
gut_mov = gut_mov.transpose(0,2,3,1)
### see the slice 86, 87,88
### use the whole image as the groundtruth
#initial the pyramid parameters
for frame in range(0,5):
    gut_ref_sample = gut_ref[...,40:120]
    gut_mov_sample = gut_mov[frame,:,:,40:120]
    print(gut_ref_sample.shape)
    print(gut_mov_sample.shape)
    from skimage.exposure import match_histograms
    gut_mov_sample = match_histograms(gut_mov_sample, gut_ref_sample)
    # visualization.visualize_2d_image(gut_mov_sample[...,slice],title = "moving image")
    # visualization.visualize_2d_image(gut_ref_sample[...,slice],title = "reference image")
    slice = 39
    option['motion']=np.zeros([gut_ref_sample.shape[0],gut_ref_sample.shape[1],gut_ref_sample.shape[2],3])
    option['r']=5
    option['layer']=2
    option['iter']=10
    option['movRange']=5.
    option['tol']=1e-6
    thresFactor= 5.
    maskRange  = [5.,4000.]
    smoothPenalty_raw=0.03

    option['mask_ref']=mask.getMask(gut_ref_sample,thresFactor)
    option['mask_ref']=mask.bwareafilt3_wei(option['mask_ref'],maskRange)
    option['mask_mov']=mask.getMask(gut_mov_sample,thresFactor)
    option['mask_mov']=mask.bwareafilt3_wei(option['mask_mov'],maskRange)

    print(option['mask_ref'].shape)
    Pnltfactor = prep.getSmPnltNormFctr(gut_ref_sample, option)
    smoothPenalty=Pnltfactor*smoothPenalty_raw
    option['smoothPenalty']=smoothPenalty
    option['zRatio'] = 20

    # motion_current, _ , new_coords,error_logs = calFlow3d_Wei_v1.getMotion(gut_mov_sample, gut_ref_sample,option,verbose = True)
    # corrected_mem = calFlow3d_Wei_v1.correctMotion(gut_mov_sample, motion_current)
    # visualization.visualize_2d_image(corrected_mem[...,47],title = "corrected image")

    gut_single_plane = gut_mov_sample[:,:,slice:slice+1]
    option['zRatio_HR']= 20

    H, W, D = gut_single_plane.shape
    X, Y, Z = np.indices((H, W, D))

    # phase_init = np.stack([
    #     X.astype(np.float32),
    #     Y.astype(np.float32),
    #     (slice + Z * option["zRatio"] / option["zRatio_HR"]).astype(np.float32)
    # ], axis=-1)
    phase_init, z_init = calFlowCrossResolution.FindInitPhase(gut_single_plane,gut_ref_sample,64,return_debug = False)

    print(phase_init.shape)
    option["phase"] = np.array(phase_init)

#     phase_new,motion_current,data_mov_mapped= calFlowCrossResolution.getMotion(gut_single_plane, gut_ref_sample,option,verbose = True)
#     # visualization.visualize_2d_image(data_mov_mapped.get(),title = "mapped image")
#     IO.write_multichannel_volume_as_ome_tiff(
#             volume=[data_mov_mapped.get().squeeze(),gut_mov_sample[...,slice],gut_ref_sample[...,slice],corrected_mem[...,slice]],      # single channel
#             out_dir="/home/cyf/wbi/Virginia/HR_exp/",
#             frame_idx= frame,
#             label=f'Gut_HR_S{slice+40}_optiInit'
#     )

# %%
gut_ref,gut_ref_desc = IO.readTifff(gut_ref_Path)
gut_mov,gut_mov_desc = IO.readTifff(gut_mov_Path)
# print(gut_ref_desc)
gut_ref = gut_ref.transpose(1,2,0)
gut_mov = gut_mov.transpose(0,2,3,1)
### see the slice 86, 87,88
### use the whole image as the groundtruth
#initial the pyramid parameters
for frame in range(0,5):
    gut_ref_sample = gut_mov[2,:,:,40:120]
    gut_mov_sample = gut_mov[frame,:,:,40:120]
    print(gut_ref_sample.shape)
    print(gut_mov_sample.shape)
    from skimage.exposure import match_histograms
    gut_mov_sample = match_histograms(gut_mov_sample, gut_ref_sample)
    # visualization.visualize_2d_image(gut_mov_sample[...,slice],title = "moving image")
    # visualization.visualize_2d_image(gut_ref_sample[...,slice],title = "reference image")
    slice = 39
    option['motion']=np.zeros([gut_ref_sample.shape[0],gut_ref_sample.shape[1],gut_ref_sample.shape[2],3])
    option['r']=5
    option['layer']=3
    option['iter']=10
    option['movRange']=15.
    option['tol']=1e-6
    thresFactor= 5.
    maskRange  = [5.,4000.]
    smoothPenalty_raw=0.03

    option['mask_ref']=mask.getMask(gut_ref_sample,thresFactor)
    option['mask_ref']=mask.bwareafilt3_wei(option['mask_ref'],maskRange)
    option['mask_mov']=mask.getMask(gut_mov_sample,thresFactor)
    option['mask_mov']=mask.bwareafilt3_wei(option['mask_mov'],maskRange)

    print(option['mask_ref'].shape)
    Pnltfactor = prep.getSmPnltNormFctr(gut_ref_sample, option)
    smoothPenalty=Pnltfactor*smoothPenalty_raw
    option['smoothPenalty']=smoothPenalty
    option['zRatio'] =  1

    # motion_current, _ , new_coords,error_logs = calFlow3d_Wei_v1.getMotion(gut_ref_sample, gut_mov_sample,option,verbose = False)
    # corrected_mem = calFlow3d_Wei_v1.correctMotion(gut_ref_sample, motion_current)
    # visualization.visualize_2d_image(corrected_mem[...,47],title = "corrected image")
    # visualization.visualize_2d_image(gut_mov_sample[...,47],title = "raw image")
    gut_single_plane = gut_mov_sample[:,:,slice:slice+1]
    option['zRatio_HR']= 1

    option["wrong_region_enable"] = True
    option["wrong_region_metric"] = "mae"
    option["wrong_region_mad_threshold"] = 3
    option["wrong_region_min_component_size"] = 2
    option["wrong_region_exclude_mode"] = "highresidual"

    H, W, D = gut_single_plane.shape
    X, Y, Z = np.indices((H, W, D))
    # phase_init = np.stack([
    #     X.astype(np.float32),
    #     Y.astype(np.float32),
    #     (slice + Z * option["zRatio"] / option["zRatio_HR"]).astype(np.float32)
    # ], axis=-1)
    phase_init, z_init,debug = calFlowCrossResolution.FindInitPhase_robust(gut_single_plane,
                                                          gut_ref_sample,
                                                          100,
                                                          return_debug = True,
                                                          use_gradient=False,
                                                          smooth_sigma = 30,
                                                          overlap = 0.6,
                                                          )
    option["phase"] = np.array(phase_init)

    phase_new,motion_current,data_mov_mapped= calFlowCrossResolution.getMotion_v2(gut_single_plane, gut_ref_sample,option,verbose =True)
#     phase_new,motion_current,data_mov_mapped= calFlowCrossResolution.getMotion(gut_single_plane, gut_ref_sample,option,verbose =True)
    H_layer = calFlowCrossResolution.generate_continuous_H_gpu(
            gut_ref_sample,
            zRatio=1
    )
    data_mov_init= calFlowCrossResolution.apply_H_to_matrix_gpu(phase_init,H_layer)
    # visualization.visualize_2d_image(gut_single_plane)
    visualization.visualize_2d_image(data_mov_init.get(),title="init map")
    visualization.visualize_2d_image(data_mov_mapped,title = "eventual image")
    visualization.visualize_2d_image(gut_single_plane,title = "origin image")
    # visualization.plot_deformed_grid_plotly(phase_new[:,:,0,:],step = 5)
    # visualization.visualize_2d_image(data_mov_mapped.get(),title = "mapped image")
    IO.write_multichannel_volume_as_ome_tiff(
            volume=[data_mov_mapped.squeeze(),gut_mov_sample[...,slice]],      # single channel
            out_dir="/home/cyf/wbi/Virginia/HR_exp/gut_exp",
            frame_idx= frame,
            label=f'Gut_HR_S{slice+40}_align2Third'
    )

# %%
delta = phase_new - phase_init
print(delta.shape)
visualization.visualize_2d_image(delta[:,:,0,2],autocontrast=False)

# %%

visualization.plot_deformed_grid_plotly(phase_new[:,:,0,:],spacing = (1,1,10) ,step = 3)

# %%
i,j = 6,14
x_start = int(debug['xs'][i])-40
x_end = x_start + 140
y_start = int(debug['ys'][i])-40
y_end = y_start +140
visualization.visualize_2d_image(gut_single_plane[y_start:y_end, x_start:x_end,0])
visualization.visualize_3d_image(gut_ref_sample[y_start:y_end, x_start:x_end])
print(debug['scores'].shape)
print(f"x: [{x_start}, {x_end}]")
print(f"y: [{y_start}, {y_end}]")
print(np.argmax(debug['scores'][i][j]))

# %%
print(len(debug['ys']))
print(len(debug['xs']))
print(gut_single_plane.shape)

# %%
#mask掉亮的点
#考虑不用corr
#根据统计需要拒绝一些选出来的most correlated slice
#详细去看看褶为啥不一样

i,j = 6,8
visualization.plot_sequence(debug['scores'][i][j].get())
print(np.argmax(debug['scores'][i][j].get()))
y_start = int(debug['ys'][i])
y_end = y_start + 150
x_start = int(debug['xs'][j])
x_end = x_start +150
visualization.visualize_2d_image(gut_single_plane[y_start:y_end, x_start:x_end,0],autocontrast=False)
# visualization.visualize_2d_image(data_mov_mapped.get()[y_start:y_end, x_start:x_end,0],autocontrast=False)
visualization.visualize_3d_image(gut_ref_sample[y_start:y_end, x_start:x_end])

# %% [markdown]
# Dorsal
# ---

# %%
dorsal_ref,dorsal_ref_desc = IO.readTifff(dorsal_ref_Path)
dorsal_mov,dorsal_mov_desc = IO.readTifff(dorsal_mov_Path)

dorsal_ref = dorsal_ref.transpose(1,2,0)
dorsal_mov = dorsal_mov.transpose(0,2,3,1)
### see the slice 86, 87,88
### use the whole image as the groundtruth
#initial the pyramid parameters
for frame in range(1,3):
    dorsal_ref_sample = dorsal_mov[0,:,:,16:]
    dorsal_mov_sample = dorsal_mov[frame,:,:,16:]
    print(dorsal_ref_sample.shape)
    print(dorsal_mov_sample.shape)
    from skimage.exposure import match_histograms
    dorsal_mov_sample = match_histograms(dorsal_mov_sample, dorsal_ref_sample)
    # visualization.visualize_2d_image(gut_mov_sample[...,slice],title = "moving image")
    # visualization.visualize_2d_image(gut_ref_sample[...,slice],title = "reference image")
    slice = 53
    option['motion']=np.zeros([dorsal_ref_sample.shape[0],dorsal_ref_sample.shape[1],dorsal_ref_sample.shape[2],3])
    option['r']=5
    option['layer']=2
    option['iter']=15
    option['movRange']=10.
    option['tol']=1e-6
    thresFactor= 5.
    maskRange  = [5.,4000.]
    smoothPenalty_raw=0.1

    option['mask_ref']=mask.getMask(dorsal_ref_sample,thresFactor)
    option['mask_ref']=mask.bwareafilt3_wei(option['mask_ref'],maskRange)
    option['mask_mov']=mask.getMask(dorsal_mov_sample,thresFactor)
    option['mask_mov']=mask.bwareafilt3_wei(option['mask_mov'],maskRange)

    print(option['mask_ref'].shape)
    Pnltfactor = prep.getSmPnltNormFctr(dorsal_ref_sample, option)
    smoothPenalty=Pnltfactor*smoothPenalty_raw
    option['smoothPenalty']=smoothPenalty
    option['zRatio'] = 1

    motion_current, _ , new_coords,error_logs = calFlow3d_Wei_v1.getMotion(dorsal_mov_sample, dorsal_ref_sample,option,verbose = True)
    corrected_mem = calFlow3d_Wei_v1.correctMotion(dorsal_mov_sample, motion_current)
    # visualization.visualize_2d_image(corrected_mem[...,47],title = "corrected image")

    dorsal_single_plane = dorsal_mov_sample[:,:,slice:slice+1]
    option['zRatio_HR']=1

    H, W, D = dorsal_single_plane.shape
    X, Y, Z = np.indices((H, W, D))
    phase_init = np.stack([
        X.astype(np.float32),
        Y.astype(np.float32),
        (slice + Z * option["zRatio"] / option["zRatio_HR"]).astype(np.float32)
    ], axis=-1)
    print(phase_init.shape)
    option["phase"] = np.array(phase_init)

    phase_new,motion_current,data_mov_mapped= calFlowCrossResolution.getMotion(dorsal_single_plane, dorsal_ref_sample,option,verbose = True)
    # visualization.visualize_2d_image(data_mov_mapped.get(),title = "mapped image")
    IO.write_multichannel_volume_as_ome_tiff(
            volume=[data_mov_mapped.get().squeeze(),dorsal_mov_sample[...,slice],dorsal_ref_sample[...,slice],corrected_mem[...,slice]],      # single channel
            out_dir="/home/cyf/wbi/Virginia/HR_exp/",
            frame_idx= frame,
            label=f'Dorsal_HR_S{slice+16}_align2First'
    )

# %% [markdown]
# Ventral
# ---

# %%
ventral_ref,ventral_ref_desc = IO.readTifff(ventral_ref_Path)
ventral_mov,ventral_mov_desc = IO.readTifff(ventral_mov_Path)

ventral_ref = ventral_ref.transpose(1,2,0)
ventral_mov = ventral_mov.transpose(0,2,3,1)
### see the slice 86, 87,88
### use the whole image as the groundtruth
#initial the pyramid parameters
for frame in range(1,3):
    ventral_ref_sample = ventral_mov[0,:,:,38:]
    ventral_mov_sample = ventral_mov[frame,:,:,38:]
    print(ventral_ref_sample.shape)
    print(ventral_mov_sample.shape)
    from skimage.exposure import match_histograms
    ventral_mov_sample = match_histograms(ventral_mov_sample, ventral_ref_sample)
    # visualization.visualize_2d_image(gut_mov_sample[...,slice],title = "moving image")
    # visualization.visualize_2d_image(gut_ref_sample[...,slice],title = "reference image")
    slice = 16
    option['motion']=np.zeros([ventral_ref_sample.shape[0],ventral_ref_sample.shape[1],ventral_ref_sample.shape[2],3])
    option['r']=5
    option['layer']=2
    option['iter']=15
    option['movRange']=10.
    option['tol']=1e-6
    thresFactor= 5.
    maskRange  = [5.,4000.]
    smoothPenalty_raw=0.1

    option['mask_ref']=mask.getMask(ventral_ref_sample,thresFactor)
    option['mask_ref']=mask.bwareafilt3_wei(option['mask_ref'],maskRange)
    option['mask_mov']=mask.getMask(ventral_mov_sample,thresFactor)
    option['mask_mov']=mask.bwareafilt3_wei(option['mask_mov'],maskRange)

    print(option['mask_ref'].shape)
    Pnltfactor = prep.getSmPnltNormFctr(ventral_ref_sample, option)
    smoothPenalty=Pnltfactor*smoothPenalty_raw
    option['smoothPenalty']=smoothPenalty
    option['zRatio'] = 1

    motion_current, _ , new_coords,error_logs = calFlow3d_Wei_v1.getMotion(ventral_mov_sample, ventral_ref_sample,option,verbose = True)
    corrected_mem = calFlow3d_Wei_v1.correctMotion(ventral_mov_sample, motion_current)
    # visualization.visualize_2d_image(corrected_mem[...,47],title = "corrected image")

    ventral_single_plane = ventral_mov_sample[:,:,slice:slice+1]
    option['zRatio_HR']=1

    H, W, D = ventral_single_plane.shape
    X, Y, Z = np.indices((H, W, D))
    phase_init = np.stack([
        X.astype(np.float32),
        Y.astype(np.float32),
        (slice + Z * option["zRatio"] / option["zRatio_HR"]).astype(np.float32)
    ], axis=-1)
    print(phase_init.shape)
    option["phase"] = np.array(phase_init)

    phase_new,motion_current,data_mov_mapped= calFlowCrossResolution.getMotion(ventral_single_plane, ventral_ref_sample,option,verbose = True)
    # visualization.visualize_2d_image(data_mov_mapped.get(),title = "mapped image")
    IO.write_multichannel_volume_as_ome_tiff(
            volume=[data_mov_mapped.get().squeeze(),ventral_mov_sample[...,slice],ventral_ref_sample[...,slice],corrected_mem[...,slice]],      # single channel
            out_dir="/home/cyf/wbi/Virginia/HR_exp/",
            frame_idx= frame,
            label=f'ventral_HR_S{slice+38}_align2First'
    )


