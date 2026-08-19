import h5py
from wholistic_registration import utils as wbi
from wholistic_registration.utils import preprocess, calFlow3d_Wei_v1, visualization, mask, option
import numpy as np
import nd2


#set the option
option['r']=5



#set basic parameters
frameJump=1
refLength=5
refJump =40/frameJump
initialLength=5
thresFactor=5
smFactor=50
maskRange=[5,500]
smoothPenalty_raw=0.01
save_ite = 100

#frame
T=100


filePath='/home/cyf/wbi/Virginia/221124_f338_ubi_gCaMP7f_bactin_mCherry_CAAX_7dpf002.nd2'


with nd2.ND2File(filePath) as f:
    metadata=f.metadata
    channels=metadata.channels[0]
    print(f.sizes)
    #get Zratio
    axesCalibration=channels.volume.axesCalibration
    zRatio=axesCalibration[2]/axesCalibration[0]
    print("Z ratio is", zRatio)

    #get size
    [X,Y,Z]=channels.volume.voxelCount
    print("Data size is",[X,Y,Z])

    #get total frames
    frames=metadata.contents.frameCount
    print("Total frames is",frames)

    tRange=range(0,T,frameJump)

    #initial the var
    dat_channel2=np.zeros([X,Y,Z,len(tRange)],dtype=np.int16)
    motion_history=np.zeros([X,Y,Z,3,initialLength],dtype=np.float32)
    option['motion']=np.zeros([X,Y,Z,len(tRange)])

    #load all the data(virtual)
    dask_data=f.to_dask()[:,:,1,:,:]
    Ca_data=f.to_dask()[:,:,0,:,:]

    dat_ref=dask_data[0].compute().transpose(2,1,0)
    print(dat_ref.shape)
    option['mask_ref']=mask.getMask(dat_ref,thresFactor)
    option['mask_ref']=mask.bwareafilt3_wei(option['mask_ref'],maskRange)
    Pnltfactor=preprocess.getSmPnltNormFctr(dat_ref,wbi.option)
    smoothPenalty=Pnltfactor*smoothPenalty_raw


    #start to registration
    for tCnt in range(len(tRange)):
        t=tRange[tCnt]
        print("read data (1)...")
        dat_mov=dask_data[t].compute().transpose(2,1,0)
        option['mask_mov']=mask.getMask(dat_ref,thresFactor)
        option['mask_mov']=mask.bwareafilt3_wei(option['mask_mov'],maskRange)
        print("generate reference...")
        if (tCnt - 1) % refJump == 0:
            if tCnt > refLength * refJump:
                ref_range = np.arange(tCnt - refLength * refJump, tCnt, refJump)
                # Compute median along time axis (axis=3 for 4D array)
                dat_ref = np.median(dat_channel2[:, :, :, ref_range], axis=3).astype(np.float32)
            
            if tCnt > refLength: # ginny's version
                current_jump = np.min((tCnt - refLength) // refJump, 1)
                print(f"current_jump: {current_jump}")

                ref_range = np.arange(tCnt - refLength * current_jump, tCnt, refJump)
                # Compute median along time axis (axis=3 for 4D array)
                dat_ref = np.median(dat_channel2[:, :, :, ref_range], axis=3).astype(np.float32)
            # Generate and filter mask
            # option['mask_ref'] = mask.getMask(dat_ref, thresFactor)
            # option['mask_ref'] = mask.bwareafilt3_wei(option['mask_ref'], maskRange)
            option['mask_ref']=np.full(dat_ref.shape,False,dtype=bool)
            option['mask_mov']=np.full(dat_ref.shape,False,dtype=bool)
            
            # Update penalty factor
            pnlt_factor = preprocess.getSmPnltNormFctr(dat_ref, option)
            smoothPenalty=Pnltfactor*smoothPenalty_raw
    
        print("calculating the motion")
        motion_current,_,_=calFlow3d_Wei_v1.getMotion(dat_mov,dat_ref,smoothPenalty,option)
        dat_channel1=calFlow3d_Wei_v1.correctMotion(Ca_data[0].compute().transpose(2,1,0),motion_current)
        dat_channel2[:,:,:,tCnt]=calFlow3d_Wei_v1.correctMotion(dat_mov,motion_current)
