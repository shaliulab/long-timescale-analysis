import utils.motionmapperpy.motionmapperpy as mmpy

parameters = mmpy.setRunParameters()

frame_rate = 100
projectPath = "mmpy_lts_3h"



# %%%%%%% PARAMETERS TO CHANGE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
parameters.projectPath = projectPath
parameters.method = "TSNE"

parameters.waveletDecomp = True  #% Whether to do wavelet decomposition. If False, PCA projections are used for

#% tSNE embedding.

parameters.minF = 1  #% Minimum frequency for Morlet Wavelet Transform

parameters.maxF = 50  #% Maximum frequency for Morlet Wavelet Transform,
#% equal to Nyquist frequency for your measurements.

parameters.perplexity = 32
parameters.training_perplexity = 32
parameters.maxNeighbors = 5

parameters.samplingFreq = frame_rate  #% Sampling frequency (or FPS) of data.

parameters.numPeriods = 25  #% No. of frequencies between minF and maxF.

parameters.numProcessors = -1 #% No. of processor to use when parallel
#% processing (for wavelets, if not using GPU). -1 to use all cores.

parameters.useGPU = 0  # GPU to use, set to -1 if GPU not present

parameters.training_numPoints=1000      #% Number of points in mini-tSNEs.

# %%%%% NO NEED TO CHANGE THESE UNLESS RAM (NOT GPU) MEMORY ERRORS RAISED%%%%%%%%%%
parameters.trainingSetSize=64000        #% Total number of representative points to find. Increase or decrease based on
                                        #% available RAM. For reference, 36k is a good number with 64GB RAM.

parameters.embedding_batchSize = 64000  #% Lower this if you get a memory error when re-embedding points on learned
                                        #% tSNE map.
