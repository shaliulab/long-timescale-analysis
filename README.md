# A database of continuous, long timescale recordings of *Drosophila melanogaster* postural data

![Shaevitz Lab logo](documents/images/shaevitz_logo.png)


This reposity cotains the code corresponding to
```
@article{lts-behavior,
  author = {Grace C. McKenzie-Smith and Scott Wolf and Joshua Shaevitz},
  title = {A database of continuous, long timescale recordings1
of \textit{Drosophila melanogaster} postural data}
```
## Data

The data associated with this study can be found at [URL] and is described in detail within the paper. The data set encompasses over 2 billion pose instances, each paired with edge calls and metadata.

## Analysis

Brief examples of anlysis are available here illustrating how to load and analyze tracks.

The sequential run of the analysis pipeline is:

1. `sbatch input-processing.slurm` 
2. `sbatch wavelets.slurm`
3. `sbatch subsample_for_training.slurm`
4. `python lts-tsne-subsample-3.py`
4. `embed.slurm`
5. `lts-tsne-watershed-5.py`

### input-processing.slurm

This script takes the raw data, interpolates missing data, and smooths datas needed. It also splits the data to the individual flies and saves the data in a format that is easy to load.

### wavelets.slurm

This script computes the wavelet transform of the data. It is a computationally expensive step and is parallelized over the flies as much as possible. 

### subsample_for_training.slurm

Because subsampling directly requires a lot of memory, this script subsamples the data for training the embedding for each file. It is parallelized over the flies. This also calculates

### lts-tsne-subsample-3.py

Generates training data and embeds the data using t-SNE. This training provides the model for the embedding of the full data set.

### embed.slurm

This script embeds the full data set using the model generated in the previous step. It's **extremely** slow and is parallelized over the flies but still takes multiple days to run on a cluster.


## Tracking

We used SLEAP [sleap.ai] for tracking the flies. The tracking code can be found in the tracking directory. The code is a slightly modified version that enables the input of MKV files.

The tracking process is parallelized over fly-hours to efficiently utilize GPU resources for variable-length data.


## Documentation

We also included documentation on experimental methods, including the media recipe, in the documents directory.
