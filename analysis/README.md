# A database of continuous, long timescale recordings of *Drosophila melanogaster* postural data

## Rerunning the analysis


### Converting .mat files to .h5

We use `mat_to_h5.py` to convert the .mat files to .h5 files. The .mat files are in the base_path directory, and the .h5 files will be saved in the base directory.

```bash
python mat_to_h5.py
```

We calculate node velocities here. We write the tracks and vels to a matching h5 with datasets 'tracks' containing the data and 'vels' containing the velocities.


### Converting to input type compatible with motionmapperpy

```bash
sbatch input-processing.slurm
```

`lts-tsne-input-1.py` which is submitted as a batch job through `input-processing.slurm`. Thjis script takes the .h5 files and converts them to the input type compatible with motionmapperpy. The output is saved in the project directory specificed 

### Calculating wavelets


