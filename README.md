# master-thesis-code

Code for preprocessing and performing analysis of spatiotemporal dynamics in individual radial astrocytes during stimulus responses.

## Preprocessing steps:
  0. Imaging data is assumed to be in a .raw file and have a corresponding .ini configuration file. Assumed key names is documented in cropbin/cropbin_util.py/CFG_FORMAT. Imaging data is assumed recording from an experiment with recurring visual stimulus.
  1. Crop and downsample
    a. Create configuration file in the same format as GFAP_GCamp6s.ini, including all the recordings that should be processed and their respective FOVs.
    b. Set appropriate constants in cropbin.py (Directories, cfg_file, exp_names, start/end frame, time_bin_size)
    c. Run cropbin.py
  2. Convert .raw to .tiff
    a. Set appropriate constants in raw2tif.py
    b. run raw2tif.py
  3. Align imaging data
    a. Set appropriate constants in s2p_registration.py
    b. Create ops-file from suite2p
    c. run s2p_registration.py
  4. (optional, assumes necessary installs have been made for DeepInterpolation) Noise removal with DeepInterpolation
    a. (training) Select traning and validation dataset in deepinterpolation_scripts/training.py
    b. (training) run training.py
    c. Set appropriate constants in inference.py
    d. run inference.py
  5. (optional) Create tiff from aligned and denoised data
    a. run bin2tiff.py
  6. Cell detection
    a. set exp_names and directories in s2p_celldetect.py
    b. run s2p_celldetect.py
    c. filter ROIs with suite2p GUI
  7. Merge partial ROIs
    a. Set appropratie constants in roimerge.py and run

## Data analysis
  1. Extract dF/F
    a. Set appropratie constants in dff.py and run
  2. Navigate to light_response_regions
  3. Feature extract
    a. Configure constants in feature_extract.py and run
  4. Plot
    a. Set appropriate constants in plot.py
    b. Select what plots to create by uncommenting in main()
    c. Run plot.py
