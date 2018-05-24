## Learning Tasks for Multitask Learning

The code in this repository implements the models described in the paper *Learning Tasks for Multitask Learning: Heterogenous Patient Populations in the ICU* (KDD 2018). There are two files: 

1. generate_clusters.py, which trains a sequence-to-sequence autoencoder on patient timeseries data to produce a dense representation, and then fits a Gaussian Mixture Model to the samples in this new space. 

2. run_mortality_prediction.py, which contains methods to preprocess data, as well as train and run a predictive model to predict in-hospital mortality after a certain point, given patients' physiological timeseries data. 

For more information on the arguments required to run each of these files, use the --help flag. 

### Data

Without any modification, this code assumes that you have the following files in a 'data/' folder: 
1. X.h5: an hdf file containing one row per patient per hour. Each row should include the columns {'subject_id', 'icustay_id', 'hours_in', 'hadm_id'} along with any additional features.
2. static.csv: a CSV file containing one row per patient. Should include {'subject_id', 'hadm_id', 'icustay_id', 'gender', 'age', 'ethnicity', 'first_careunit'}.
3. saps.csv: a CSV file containing one row per patient. Should include {'subject_id', 'hadm_id', 'icustay_id', 'sapsii'}. This data is found in the saps table in MIMIC III.
4. code_status.csv: a CSV file containing one row per patient. Should include {'subject_id', 'hadm_id', 'icustay_id', 'timecmo_chart', 'timecmo_nursingnote'}. This data is found in the code_status table of MIMIC III.
