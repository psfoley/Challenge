# FeTS 2021 Challenge Part 1
Task 1 ("Federated Training") aims at effective weight aggregation methods for the creation of a consensus model given a pre-defined segmentation algorithm for training, while also (optionally) accounting for network outages.

## Getting started
1. Register for the FeTS 2021 Challenge [here](https://www.med.upenn.edu/cbica/fets/miccai2021/) and submit a data request
2. ```git clone https://github.com/FETS-AI/Challenge.git```
3. ```cd Challenge/part_1```
4. create virtual environment (python 3.6-3.8)
5. ```pip install --upgrade pip```
6. ```pip install .```
7. ```jupyter notebook```

## Data Partitioning and Sharding
The FeTS 2021 data release consists of a training set, along with two csv files each providing information for how to partition the training data into non-IID institutional subsets. The release will contain a README.txt, subfolders with names formatted as FeTS21_Training_### - each containing files for a single pateint record, and two csv files: partitioning_1.csv, partitioning_2.csv. 

Each of the partitioning csv files has two columns, ‘Partition_ID’ and ‘Subject_ID’. The Subject_ID column exhausts the patient record subfolder names contained in the data release. The InstitutionName column provides an string integer identifier indicating to which institution the record should be assigned. The paths to these csvs can be provided as the value of the parameter 'institution_split_csv_filename' to the jupyter notebook function run_challenge_experiment, to specify the institutional split that should be used when running experimental federated training on your custom federation logic. A description of each of these split csvs is provided in Figure 1. We encourage participants to create and explore training performance for other non-IID splits of the training data to help in developing generalizable customizations to the federated logic that will perform well during the validation and testing phase. A third csv is hidden from participants and defines a test split to be used in the challenge testing phase. This hidden split (also described in Figure 1) is another refinement of the institution split, having similar difficulty level to the institution tumor size split in our own experiments using the default customization functions.

Table 1: Information for splits provided in the FeTS 2021 data release, as well as the hidden split not provided in the release (to be used in the competition testing phase).

|     Split name                      |     csv filename                         |     Description                                                                                                                                                                                       |     number of institutions      |
|-------------------------------------|------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------|
|     Institution Split               |     partitioning_1.csv                |     Split of FeTS 2021 training data by originating   institution.                                                                                                                                    |     17                          |
|     Institution Tumor Size Split    |     partitioning_2.csv      |     Refinement of the institution split by tumor size, further   splitting the larger institutions according to whether a record’s tumor size   fell above or below the mean for that institution.    |     22                          |
|     Test Split                      |          csv not provided        |     Undisclosed refinement of the institution split.                                                                                                                                                  |     Hidden from participants    |




