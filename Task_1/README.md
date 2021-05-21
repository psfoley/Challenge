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
The FeTS 2021 data release consists of a training set (taken from the BraTS2020 training data), along with two csv files, each providing information for how to partition the training data into non-IID institutional subsets. The release will consist of a folder ‘MICCAI_FeTS2021_TrainingData’, which contains a README.txt, subfolders with names formatted as FeTS21_Training_### each containing files for a single pateint record, and two csv files: partitioning_1.csv, partitioning_2.csv. 

Each of the partitioning csv files has two columns, ‘Partition_ID’ and ‘Subject_ID’. The Subject_ID column exhausts the patient record subfolder names contained in WOKING HEREnames of the MICCAI_FeTS2021_TrainingData  folder, each indicating a single patient record. The InstitutionName column provides a 0-indexed integer identifier indicating to which institution the record should be assigned. The paths to these csvs can be provided as the value of the parameter 'institution_split_csv_filename' to the jupyter notebook function run_challenge_experiment, to specify the institutional split used when running experimental federated training on your custom federation logic. A description of each of these split csvs is provided in Figure 1. We encourage participants to create and explore training performance for other non-IID splits of the training data to help in developing generalizable customizations to the federated logic that will perform well during the validation and testing phase. A third csv is hidden from participants and defines a test split to be used in the challenge testing phase. This hidden split (also described in Figure 1) is another refinement of the institution split, having similar difficulty level to the institution tumor size split in our own experiments using the default customization functions.

Table 1: Information for splits provided in the FeTS 2021 data release as well as the hidden split not provided in the release (to be used in the competition testing phase).

|     Split name                      |     csv filename                         |     Description                                                                                                                                                                                       |     number of institutions      |
|-------------------------------------|------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------|
|     Institution Split               |     institution_split.csv                |     Split of FeTS 2021 training data by originating   institution.                                                                                                                                    |     17                          |
|     Institution Device Split        |     institution_device_split.csv         |     Refinement of the institution split by MRI device and   configuration metadata.                                                                                                                   |     23                          |
|     Institution Tumor Size Split    |     Institution_tumorsize_split.csv      |     Refinement of the institution split by tumor size, further   splitting the larger institutions according to whether a record’s tumor size   fell above or below the mean for that institution.    |     22                          |
|     Test Split                      |          csv        not provided -       |     Undisclosed refinement of the institution split.                                                                                                                                                  |     Hidden from participants    |


Once the validation data is released, participants are free to repeatedly submit model predictions associated to these data (for n instances of training at a time) using final models resulting from federated training with the changes they implemented in their notebooks. These final models should be trained using institutional data defined by (the institution tumor size split). Mean (over the n instances) performance metrics computed using these validation data predictions will be used to rank participants. 

At the conclusion of the ranking phase, code review and replication of ranking results will be performed for top-ranking participants. Participants who pass the replication and code review phase will advance to the testing phase. Replication of top-ranking submission results will consist of copying the participant code changes into a fresh notebook template and running (n instances of) the associated federated training on the institution tumor size split. If the results differ considerably from those based on the submission of predictions, participants will be taken out of consideration. Code reviews will look for any logic that violates the spirit of generalizability by attempting to over-fit to a particular sharding. As an example, a participant may observe that a particular institution common to the shardings defined by the provided csv’s does not contribute well to global model learning. An aggregation method that identified this institution as the unique one holding a certain number of samples, and subsequently omits it from aggregation, would be an example of a technique not expected to generalize well.

Final testing of top ranking participants that pass ranking replication and code review will have their code changes applied to run (n instances of) the associated federated training on the test split. Final competition placement will be based on the testing performance metrics (DICE, precision, recall of final models, AUC of training curve, ..) associated with these final training runs.

