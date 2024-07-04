# PURL: Safe and Effective Sanitization of Link Decoration

This repository provides the necessary steps to run PURL using a customized version of OpenWPM. Follow the detailed instructions below to install dependencies, crawl websites, preprocess data, and run the classification pipeline.

## System Requirements
Following are the system requirements to run PURL:

- **Operating System**: Ubuntu 18.04 or later
- **Memory**: 64GB RAM or higher
- **Python**: 3.9+
- **Network connectivity**: Required to crawl websites and download filter lists



## Step 1: Install OpenWPM Dependencies

First, navigate to the `OpenWPM` folder in the repository:
```bash
cd OpenWPM
```

Next, install the required dependencies by running the provided installation script:
```bash
./install.sh
```

(In case there's an error which says that ```gcc``` is missing from your system, run ```sudo apt install build-essential```)

This script will create a conda environment named `openwpm` and install all necessary dependencies. Once the installation is complete, activate the conda environment:
```bash
conda activate openwpm
```

We will use this environment for the rest of the pipeline.

## Step 2: Crawl Websites Using OpenWPM

To begin crawling websites, create a `.csv` file containing the list of websites you want to crawl. Ensure the URLs are listed under the `url` column and include the protocol (http/https). Here is an example of the CSV format:
```
url
https://www.example.com
https://www.example2.com
```

With your `.csv` file ready, run the crawl using the following command from within the `OpenWPM` folder:
```bash
python3 crawl_sites.py --websites <path_to_csv_file> --output_dir <output_directory> --num_browsers <number_of_browsers> --starting_index <starting_index> --log_file <log_file> --third_party_cookies <always|never>
```

### Explanation of Arguments:
- `--websites`: Path to the `.csv` file with the list of websites to crawl.
- `--output_dir`: Path to the directory where the crawl data will be stored.
- `--num_browsers`: Number of browsers to use for the crawl (recommended: 5-10).
- `--starting_index`: Index of the first website in the `.csv` file to crawl.
- `--log_file`: Path to the log file where crawl logs will be stored.
- `--third_party_cookies`: Whether to allow third-party cookies (`always` or `never`).

**Note:** OpenWPM processes websites in batches of 1,000. If your CSV file contains fewer than 1,000 websites, a single output file will be generated. For more than 1,000 websites, multiple output folders will be created, each containing up to 1,000 websites' data. The output files will be stored in the specified `output_directory`. Each output folder will contain the following two important files:

1. ```crawl-data.sqlite``` (SQLite database containing the crawl data).
2. ```content.ldb``` (LevelDB database containing the content data such as JavaScript loaded on the website).

## Step 3: Run Preprocessing for PURL

The next step involves generating a graph and its corresponding features for each website. To do this, navigate to the `Pipeline` folder and run the following command:
```bash
cd ../Pipeline
python3 run.py --features <path_to_feature_file> --folder <crawl_data_folder> --tag <file_tag>
```
If running the above command results in following error: ```OSError: "enchant-2: cannot read file data: Is a directory```, then run this command first: ```sudo apt install libenchant-2-dev && export PYENCHANT_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/libenchant-2.so```.

### Explanation of Arguments:
- `--features`: Path to the file containing the features to extract (default: `features_new.yaml`).
- `--input_data`: Path to the folder where the crawl data was stored. For example, if the data was stored in a `datadir` directory within the `OpenWPM` folder, the path should be `../OpenWPM/datadir`.
- `--tag`: Tag to add to the output files. This helps differentiate between multiple experiments.

Running this command will generate four files for each run of crawled data:
- `features_n.csv` (features to be used by classifier to predict tracking/non-tracking use of link decoration)
- `graph_n.csv` (graph representation websites)
- `exfils_n.csv` (storage value exfiltration observed through link decorations and payloads)
- `labels_n.csv` (labels of link decorations observed during the crawl)

Where ```n``` is the index of the run.

These files will be stored in the same directory as the `run.py` script.

## Step 4: Run the Classification Pipeline for PURL

Navigate to the `classification` folder within the `Pipeline` directory:
```bash
cd classification
```

Run the classification pipeline with the following command:
```bash
python3 classify.py --result_dir <result_directory> --iterations <number_of_iterations>
```

### Explanation of Arguments:
- `--result_dir`: Path to the directory where the classification results will be stored.
- `--iterations`: Number of iterations needed to cover the complete dataset. For instance, if 9 feature files were generated in the previous step (from `features_0.csv` to `features_8.csv`), the number of iterations should be 9.

This command will execute the classification pipeline and store the results, feature importances, and other metrics, along with the model save files, in the specified `result_directory`.

A few important files that are produced as a result of this step are:

1. ```accuracy``` this file contains the accuracy metrics of each fold of training step. This can be used to determine the best model to pick for the next step (another file ```scores``` contains the confusion matrices for each fold for more information). The accuracy number here is calculated against the ground truth which is described in detail in Section 4.1 of the accepted paper.

2. ```tp_n```, where n ranges from 0 to 9 shows the predictions of the each trained model in a fold against the selected test data. The resulting file has the following structure:
   
```Ground Truth Label``` |\$| ```Model Prediction``` |\$| ```Link Decoration Name``` |\$| ```Visit ID```

4. ```model_n.sav```, where n ranges from 0 to 9, are the checkpoints of the trained models. These checkpoints can be used to run the best performing model in the next step.

## Step 5: Run the Best Model on the Complete Dataset

Finally, run the best model on the complete dataset. Navigate to the `classification` folder within the `Pipeline` directory:
```bash
cd classification
```

Execute the following command:
```bash
python3 classify_with_model.py --result_dir <result_directory> --model_path <model_save_file_path> --iterations <number_of_iterations> --generate-filterlist/--no-generate-filterlist
```

### Explanation of Arguments:
- `--result_dir`: Path to the directory where the results will be stored.
- `--model_path`: Path to the saved model file, this model is the best performing model from the previous step (based on accuracy score or any other chosen metric).
- `--iterations`: Number of iterations needed to cover the complete dataset.
- `--generate-filterlist/--no-generate-filterlist`: Option to generate a filterlist based on the model results (`--generate-filterlist` or `--no-generate-filterlist`).


Running step 5 will produce three different files:

1. ```filterlist.txt``` this will contain all the tracking link decorations along with the website on which they were observed, to be easily used to block such link decorations.
2. ```results.csv``` this contains information about each link decoration and the label assigned to the link decoration by the model. The CSV file contains three major pieces of information: features of the link decoration as observed during crawl, the label of link decoration in Ground Truth (Negative for non-tracking, Positive for tracking, and Unknown if missing in ground truth), and the label assigned by the model.
3. ```feature importance``` This file contains the most important features used by the model to identify tracking link decorations, this is related to the results discussed in Appendix of the cited paper.

### Additional Notes
Following are the two major claims of the paper "PURL: Safe and Effective Sanitization of Link Decoration" that can be verified using the results of the pipeline:

1. **PURL can effectively identify tracking link decorations:** This can be verified by checking the accuracy of the model on the test data. The accuracy of the model can be found in the ```accuracy``` file generated in step 4. PURL achieves 98% accuracy in classifying tracking link decorations.

2. **PURL can generate a filter list to sanitize tracking link decorations:** This can be verified by checking the filterlist generated in step 5. The filterlist contains all the tracking link decorations along with the website on which they were observed. The filterlist can be used to block such link decorations.

### Citation
This repository contains the code for the paper "PURL: Safe and Effective Sanitization of Link Decoration" accepted at the 31st USENIX Security Symposium. If you use this code, please consider citing the following paper:

```
@misc{munir2024purl,
    title={PURL: Safe and Effective Sanitization of Link Decoration},
    author={Shaoor Munir and Patrick Lee and Umar Iqbal and Zubair Shafiq and Sandra Siby},
    year={2024},
    eprint={2308.03417},
    archivePrefix={arXiv},
    primaryClass={cs.CR},
    url={https://arxiv.org/abs/2308.03417}
}
```
