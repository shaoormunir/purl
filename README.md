# PURL: Safe and Effective Sanitization of Link Decoration

This repository provides the necessary steps to run PURL using a customized version of OpenWPM. Follow the detailed instructions below to install dependencies, crawl websites, preprocess data, and run the classification pipeline.

## Step 1: Install OpenWPM Dependencies

First, navigate to the `OpenWPM` folder in the repository:
```bash
cd OpenWPM
```

Next, install the required dependencies by running the provided installation script:
```bash
./install.sh
```
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
python3 crawl_websites.py --websites <path_to_csv_file> --output_dir <output_directory> --num_browsers <number_of_browsers> --starting_index <starting_index> --log_file <log_file> --third_party_cookies <always|never>
```

### Explanation of Arguments:
- `--websites`: Path to the `.csv` file with the list of websites to crawl.
- `--output_dir`: Path to the directory where the crawl data will be stored.
- `--num_browsers`: Number of browsers to use for the crawl (recommended: 5-10).
- `--starting_index`: Index of the first website in the `.csv` file to crawl.
- `--log_file`: Path to the log file where crawl logs will be stored.
- `--third_party_cookies`: Whether to allow third-party cookies (`always` or `never`).

**Note:** OpenWPM processes websites in batches of 1,000. If your CSV file contains fewer than 1,000 websites, a single output file will be generated. For more than 1,000 websites, multiple output files will be created, each containing up to 1,000 websites' data. The output files will be stored in the specified `output_directory`.

## Step 3: Run Preprocessing for PURL

The next step involves generating a graph and its corresponding features for each website. To do this, navigate to the `Pipeline` folder and run the following command:
```bash
cd ../Pipeline
python3 run.py --features <path_to_feature_file> --folder <crawl_data_folder> --tag <file_tag>
```

### Explanation of Arguments:
- `--features`: Path to the file containing the features to extract (default: `features_new.yaml`).
- `--input_data`: Path to the folder where the crawl data was stored. For example, if the data was stored in a `datadir` directory within the `OpenWPM` folder, the path should be `../OpenWPM/datadir`.
- `--tag`: Tag to add to the output files. This helps differentiate between multiple experiments.

Running this command will generate four files for each run of crawled data:
- `features_n.csv`
- `graph_n.csv`
- `exfils_n.csv`
- `labels_n.csv`

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
- `--model_path`: Path to the saved model file.
- `--iterations`: Number of iterations needed to cover the complete dataset.
- `--generate-filterlist/--no-generate-filterlist`: Option to generate a filterlist based on the model results (`--generate-filterlist` or `--no-generate-filterlist`).
