# PURL: Safe and Effective Sanitization of Link Decoration
To run PURL, you first need to run OpenWPM on a select subset of websites. PURL relies on a customized version of OpenWPM which is included in the ```OpenWPM``` folder of this repository.

## Step 1: Install OpenWPM dependencies
To run OpenWPM, first navigate to the OpenWPM folder and install the required dependencies using the following command:
```
./install.sh
```
This will create a conda environment called ```openwpm``` and install the required dependencies. Next, activate the conda environment using the following command:
```
conda activate openwpm
```

We will use this environment for the rest of the pipeline

## Step 2: Crawl websites using OpenWPM
To crawl websites using OpenWPM, create a .csv file with the list of websites you want to crawl. The .csv file should have the website URLs in the ```url```column. The URLs should also have the protocol (http/https) included. For example:
```
url
https://www.example.com
https://www.example2.com
```
To run a crawl on the websites in the .csv file, use the following command in the OpenWPM folder:
```
python3 crawl_websites.py --websites <path_to_csv_file> --output_dir <output_directory> --num_browsers <number_of_browsers> --starting_index <starting index of the csv file> --log_file <log_file> --third_party_cookies <always,never>
```
This is what each argument does:
- ```--websites```: Path to the .csv file with the list of websites to crawl.
- ```--output_dir```: Path to the directory where the crawl data will be stored.
- ```--num_browsers```: Number of browsers to use for the crawl.
- ```--starting_index```: Index of the first website in the .csv file to crawl.
- ```--log_file```: Path to the log file where the crawl logs will be stored.
- ```--third_party_cookies```: Whether to allow third-party cookies. Options are ```always``` and ```never```.

OpenWPM will crawl files in batches of 1,000, so if your CSV file has less than 1,000 sites, only one file will be created as an output containing all the website data. However, if there are more than 1,000 websites, multiple files will be created. The output files will be stored in the ```<output_directory>``` specified in the command.

## Step 3: Run preprocessing for PURL
Next step is to generate a graph and its corresponding features for each website. To do this, navigate to the  run the following command:
```
python3 run.py --features <path_to_feature_file> --folder <craw_data_folder> --tag <file_tag>
```
This is what each argument does:
- ```--features```: Name of the file containing the features to extract, by default it should be ```features_new.yaml```.
- ```--folder```: Path of the folder where the crawl data was stored. For example if the data was stored in a ```datadir``` directory in the ```OpenWPM``` folder, the path should be ```../OpenWPM/datadir```
- ```--tag```: Tag to add to the output files. This is useful when running multiple experiments.

Running this command will generate three files for each run of crawled data. For example, if you crawled 100 websites, you will have one subfolder called ```datadir-0```in the ```datadir``` folder. This command with generate four files: ```features_0.csv```, ```graph_0.csv```,```exfils_0.csv```, and ```labels_0.csv``` in the same directory as the ```run.py``` file.

## Step 4: Run the classification pipeline for PURL