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

## Step 2: Crawl websites using OpenWPM
To crawl websites using OpenWPM, create a .csv file with the list of websites you want to crawl. The .csv file should have the website URLs in the ```url```column. The URLs should also have the protocol (http/https) included. For example:
```
url
https://www.example.com
https://www.example2.com
```
To run a crawl on the websites in the .csv file, use the following command:
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
