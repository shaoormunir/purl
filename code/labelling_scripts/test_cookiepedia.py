import pandas as pd

def find_sites(df):

	name = df['name'].iloc[0] + "_" + df['domain'].iloc[0]
	if len(df) > 20:
		print(name, len(df), df['cat_id'].tolist())

def process_cp(df):

	df.groupby(['name', 'domain']).apply(find_sites)

if __name__ == "__main__":

	cookiepedia = "cookiepedia.csv"
	df = pd.read_csv(cookiepedia)

	process_cp(df)