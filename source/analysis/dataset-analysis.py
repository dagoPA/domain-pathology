import boto3
import pandas as pd
from io import BytesIO
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
from botocore.config import Config
from botocore import UNSIGNED

# Se asume que 'source.config' es un módulo local que proporciona las funciones
# get_output_dir() y get_dataset_dir(). Si este módulo no existe,
# necesitarás reemplazar las llamadas a estas funciones con rutas codificadas
# o con otro método de configuración.
from source.config import get_output_dir, get_dataset_dir


def get_patient_uid(row):
	"""
	Creates a unique patient identifier from a slide ID row.

	This function standardizes the patient ID extraction from different naming
	conventions found in the dataset.

	Parameters
	----------
	row : pandas.Series
	   A row from the DataFrame, which must contain an 'id' column with the
	   slide identifier.

	Returns
	-------
	str
	   The extracted unique patient identifier.
	"""
	if 'patient_' in row['id']:
		# Handles format like 'patient_001_node_4.tif'
		return '_'.join(row['id'].split('_')[:2])
	else:
		# Handles format like 'test_001.tif'
		return row['id'].split('.')[0]


def process_camelyon16(s3_client, bucket):
	"""
	Downloads and processes the CAMELYON16 metadata file from S3.

	It renames columns for consistency, constructs the full S3 path for each
	image, and maps center IDs to their respective names.

	Parameters
	----------
	s3_client : boto3.client
	   An initialized S3 client for interacting with AWS S3.
	bucket : str
	   The name of the S3 bucket containing the CAMELYON16 dataset.

	Returns
	-------
	pandas.DataFrame
	   A DataFrame containing the processed metadata for CAMELYON16. Returns
	   an empty DataFrame if processing fails.
	"""
	try:
		df = pd.read_csv(
			BytesIO(s3_client.get_object(Bucket=bucket, Key='CAMELYON16/evaluation/reference.csv')['Body'].read()))
		df.rename(columns={'image': 'id', 'class': 'label', 'type': 'task', 'center': 'center_id'}, inplace=True)
		df['path'] = df['id'].apply(lambda x: f"CAMELYON16/images/{x}")
		df['center_name'] = df['center_id'].apply(lambda cid: {0: 'RUMC', 1: 'UMCU'}.get(cid, 'Unknown_C16'))
		return df[['id', 'label', 'path', 'task', 'center_name']]
	except Exception as e:
		print(f"Warning: Could not process CAMELYON16 metadata. Error: {e}", file=sys.stderr)
		return pd.DataFrame()


def process_camelyon17(s3_client, bucket):
	"""
	Downloads and processes the CAMELYON17 metadata file from S3.

	It renames columns, filters for '.tif' files, constructs the image path,
	and derives the center name from the patient ID.

	Parameters
	----------
	s3_client : boto3.client
	   An initialized S3 client for interacting with AWS S3.
	bucket : str
	   The name of the S3 bucket containing the CAMELYON17 dataset.

	Returns
	-------
	pandas.DataFrame
	   A DataFrame containing the processed metadata for CAMELYON17. Returns
	   an empty DataFrame if processing fails.
	"""
	try:
		df = pd.read_csv(BytesIO(s3_client.get_object(Bucket=bucket, Key='CAMELYON17/stages.csv')['Body'].read()))
		df.rename(columns={'patient': 'id', 'stage': 'label'}, inplace=True)
		df = df[df['id'].str.endswith('.tif', na=False)].copy()
		df['path'] = df['id'].apply(lambda x: f"CAMELYON17/images/{x}")
		df['task'] = 'Staged'
		df['center_name'] = df['id'].apply(lambda pid: {
			0: 'CWZ', 1: 'LPON', 2: 'RST', 3: 'RUMC', 4: 'UMCU'
		}.get(int(pid.split('_')[1]) // 20, 'Unknown_C17'))
		return df[['id', 'label', 'path', 'task', 'center_name']]
	except Exception as e:
		print(f"Warning: Could not process CAMELYON17 metadata. Error: {e}", file=sys.stderr)
		return pd.DataFrame()


def get_all_existing_image_keys(s3_client, bucket):
	"""
	Scans S3 to find all existing .tif image keys in the dataset directories.

	This is used to verify that the images listed in the metadata files
	actually exist in the S3 bucket before including them in the final dataset.

	Parameters
	----------
	s3_client : boto3.client
	   An initialized S3 client for interacting with AWS S3.
	bucket : str
	   The name of the S3 bucket to scan.

	Returns
	-------
	set
	   A set of S3 keys for all found '.tif' images.
	"""
	existing_keys = set()
	for prefix in ['CAMELYON16/images/', 'CAMELYON17/images/']:
		paginator = s3_client.get_paginator('list_objects_v2')
		pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
		for page in pages:
			if "Contents" in page:
				for obj in page['Contents']:
					if obj['Key'].endswith('.tif'):
						existing_keys.add(obj['Key'])
	return existing_keys


def plot_and_save_statistics(df, output_dir):
	"""
	Generates and saves a 2x2 grid of plots summarizing the dataset.

	The plots show the distribution of Whole Slide Images (WSIs) and unique
	patients by metastasis status and by the hospital/site of origin.

	Parameters
	----------
	df : pandas.DataFrame
	   The verified DataFrame containing the combined dataset statistics.
	output_dir : str
	   The directory where the output plot image will be saved.

	Returns
	-------
	str
		The base name of the plot file.
	"""
	if df.empty:
		print("No verified data to plot.")
		return None

	sns.set_theme(style="whitegrid")

	# --- PLOT STYLING CONFIGURATION ---
	plt.rcParams['font.family'] = 'serif'
	plt.rcParams['font.serif'] = ['DejaVu Serif', 'Bitstream Vera Serif', 'Times New Roman', 'serif']
	plt.rcParams['font.size'] = 10
	plt.rcParams['axes.titlesize'] = 12
	plt.rcParams['axes.labelsize'] = 10
	plt.rcParams['xtick.labelsize'] = 8
	plt.rcParams['ytick.labelsize'] = 8
	plt.rcParams['figure.titlesize'] = 14
	sns.set_theme(style="whitegrid")

	fig, axes = plt.subplots(2, 2, figsize=(8, 7), dpi=300)
	fig.suptitle('Statistical Overview of the CAMELYON Dataset for Metastasis Classification')

	bbox_props = dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.6)

	# --- PLOT 1: WSI Distribution by Metastasis Presence ---
	total_wsi = len(df)
	ax1 = sns.countplot(ax=axes[0, 0], x='binary_class', data=df, palette='viridis', hue='binary_class', legend=False)
	axes[0, 0].set_title('WSI Distribution by Metastasis Presence')
	axes[0, 0].set_xlabel('Metastasis Presence')
	axes[0, 0].set_ylabel('WSI Count')
	for p in ax1.patches:
		ax1.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center',
		             va='bottom', xytext=(0, 3), textcoords='offset points', fontsize=8, bbox=bbox_props)

	# --- PLOT 2: WSI Distribution by Hospital / Site ---
	total_sites = df['center_name'].nunique()
	ax2 = sns.countplot(ax=axes[0, 1], y='center_name', data=df, palette='plasma', hue='center_name', legend=False,
	                    order=df['center_name'].value_counts().index)
	axes[0, 1].set_title('WSI Distribution by Hospital / Site')
	axes[0, 1].set_xlabel('WSI Count')
	axes[0, 1].set_ylabel('Hospital / Site')
	for p in ax2.patches:
		ax2.annotate(f'{int(p.get_width())}', (p.get_width(), p.get_y() + p.get_height() / 2.), ha='left', va='center',
		             xytext=(3, 0), textcoords='offset points', fontsize=8, bbox=bbox_props)

	# --- PLOT 3: Patient Distribution by Metastasis Presence ---
	total_patients = df['patient_uid'].nunique()
	patient_class_counts = df.groupby('binary_class')['patient_uid'].nunique().reset_index()
	ax3 = sns.barplot(ax=axes[1, 0], x='binary_class', y='patient_uid', data=patient_class_counts, palette='viridis',
	                  hue='binary_class', legend=False)
	axes[1, 0].set_title('Patient Distribution by Metastasis Presence')
	axes[1, 0].set_xlabel('Metastasis Presence')
	axes[1, 0].set_ylabel('Unique Patient Count')
	for p in ax3.patches:
		ax3.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center',
		             va='bottom', xytext=(0, 3), textcoords='offset points', fontsize=8, bbox=bbox_props)

	# --- PLOT 4: Patient Distribution by Hospital / Site ---
	patient_site_counts = df.groupby('center_name')['patient_uid'].nunique().reset_index()
	ax4 = sns.barplot(ax=axes[1, 1], y='center_name', x='patient_uid', data=patient_site_counts, palette='plasma',
	                  hue='center_name', legend=False,
	                  order=patient_site_counts.sort_values('patient_uid', ascending=False).center_name)
	axes[1, 1].set_title('Patient Distribution by Hospital / Site')
	axes[1, 1].set_xlabel('Unique Patient Count')
	axes[1, 1].set_ylabel('Hospital / Site')
	for p in ax4.patches:
		ax4.annotate(f'{int(p.get_width())}', (p.get_width(), p.get_y() + p.get_height() / 2.), ha='left', va='center',
		             xytext=(3, 0), textcoords='offset points', fontsize=8, bbox=bbox_props)

	plt.tight_layout(rect=[0, 0.03, 1, 0.95])

	# --- SAVE THE PLOT ---
	plot_basename = 'camelyon_statistical_overview'
	plot_path = os.path.join(output_dir, f'{plot_basename}.png')
	plt.savefig(plot_path, dpi=300)
	print(f"Plot saved to {plot_path}")
	plt.show()
	return plot_basename


def main_analysis(bucket_name):
	"""
	Main function to orchestrate the download, processing, and analysis of CAMELYON data.

	It fetches data for CAMELYON16 and CAMELYON17, combines them, verifies
	image existence in S3, generates a master dataset CSV, a summary plot,
	and a summary statistics CSV.

	Parameters
	----------
	bucket_name : str
	   The name of the public S3 bucket where the CAMELYON dataset is stored.
	"""
	output_dir = get_output_dir()
	dataset_dir = get_dataset_dir()

	s3_config = Config(signature_version=UNSIGNED)
	s3 = boto3.client('s3', config=s3_config)

	df16 = process_camelyon16(s3, bucket_name)
	df17 = process_camelyon17(s3, bucket_name)
	combined_df = pd.concat([df16, df17], ignore_index=True)

	existing_keys = get_all_existing_image_keys(s3, bucket_name)
	verified_df = combined_df[combined_df['path'].isin(existing_keys)].copy()

	if verified_df.empty:
		print("No data could be verified against S3 keys. Cannot generate report.", file=sys.stderr)
		return

	verified_df['binary_class'] = verified_df['label'].apply(
		lambda x: 'Positive' if x in ['Tumor', 'macro', 'micro', 'itc'] else 'Negative'
	)
	verified_df['patient_uid'] = verified_df.apply(get_patient_uid, axis=1)

	# --- CREATE AND SAVE THE MASTER DATASET ---
	# 1. Create the processed master dataset with specific columns: [slide, patient, domain, label]
	master_df = verified_df.copy()
	master_df['domain'] = master_df['center_name']
	master_df['label'] = master_df['label'].apply(
		lambda x: 1 if x in ['Tumor', 'macro', 'micro', 'itc'] else 0
	)

	# Select and rename columns as requested
	final_master_df = master_df[['id', 'patient_uid', 'domain', 'label']].copy()
	final_master_df.rename(columns={'id': 'slide', 'patient_uid': 'patient'}, inplace=True)

	# Save the final master dataset to the 'datasets' directory
	master_csv_path = os.path.join(dataset_dir, 'camelyon_master_dataset.csv')
	final_master_df.to_csv(master_csv_path, index=False)
	print(f"Processed master dataset saved to {master_csv_path}")

	# --- PLOT AND SAVE VISUALIZATIONS ---
	plot_basename = plot_and_save_statistics(verified_df, output_dir)

	# --- CREATE AND SAVE THE SUMMARY STATISTICS CSV ---
	# 2. Create a summary of statistics and save it to the 'output' directory
	# with the same name as the plot.
	if plot_basename:
		# Calculate statistics
		total_wsi = len(verified_df)
		total_patients = verified_df['patient_uid'].nunique()
		total_sites = verified_df['center_name'].nunique()

		slides_per_class = verified_df['binary_class'].value_counts()
		patients_per_class = verified_df.groupby('binary_class')['patient_uid'].nunique()

		slides_per_site = verified_df['center_name'].value_counts()
		patients_per_site = verified_df.groupby('center_name')['patient_uid'].nunique()

		# Assemble summary data into a list of dictionaries
		summary_data = []
		summary_data.append({'Category': 'Overall', 'Statistic': 'Total WSI', 'Value': total_wsi})
		summary_data.append({'Category': 'Overall', 'Statistic': 'Total Patients', 'Value': total_patients})
		summary_data.append({'Category': 'Overall', 'Statistic': 'Total Sites', 'Value': total_sites})

		for class_name, count in slides_per_class.items():
			summary_data.append({'Category': 'By Class', 'Statistic': f'WSI Count ({class_name})', 'Value': count})
		for class_name, count in patients_per_class.items():
			summary_data.append({'Category': 'By Class', 'Statistic': f'Patient Count ({class_name})', 'Value': count})

		for site_name, count in slides_per_site.items():
			summary_data.append({'Category': 'By Site', 'Statistic': f'WSI Count ({site_name})', 'Value': count})
		for site_name, count in patients_per_site.items():
			summary_data.append({'Category': 'By Site', 'Statistic': f'Patient Count ({site_name})', 'Value': count})

		summary_df = pd.DataFrame(summary_data)

		# Save the summary dataframe to the 'output' directory
		summary_csv_path = os.path.join(output_dir, f'{plot_basename}.csv')
		summary_df.to_csv(summary_csv_path, index=False)
		print(f"Summary statistics saved to {summary_csv_path}")


if __name__ == '__main__':
	# The bucket name for the public CAMELYON dataset.
	main_analysis(bucket_name='camelyon-dataset')
