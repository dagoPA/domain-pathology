import sys
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import boto3
from botocore.config import Config
from botocore import UNSIGNED
from source.config.locations import get_output_dir, get_labels_dir

def generate_full_c17_dataframe():
	"""
	Programmatically builds a DataFrame representing the entire 1000-slide
	CAMELYON17 dataset based on its known structure.
	"""
	print("Generating a map of the complete 1000-slide C17 dataset...")
	records = []
	centers = ['CWZ', 'LPON', 'RST', 'RUMC', 'UMCU']

	for i in range(200):  # 200 total patients
		patient_uid = f'patient_{i:03d}'
		dataset_type = 'Training' if i < 100 else 'Testing'
		center_name = centers[(i % 100) // 20]  # Assigns centers in blocks of 20 for both train and test

		for j in range(5):  # 5 nodes (slides) per patient
			slide_id = f'{patient_uid}_node_{j}.tif'
			records.append({
				'slide_id': slide_id,
				'patient_uid': patient_uid,
				'set': dataset_type,
				'center_name': center_name
			})

	return pd.DataFrame(records)

def load_c17_labels_from_s3(s3_client, bucket, prefix):
	"""Loads the official C17 training labels from the GigaDB/Wasabi mirror."""
	metadata_key = f"{prefix}/CAMELYON17/training/stage_labels.csv"
	print(f"Loading labels from official source: {metadata_key}")
	try:
		df = pd.read_csv(BytesIO(s3_client.get_object(Bucket=bucket, Key=metadata_key)['Body'].read()))
		df.rename(columns={'patient': 'slide_id', 'stage': 'label'}, inplace=True)
		# Filter for slide-level data (.tif)
		df = df[df['slide_id'].str.endswith('.tif', na=False)].copy()
		return df[['slide_id', 'label']]
	except Exception as e:
		print(f"Warning: Could not process C17 metadata. Error: {e}", file=sys.stderr)
		return pd.DataFrame()


def plot_and_save_statistics(df, output_dir):
	"""
	Generates and saves a 2x2 grid of plots.
	- Label-based plots use only the training set data.
	- Site-based plots use the full 1000-slide dataset.
	"""
	plt.rcParams['font.family'] = 'serif'
	plt.rcParams['font.serif'] = ['DejaVu Serif', 'Times New Roman', 'serif']
	sns.set_theme(style="whitegrid")

	fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=300)
	fig.suptitle('Statistical Overview of the CAMELYON17 Dataset')
	bbox_props = dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.6)

	# --- Data for plots: Train-only for labels, Full dataset for sites ---
	df_train = df[df['set'] == 'Training'].dropna(subset=['label'])
	df_full = df

	# Plot 1: WSI Distribution by Detailed Label (Training Set)
	ax1 = sns.countplot(ax=axes[0, 0], x='label', data=df_train, palette='viridis', hue='label', legend=False,
	                    order=df_train['label'].value_counts().index)
	axes[0, 0].set_title('WSI Label Distribution (Training Set)')
	axes[0, 0].set_xlabel('Metastasis Label')
	axes[0, 0].set_ylabel('WSI Count')
	axes[0, 0].tick_params(axis='x', rotation=45)
	for p in ax1.patches:
		ax1.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center',
		             va='bottom', xytext=(0, 3), textcoords='offset points', fontsize=8, bbox=bbox_props)

	# Plot 2: WSI Distribution by Contributing Center (Full Dataset)
	ax2 = sns.countplot(ax=axes[0, 1], y='center_name', data=df_full, palette='plasma', hue='center_name', legend=False,
	                    order=df_full['center_name'].value_counts().index)
	axes[0, 1].set_title('WSI Distribution by Center (Full Dataset)')
	axes[0, 1].set_xlabel('WSI Count')
	axes[0, 1].set_ylabel('Contributing Center')


	# Plot 3: Patient Distribution by Detailed Label (Training Set)
	patient_class_counts = df_train.groupby('label')['patient_uid'].nunique().reset_index()
	ax3 = sns.barplot(ax=axes[1, 0], x='label', y='patient_uid', data=patient_class_counts, palette='viridis',
	                  hue='label', legend=False,
	                  order=patient_class_counts.sort_values('patient_uid', ascending=False).label)
	axes[1, 0].set_title('Patient Label Distribution (Training Set)')
	axes[1, 0].set_xlabel('Metastasis Label')
	axes[1, 0].set_ylabel('Unique Patient Count')
	axes[1, 0].tick_params(axis='x', rotation=45)
	for p in ax3.patches:
		ax3.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center',
		             va='bottom', xytext=(0, 3), textcoords='offset points', fontsize=8, bbox=bbox_props)

	# Plot 4: Patient Distribution by Contributing Center (Full Dataset)
	patient_site_counts = df_full.groupby('center_name')['patient_uid'].nunique().reset_index()
	ax4 = sns.barplot(ax=axes[1, 1], y='center_name', x='patient_uid', data=patient_site_counts, palette='plasma',
	                  hue='center_name', legend=False,
	                  order=patient_site_counts.sort_values('patient_uid', ascending=False).center_name)
	axes[1, 1].set_title('Patient Distribution by Center (Full Dataset)')
	axes[1, 1].set_xlabel('Unique Patient Count')
	axes[1, 1].set_ylabel('Contributing Center')
	for p in ax4.patches:
		ax4.annotate(f'{int(p.get_width())}', (p.get_y() + p.get_height() / 2., p.get_width()), ha='left', va='center',
		             xytext=(3, 0), textcoords='offset points', fontsize=8, bbox=bbox_props)

	plt.tight_layout(rect=[0, 0.03, 1, 0.95])

	# Changed basename for output files
	plot_basename = 'dataset-summary'
	plot_path = os.path.join(output_dir, f'{plot_basename}.png')
	plt.savefig(plot_path, dpi=300)
	print(f"Plot saved to {plot_path}")
	plt.show()
	return plot_basename


def generate_dataset_statistics():
	"""Main function to orchestrate the analysis from the known dataset structure."""
	WASABI_ENDPOINT_URL = 'https://s3.ap-northeast-1.wasabisys.com'
	BUCKET_NAME = 'gigadb-datasets'
	PREFIX = 'live/pub/10.5524/100001_101000/100439'
	output_dir = get_output_dir()

	# 1. Generate the complete 1000-slide map of the dataset
	master_df = generate_full_c17_dataframe()

	# 2. Load the training labels from the official metadata file
	s3_config = Config(signature_version=UNSIGNED)
	s3 = boto3.client('s3', config=s3_config, endpoint_url=WASABI_ENDPOINT_URL)
	labels_df = load_c17_labels_from_s3(s3, BUCKET_NAME, PREFIX)

	# 3. Merge labels into the master map
	final_df = pd.merge(master_df, labels_df, on='slide_id', how='left')

	# 4. Binarize labels for ML convenience: 1=Positive, 0=Negative, NaN=Test/Unlabeled
	positive_labels = ['macro', 'micro', 'itc']
	final_df['binary_label'] = final_df['label'].apply(
		lambda x: 1 if x in positive_labels else (0 if pd.notna(x) else pd.NA)
	)

	# 5. Generate and save plot using the original column names
	plot_basename = plot_and_save_statistics(final_df, output_dir)

	# 6. NOW, rename columns for the final CSV.
	# We use inplace=True to modify the DataFrame directly.
	final_df.rename(columns={'slide_id': 'slide', 'patient_uid': 'patient', 'label': 'detailed_label',
	                         'center_name': 'domain'}, inplace=True)

	# 7. Save the master dataset CSV with the new column names
	master_csv_path = get_labels_dir() + '/camelyon17-labels.csv'
	# this is a test
	final_df.to_csv(master_csv_path, index=False)
	print(f"Processed master dataset saved to {master_csv_path}")

	# 8. Create and save summary statistics CSV
	if plot_basename:
		# Filter for the training set (where labels are known) using the original 'set' column
		df_train_summary = final_df[final_df['set'] == 'Training'].dropna(subset=['detailed_label'])
		summary_df = pd.DataFrame([
			# Use new column names for the summary
			{'statistic': 'Total WSI in Training Set', 'value': df_train_summary['slide'].nunique()},
			{'statistic': 'Total Patients in Training Set', 'value': df_train_summary['patient'].nunique()}
		])
		summary_csv_path = os.path.join(output_dir, f'{plot_basename}.csv')
		summary_df.to_csv(summary_csv_path, index=False)
		print(f"Summary statistics for Training Set saved to {summary_csv_path}")


if __name__ == '__main__':
	generate_dataset_statistics()