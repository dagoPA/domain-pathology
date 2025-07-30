import boto3
from botocore.config import Config
from botocore import UNSIGNED
import sys
import threading
import os
import pandas as pd
from source.config.locations import get_dataset_dir, get_labels_csv_path

# --- Constants ---
BUCKET_NAME = 'camelyon-dataset'
S3_PREFIX = 'CAMELYON17/images/'


class ProgressPercentage(object):
    """A progress bar for Boto3 downloads.

    Displays the download progress of a file from S3, including the percentage
    and the amount of data transferred in megabytes.

    Parameters
    ----------
    filename : str
        The local name of the file being downloaded.
    size : float
        The total size of the file in bytes.

    """
    def __init__(self, filename, size):
        self._filename = filename
        self._size = float(size)
        self._seen_so_far = 0
        self._lock = threading.Lock()

    def __call__(self, bytes_amount):
        """Callable method invoked by Boto3 during file transfer.

        Parameters
        ----------
        bytes_amount : int
            The number of bytes transferred in the last chunk.

        """
        with self._lock:
            self._seen_so_far += bytes_amount
            percentage = (self._seen_so_far / self._size) * 100
            size_mb = self._size / (1024 * 1024)
            seen_so_far_mb = self._seen_so_far / (1024 * 1024)

            sys.stdout.write(
                f"\r-> Downloading {self._filename}: {seen_so_far_mb:.2f}MB / {size_mb:.2f}MB ({percentage:.2f}%)"
            )
            sys.stdout.flush()


def download_tif_files(max_patients=None):
    """Downloads TIFF files from the specified S3 prefix to the dataset directory.

    This function connects to the S3 bucket, lists all objects under the
    given prefix, filters for `.tif` files, and downloads each to the local
    dataset directory, skipping files that already exist and have a matching size.

    It can optionally limit the download to a specific number of patients based
    on the master labels CSV.

    Parameters
    ----------
    max_patients : int, optional
        If provided, limits the download to the slides from the first `max_patients`.
        Requires the `camelyon17-labels.csv` file to be present.
        If None (default), all .tif files are downloaded.

    Raises
    ------
    Exception
        Catches and prints any exceptions that occur during the S3
        operations or file handling.
    """
    s3_config = Config(signature_version=UNSIGNED)
    s3 = boto3.client('s3', config=s3_config)
    dataset_dir = get_dataset_dir()

    print(f"Dataset directory: {dataset_dir}")
    print(f"S3 Source: s3://{BUCKET_NAME}/{S3_PREFIX}")

    try:
        # Step 1: Get a map of all available .tif files on S3 for quick lookup
        print("Listing all available files from S3...")
        paginator = s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=BUCKET_NAME, Prefix=S3_PREFIX)

        s3_objects_map = {}
        for page in pages:
            if "Contents" in page:
                for obj in page["Contents"]:
                    if obj['Key'].endswith('.tif'):
                        s3_objects_map[os.path.basename(obj['Key'])] = obj

        if not s3_objects_map:
            print(f"No .tif files found at s3://{BUCKET_NAME}/{S3_PREFIX}")
            return

        # Step 2: Determine the target list of files to download
        target_filenames = []
        if max_patients is not None and max_patients > 0:
            print(f"Limiting download to the first {max_patients} patients.")
            labels_csv_path = get_labels_csv_path()
            if not os.path.exists(labels_csv_path):
                print(f"Error: Labels file not found at {labels_csv_path}", file=sys.stderr)
                print("Please run the `summarize_dataset` script first to generate it.", file=sys.stderr)
                return

            df = pd.read_csv(labels_csv_path)
            # Get unique patients in the order they appear
            all_patients = df['patient'].unique()
            # Select the subset of patients
            patient_subset = all_patients[:max_patients]
            # Get all slides for this subset of patients
            target_filenames = df[df['patient'].isin(patient_subset)]['slide'].tolist()
            print(f"Identified {len(target_filenames)} slides for {len(patient_subset)} patients.")
        else:
            print("Preparing to download all available slides.")
            target_filenames = list(s3_objects_map.keys())

        print(f"Found {len(target_filenames)} TIFF files to process.")

        # Step 3: Iterate through the target list and download
        for filename in target_filenames:
            if filename not in s3_objects_map:
                print(f"\nWarning: File '{filename}' is listed in labels but not found on S3. Skipping.")
                continue

            s3_object = s3_objects_map[filename]
            key = s3_object['Key']
            file_size = s3_object['Size']
            download_path = os.path.join(dataset_dir, filename)

            print(f"\nProcessing s3://{BUCKET_NAME}/{key}...")
            if os.path.exists(download_path) and os.path.getsize(download_path) == file_size:
                print(f"File '{filename}' already exists and size matches. Skipping download.")
            else:
                progress = ProgressPercentage(filename, file_size)
                s3.download_file(
                    Bucket=BUCKET_NAME,
                    Key=key,
                    Filename=download_path,
                    Callback=progress
                )
                sys.stdout.write("\n")
                print("Download completed successfully!")

    except Exception as e:
        sys.stdout.write("\n")
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    # Example of how to run with a limit from command line:
    # python -m source.dataset.download_dataset 10
    if len(sys.argv) > 1:
        try:
            patient_limit = int(sys.argv[1])
            print(f"Running download with a limit of {patient_limit} patients.")
            download_tif_files(max_patients=patient_limit)
        except ValueError:
            print("Invalid argument. Please provide an integer for max_patients.", file=sys.stderr)
            sys.exit(1)
    else:
        download_tif_files()