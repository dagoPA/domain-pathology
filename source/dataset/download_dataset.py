import boto3
from botocore.config import Config
from botocore import UNSIGNED
import sys
import threading
import os
from source.config.locations import get_output_dir, get_labels_dir

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


def download_tif_files():
    """Downloads all TIFF files from the specified S3 prefix to the dataset directory.

    This function connects to the S3 bucket, lists all objects under the
    given prefix, filters for `.tif` files, and downloads each to the local
    dataset directory, skipping files that already exist.

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
        paginator = s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=BUCKET_NAME, Prefix=S3_PREFIX)

        files_to_download = []
        for page in pages:
            if "Contents" in page:
                for obj in page["Contents"]:
                    if obj['Key'].endswith('.tif'):
                        files_to_download.append(obj)

        if not files_to_download:
            print(f"No .tif files found at s3://{BUCKET_NAME}/{S3_PREFIX}")
            return

        print(f"Found {len(files_to_download)} TIFF files to process.")

        for s3_object in files_to_download:
            key = s3_object['Key']
            file_size = s3_object['Size']
            filename = os.path.basename(key)
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
    download_tif_files()