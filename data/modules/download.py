import os
import requests
import shutil
import zipfile
from tqdm import tqdm

__all__ = ['download']

# https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs
url = "https://storage.googleapis.com/kaggle-data-sets/4538/7213/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20240714%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240714T151101Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=b2b4c046d709f2f0a37cb68d8495d0e2cf744395000279b8efb257c62e888b3802a4af6965d8d09a644fcc9ed35ba5b56f30846ad7bc3bc5d250544234e174af27a9f7df8502df2bcc3a4c06528665163873619646d14d476a67f706e25dc99cb23455f7c2989680bb7410c4ade2d2e4455f41e2f97cff587b045344639cad433b49244bde4cf5bc5f763e1537b39ea14fde474f7a725ecb75a8a21c97fe158c7e2e4ff42bfffa4ab179de11bd5d42a053f2c257b89190100137a78cb76ebe421633ec018b5f909f7ae5d81d09006a0c830c180ea68903e10dfd4647b2c55b7d10e674ecece7c4c18059200a03619f76ddc3ab600772f8042f5482703fe34af4"


def download():
    # Ensure the cache directory exists
    cache_dir = './data/cache'
    os.makedirs(cache_dir, exist_ok=True)

    # Check if the file already exists in the cache
    cache_path = os.path.join(cache_dir, 'archive.zip')
    if os.path.exists(cache_path):
        print("Using cached file.")
    else:
        # Request the file with streaming enabled
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kilobyte

        # Download the file with a progress bar
        with open(cache_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc='Downloading archive.zip') as pbar:
                for data in response.iter_content(block_size):
                    f.write(data)
                    pbar.update(len(data))
        print("Download complete.")

    # Ensure the unprocessed data directory exists
    unprocessed_dir = './data/unprocessed'
    os.makedirs(unprocessed_dir, exist_ok=True)

    # Extract the contents to `./data/unprocessed`
    with zipfile.ZipFile(cache_path, 'r') as zip_ref:
        zip_ref.extractall(unprocessed_dir)

    # Remove the unnecessary folders
    shutil.rmtree(os.path.join(unprocessed_dir, 'Data'), ignore_errors=True)
    shutil.rmtree(os.path.join(unprocessed_dir, 'ETFs'), ignore_errors=True)
    shutil.rmtree(os.path.join(unprocessed_dir, 'US'), ignore_errors=True)

    # Rename the extracted folder
    os.rename(os.path.join(unprocessed_dir, 'Stocks'), os.path.join(unprocessed_dir, 'US'))

    print("Extraction complete.")


# Call the function to execute the download and extraction
if __name__ == "__main__":
    download()
