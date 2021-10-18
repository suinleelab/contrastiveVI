"""Data preprocessing utilities."""
import requests


def download_binary_file(file_url: str, output_path: str) -> None:
    """Download binary data file from a URL.

    Args:
        file_url: URL where the file is hosted.
        output_path: Output path for the downloaded file.

    Returns:
        None.
    """
    request = requests.get(file_url)
    with open(output_path, "wb") as f:
        f.write(request.content)
    print(f"Downloaded data from {file_url} at {output_path}")
