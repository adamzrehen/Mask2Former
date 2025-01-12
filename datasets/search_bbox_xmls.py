import os
import re
from collections import Counter


def get_file_sizes(folders):
    file_sizes = []
    file_paths = []

    for folder in folders:
        for root, _, files in os.walk(folder):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    file_size = os.path.getsize(file_path)
                    file_sizes.append(file_size)
                    file_paths.append(file_path)
                except OSError as e:
                    print(f"Error reading file size for {file_path}: {e}")

    return file_sizes, file_paths


def find_most_frequent_range(file_sizes, range_size=10):
    rounded_sizes = [size // range_size * range_size for size in file_sizes]
    most_common_range, _ = Counter(rounded_sizes).most_common(1)[0]
    return most_common_range, most_common_range + range_size


def find_outliers(file_sizes, file_paths, frequent_range, threshold=100):
    outliers = []
    for size, path in zip(file_sizes, file_paths):
        if size < frequent_range[0] - threshold or size > frequent_range[1] + threshold:
            outliers.append((path, size))
    return outliers


def main(folders):
    file_sizes, file_paths = get_file_sizes(folders)
    if not file_sizes:
        print("No files found.")
        return

    most_frequent_range = find_most_frequent_range(file_sizes)
    print(f"Most frequent size range: {most_frequent_range[0]} to {most_frequent_range[1]} bytes")

    outliers = find_outliers(file_sizes, file_paths, most_frequent_range)

    sorted_files = sorted(
        outliers,
        key=lambda x: int(re.search(r'_(\d+)\.xml$', x[0]).group(1)) if re.search(r'_(\d+)\.xml$', x[0]) else float(
            'inf')
    )
    first_xml = sorted_files[0]


# Example usage
folders = ["/home/adam/Documents/Data/REAL-Colon Dataset/001-001_annotations"]  # Replace with your folders
main(folders)