import lzma
import os

LIDAR_DIR = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..", "data", "wind_data", "lidar"
)


def unzip_all_7z_files(directory):
    """Extract all LZMA-compressed .7z archives in a directory in-place.

    Each .rtd.7z file is decompressed to a .rtd file in the same directory.

    Args:
        directory (str): Path to the directory containing .7z files.
    """
    archives = [f for f in os.listdir(directory) if f.endswith(".7z")]
    print(f"Found {len(archives)} archive(s) to extract.")

    for filename in archives:
        archive_path = os.path.join(directory, filename)
        out_path = os.path.join(directory, filename[:-3])  # strip .7z
        if os.path.exists(out_path):
            print(f"Skipping (already extracted): {filename}")
            continue
        print(f"Extracting: {filename}")
        with lzma.open(archive_path, "rb", format=lzma.FORMAT_ALONE) as f_in:
            with open(out_path, "wb") as f_out:
                f_out.write(f_in.read())

    print("Done.")


if __name__ == "__main__":
    unzip_all_7z_files(LIDAR_DIR)
