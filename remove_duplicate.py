import os
import re
from collections import defaultdict


def extract_car_count_and_fid(filename):
    """
    Extract car_countID and fID from filename like 'car_1_id396645_f3.jpg'
    Returns: (car_countID, fID) or None if pattern doesn't match
    """
    pattern = r'car_(\d+)_id\d+_f(\d+)\.jpg'
    match = re.match(pattern, filename)
    if match:
        return match.group(1), match.group(2)
    return None


def remove_duplicate_images(output_dir="output"):
    """
    Traverse all subfolders in the output directory, find duplicate images based on car_countID and fID,
    and remove duplicates while keeping one file per unique combination.
    """
    total_removed = 0
    total_processed = 0

    print(f"[INFO] Starting duplicate removal process in {output_dir}")

    for root, dirs, files in os.walk(output_dir):
        # Group files by (car_countID, fID)
        car_count_fid_files = defaultdict(list)

        for filename in files:
            if filename.endswith('.jpg'):
                car_count_fid = extract_car_count_and_fid(filename)
                if car_count_fid:
                    car_count_fid_files[car_count_fid].append(filename)
                    total_processed += 1

        # Remove duplicates for each (car_countID, fID)
        for car_count_fid, file_list in car_count_fid_files.items():
            if len(file_list) > 1:
                print(f"[INFO] Found {len(file_list)} duplicate files for {car_count_fid} in {root}")
                print(f"[INFO] Files: {file_list}")

                # Keep the first file, remove the rest
                files_to_remove = file_list[1:]
                for file_to_remove in files_to_remove:
                    file_path = os.path.join(root, file_to_remove)
                    try:
                        os.remove(file_path)
                        print(f"[INFO] Removed duplicate: {file_to_remove}")
                        total_removed += 1
                    except Exception as e:
                        print(f"[ERROR] Failed to remove {file_to_remove}: {e}")

    print(f"[INFO] Duplicate removal completed!")
    print(f"[INFO] Total files processed: {total_processed}")
    print(f"[INFO] Total duplicates removed: {total_removed}")


if __name__ == "__main__":
    remove_duplicate_images()
