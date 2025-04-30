import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import time
import warnings

warnings.filterwarnings("ignore")


def count_images_in_folder(folder_path: Path) -> int:
    """
    Count the number of PNG images in a folder.
    """
    if not folder_path.exists():
        return 0
    return sum(1 for f in folder_path.iterdir() if f.suffix == '.png')


def process_row(args):
    """
    Worker function to process one row of the DataFrame.
    """
    i, row, base_path = args
    folder_name = str(row['AccessionNumber']).zfill(6)
    full_path = base_path / folder_name
    return i, count_images_in_folder(full_path)


if __name__ == "__main__":
    start_time = time.time()

    # Define paths
    folder_path = Path("/data/groups/public/derived/Kaggle_SPR_Screening_Mammography/pngs/processed_png")
    csv_path = Path("/projects/xin-275d/challenges/Kaggle_SPR/data_csv/train.csv")
    output_csv_path = Path("/projects/xin-275d/challenges/Kaggle_SPR/data_csv/train_updated.csv")

    # Load CSV
    print("Loading CSV...")
    df = pd.read_csv(csv_path)
    df['AccessionNumber'] = df['AccessionNumber'].astype(str).str.zfill(6)
    df['image_count'] = 0

    # Prepare arguments
    args = [(i, row, folder_path) for i, row in enumerate(df.to_dict('records'))]

    print(f"Counting images using {cpu_count()} processes...")

    # Use imap_unordered for progress bar
    image_counts = [0] * len(df)
    with Pool(cpu_count()) as pool:
        for i, count in tqdm(pool.imap_unordered(process_row, args), total=len(df), desc="Processing"):
            image_counts[i] = count

    # Update DataFrame
    df['image_count'] = image_counts

    # Save CSV
    print("Saving updated CSV...")
    df.to_csv(output_csv_path, index=False)

    elapsed = time.time() - start_time
    print(f"Done! Total time: {elapsed:.2f} seconds.")
