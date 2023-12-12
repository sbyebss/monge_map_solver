import argparse
import os

import pandas as pd


def main(parquet_path, output_folder):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    df = pd.read_parquet(parquet_path)
    # print(df.head())

    en_meta_data = df[df["LANGUAGE"] == "en"]
    # print(en_meta_data)
    en_meta_data.to_parquet(f"{output_folder}/laion-art-en.parquet")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a parquet file.")
    parser.add_argument("parquet_path", type=str, help="Path to the parquet file")
    parser.add_argument("output_folder", type=str, help="Path to the output folder")

    args = parser.parse_args()
    main(args.parquet_path, args.output_folder)
