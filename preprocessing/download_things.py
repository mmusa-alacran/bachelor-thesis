#!/usr/bin/env python3
"""
Download THINGS images and verify they match the CSV mapping.

Usage:
    # Step 1: Install osfclient
    pip install osfclient

    # Step 2: Download the images zip from OSF
    osf -p jum2f fetch osfstorage/images_THINGS.zip ./images_THINGS.zip

    # Step 3: Get the password
    osf -p jum2f fetch osfstorage/password_images.txt ./password_images.txt

    # Step 4: Extract, using the password from the file fetched above
    unzip -P <PASSWORD> images_THINGS.zip -d THINGS_images/

    # Step 5: Verify all images are present
    python3 download_things.py --verify --things-dir THINGS_images --csv-dir data_samples/

This script verifies that downloaded THINGS images match the experimental CSV mapping.
"""

import argparse
import os
import sys
import csv


def verify_images(things_dir: str, csv_dir: str):
    """Verify all images referenced in the CSVs exist in the THINGS directory."""

    for split, csv_name in [("train", "THINGS_train_imgs_paths.csv"),
                            ("test", "THINGS_test_imgs_paths.csv")]:
        csv_path = os.path.join(csv_dir, csv_name)
        if not os.path.isfile(csv_path):
            print(f"ERROR: CSV not found: {csv_path}")
            continue

        total = 0
        found = 0
        missing = []

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row_i, row in enumerate(reader, start=1):
                things_path = row["THINGS_image_path"]  # e.g. "aardvark/aardvark_01b.jpg"
                total += 1

                # Try finding the image in the THINGS directory
                # The THINGS database might have images under:
                #   THINGS_images/<category>/<filename>.jpg
                # or possibly in a subdirectory structure
                full_path = os.path.join(things_dir, things_path)

                if os.path.isfile(full_path):
                    found += 1
                else:
                    missing.append((row_i, things_path))

        print(f"\n{'='*60}")
        print(f"[{split.upper()}] {csv_name}")
        print(f"  Total: {total}, Found: {found}, Missing: {total - found}")

        if missing:
            print(f"  First 10 missing:")
            for idx, path in missing[:10]:
                print(f"    Row {idx}: {path}")
        else:
            print(f"  OK: all {split} images found")


def list_structure(things_dir: str):
    """Show the directory structure of the downloaded THINGS images."""
    if not os.path.isdir(things_dir):
        print(f"ERROR: Directory not found: {things_dir}")
        return

    # Count categories and images
    categories = []
    total_images = 0
    for entry in sorted(os.listdir(things_dir)):
        subdir = os.path.join(things_dir, entry)
        if os.path.isdir(subdir):
            n_imgs = len([f for f in os.listdir(subdir)
                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
            categories.append((entry, n_imgs))
            total_images += n_imgs

    print(f"\nTHINGS directory: {things_dir}")
    print(f"  Categories: {len(categories)}")
    print(f"  Total images: {total_images}")
    if categories:
        print(f"  First 5: {', '.join(f'{c[0]} ({c[1]})' for c in categories[:5])}")
        print(f"  Last 5:  {', '.join(f'{c[0]} ({c[1]})' for c in categories[-5:])}")


def main():
    p = argparse.ArgumentParser(description="Verify THINGS images against CSV mapping")
    p.add_argument("--things-dir", type=str, default="THINGS_images",
                   help="Path to extracted THINGS images directory")
    p.add_argument("--csv-dir", type=str, default="data_samples/",
                   help="Directory containing THINGS_train/test_imgs_paths.csv")
    p.add_argument("--verify", action="store_true",
                   help="Verify all CSV-referenced images exist")
    p.add_argument("--structure", action="store_true",
                   help="Show directory structure")
    args = p.parse_args()

    if args.structure or not args.verify:
        list_structure(args.things_dir)

    if args.verify:
        verify_images(args.things_dir, args.csv_dir)


if __name__ == "__main__":
    main()
