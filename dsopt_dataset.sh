#!/bin/bash

# Define variables
GITHUB_URL="https://github.com/nbfigueroa/ds-opt/tree/master/datasets"
LOCAL_DIR="datasets"
USERNAME="nbfigueroa"
REPO="ds-opt"
BRANCH="master"
FOLDER_PATH="datasets"

# Create local directory if it doesn't exist
mkdir -p "$LOCAL_DIR"
echo "Created directory: $LOCAL_DIR (if it doesn't exist)"

echo "Downloading 3D files from GitHub using API..."
FILES=$(curl -s "https://api.github.com/repos/$USERNAME/$REPO/contents/$FOLDER_PATH" | grep "download_url" | cut -d '"' -f 4)

for FILE_URL in $FILES; do
  FILENAME=$(basename "$FILE_URL")
  
  # Check if filename starts with 3D
  if [[ "$FILENAME" == 3D* ]]; then
    if [ ! -f "$LOCAL_DIR/$FILENAME" ]; then
      echo "Downloading $FILENAME..."
      curl -L "$FILE_URL" -o "$LOCAL_DIR/$FILENAME"
    else
      echo "File $FILENAME already exists, skipping..."
    fi
  fi
done

echo "Download completed!"
