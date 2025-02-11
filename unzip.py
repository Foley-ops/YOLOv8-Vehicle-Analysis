import zipfile

# Open the zip file
with zipfile.ZipFile("archive.zip", "r") as zip_ref:
    # Extract all files to the current directory
    zip_ref.extractall()

    # Extract a specific file
    # zip_ref.extract("file_to_extract.txt", "destination_folder/") 