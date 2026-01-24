import kagglehub

# Download latest version
path = kagglehub.dataset_download("sanikatiwarekar/deep-fake-detection-dfd-entire-original-dataset")

print("Path to dataset files:", path)