import kagglehub

path = kagglehub.dataset_download(
    "sautkin/imagenet1k1",
)

print(path)