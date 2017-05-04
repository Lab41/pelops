all: base image2vecs siamese ranker build-notebook

# Base requirements for all containers
base:
	docker build -t l41-pelops-base -f docker/Dockerfile.base .

# Jupyter notebook server
build-notebook: base
	docker build -t l41-pelops-notebook -f docker/Dockerfile.notebook ./docker/

notebook: build-notebook
	docker run -p 8888:8888 -it l41-pelops-notebook

# Tests
test: base
	docker build -t l41-pelops-tests -f docker/Dockerfile.test .
	docker run l41-pelops-tests

# Image processing
image2vecs: base
	docker build -t l41-pelops-i2v -f docker/Dockerfile.images2vecs .

siamese: base
	docker build -t l41-pelops-siamese -f docker/Dockerfile.vectorSiamese .

ranker: base
	docker build -t l41-pelops-ranker -f docker/Dockerfile.rankDirectories .
