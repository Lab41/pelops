all: base image2vecs siamese

# Base requirements for all containers
base:
	docker build -t l41-pelops-base -f docker/Dockerfile.base .

# Tests
test: base
	docker build -t l41-pelops-tests -f docker/Dockerfile.test .
	docker run l41-pelops-tests

# Image processing
image2vecs: base
	docker build -t l41-pelops-i2v -f docker/Dockerfile.images2vecs .

siamese: base
	docker build -t l41-pelops-siamese -f docker/Dockerfile.vectorSiamese .
