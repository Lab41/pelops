all: base

# Base requirements for all containers
base:
	docker build -t l41-pelops-base -f docker/Dockerfile.base .

# Tests
test: base
	docker build -t l41-pelops-tests -f docker/Dockerfile.test .
	docker run l41-pelops-tests
