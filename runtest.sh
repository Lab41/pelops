#!/bin/sh
docker build -f ./docker/Dockerfile.test -t pelops_test .
docker run pelops_test
