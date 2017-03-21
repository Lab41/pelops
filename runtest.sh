#!/bin/sh
docker build -f ./Dockerfile.test -t pelops_test .
docker run pelops_test
