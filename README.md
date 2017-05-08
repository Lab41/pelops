# Pelops

[![CircleCI](https://circleci.com/gh/Lab41/pelops.svg?style=svg)](https://circleci.com/gh/Lab41/pelops)[![codecov](https://codecov.io/gh/Lab41/pelops/branch/master/graph/badge.svg)](https://codecov.io/gh/Lab41/pelops)

<!-- Need to set width, which can't be done with MarkDown on Github -->
<img src="/misc/pelops.png" alt="Pelops Logo" width="200"/>

Pelops is a project by [Lab41](http://www.lab41.org/) that uses deep learning
based methods to automatically identify cars by using their large scale
features—color, shape, light configuration, etc.

# Install Instructions

Pelops provides several Docker containers the assist in running the project.
You can build them by checking out the code and running make:

```bash
git clone https://github.com/Lab41/pelops.git
cd pelops
make
```

Then:

```bash
make notebook
```

Which will run a container containing Pelops and a notebook server.

Otherwise you can install Pelops using `pip`:

```bash
git clone https://github.com/Lab41/pelops.git
pip install pelops
```

There are several dependencies that will need to be installed. The
[`requirements.txt`](requirements.txt) should include most of them, but other
programs such as [keras](https://keras.io/) and
[Tensorflow](https://www.tensorflow.org/) are also required. For this reason
it is suggested to use the notebook container to run Pelops.

# Documentation
TODO

# Turning Chips to Features

1. build the docker containers using make:

```bash
make
```

2. map folders with images and and output directory, and run:

```bash
CHIPDIR1=/folder/with/chips && \
OUTPUTDIR=/folder/for/output && \
docker run -v ${CHIPDIR1}:/pelops_root/INPUT_DIR -v ${OUTPUTDIR}:/pelops_root/OUTPUT_DIR l41-pelops-i2v
```

Note: Docker creates output files owned by root. Grant write privileges to OUTPUT_DIR for the current user and add `-u $(id -u $USER)` to the docker run command above to create output files owned by the current user.

3. Advanced, bring your own model:

```bash
CHIPDIR1=/folder/with/chips && \
OUTPUTDIR=/folder/for/output && \
MODELDIR=/folder/with/models && \
MODELFILE=name_of_model_file && \
WEIGHTFILE=name_of_weight_file && \
LAYERNAME=layername && \
docker run -v ${CHIPDIR1}:/pelops_root/INPUT_DIR -v ${OUTPUTDIR}:/pelops_root/OUTPUT_DIR -v ${MODELDIR}:/pelops_root/MODEL_DIR -e MODEL="/pelops_root/${MODELFILE}" -e WEIGHTS="/pelops_root/${WEIGHTFILE}" -e LAYER="${LAYERNAME}" l41-pelops-i2v
```

Run the Siamese model as follows:

```bash
CHIPDIR1=/folder/with/chips && \
CHIPDIR2=/folder/with/other/chips && \
OUTPUTDIR=/folder/for/output && \
MODELDIR=/folder/with/models && \
MODELFILE=name_of_model_file.json && \
WEIGHTFILE=name_of_weight_file.hdf5 && \
VECTORFILE=name_of_VECTOR_file.json && \
docker run -v ${CHIPDIR1}:/pelops_root/INPUT_DIR1 -v ${CHIPDIR2}:/pelops_root/INPUT_DIR2 -v ${OUTPUTDIR}:/pelops_root/OUTPUT_DIR -v ${MODELDIR}:/pelops_root/MODEL_DIR -e WEIGHTS="/pelops_root/MODEL_DIR/${WEIGHTFILE}" -e MODEL="/pelops_root/MODEL_DIR/${MODELFILE}" -e VECTORS="/pelops_root/INPUT_DIR1/${VECTORFILE}" l41-pelops-siamese
```

Run the Ranker to compare two directories as follows:

```bash
CHIPDIR1=/folder/with/chips && \
CHIPDIR2=/folder/with/other/chips && \
OUTPUTDIR=/folder/for/output && \
MODELDIR=/folder/with/models && \
MODELFILE=name_of_model_file.json && \
WEIGHTFILE=name_of_weight_file.hdf5 && \
VECTORFILE=name_of_VECTOR_file.json && \
docker run -v ${CHIPDIR1}:/pelops_root/INPUT_DIR1 -v ${CHIPDIR2}:/pelops_root/INPUT_DIR2 -v ${OUTPUTDIR}:/pelops_root/OUTPUT_DIR -v ${MODELDIR}:/pelops_root/MODEL_DIR -e WEIGHTS="/pelops_root/MODEL_DIR/${WEIGHTFILE}" -e MODEL="/pelops_root/MODEL_DIR/${MODELFILE}" -e VECTORS="/pelops_root/INPUT_DIR1/${VECTORFILE}" l41-pelops-ranker
```

Note: Docker creates output files owned by root. Grant write privileges to OUTPUT_DIR for the current user and add `-u $(id -u $USER)` to the docker run commands above to create output files owned by the current user.

# Tests

Tests are currently written in [pytest](https://docs.pytest.org/en/latest/). The tests are automatically run when submitting pull requests.

You can run the tests in a container by calling:

```bash
make test
```

This will build a docker container, mount your local version of the code, and
run the tests.

# Contributing to Pelops

Want to contribute?  Awesome!

Please make sure you have [`pre-commit`](http://pre-commit.com/) installed so
that your code is checked for various issues.

After that, send us a pull request! We're happy to review them!
