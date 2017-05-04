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

```make```

2. map folders with images and and output directory, and run:

```docker run -v **/folder/with/chips/**:/pelops_root/INPUT_DIR -v **/dir/for/output/**:/pelops_root/OUTPUT_DIR l41-pelops-i2v```

Note: Docker creates output files owned by root. Grant write privileges to OUTPUT_DIR for the current user and add ```-u $(id -u $USER)``` to the docker run command above to create output files owned by the current user.

3. Advanced, bring your own model:

```docker run -v **/folder/with/chips/**:/pelops_root/INPUT_DIR -v **/dir/for/output/**:/pelops_root/OUTPUT_DIR -v **/folder/with/model_files/**:/pelops_root/MODEL_DIR -e MODEL='/pelops_root/**MODELFILENAME**' -e WEIGHTS='/pelops_root/**WEIGHTSFILENAME**' -e LAYER='**layernameToUseAsOutput**' l41-pelops-i2v```

Run the Siamese model as follows:

```docker run -v /folder/with/chips1:/pelops_root/INPUT_DIR1 -v /folder/with/chips2:/pelops_root/INPUT_DIR2 -v /dir/for/output:/pelops_root/OUTPUT_DIR -v /folder/with/saved/model:/pelops_root/MODEL_DIR -e WEIGHTS='/pelops_root/MODEL_DIR/model_name.weights.hdf5' -e MODEL='/pelops_root/MODEL_DIR/model_name.model.json' -e VECTORS='/pelops_root/INPUT_DIR1/vectors.json' l41-pelops-siamese```

Run the Ranker to compare two directories as follows:

```docker run -v /folder/with/chips1:/pelops_root/INPUT_DIR1 -v /folder/with/chips2:/pelops_root/INPUT_DIR2 -v /dir/for/output:/pelops_root/OUTPUT_DIR -v /folder/with/saved/model:/pelops_root/MODEL_DIR -e WEIGHTS='/pelops_root/MODEL_DIR/model_name.weights.hdf5' -e MODEL='/pelops_root/MODEL_DIR/model_name.model.json' -e VECTORS='/pelops_root/INPUT_DIR1/vectors.json' l41-pelops-ranker```

Note: Docker creates output files owned by root. Grant write privileges to OUTPUT_DIR for the current user and add ```-u $(id -u $USER)``` to the docker run commands above to create output files owned by the current user.

# Tests
Tests are currently written in py.test for Python. The tests are automatically run when building the containers.

They can also be tested using:
TODO

# Contributing to Pelops

Want to contribute?  Awesome!

Please make sure you have [`pre-commit`](http://pre-commit.com/) installed so
that your code is checked for various issues.

After that, send us a pull request! We're happy to review them!
