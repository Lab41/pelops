# Pelops

[![CircleCI](https://circleci.com/gh/Lab41/pelops.svg?style=svg)](https://circleci.com/gh/Lab41/pelops)[![codecov](https://codecov.io/gh/Lab41/pelops/branch/master/graph/badge.svg)](https://codecov.io/gh/Lab41/pelops)

<!-- Need to set width, which can't be done with MarkDown on Github -->
<img src="/misc/pelops.png" alt="Pelops Logo" width="200"/>

Pelops is a project by [Lab41](http://www.lab41.org/) that uses deep learning
based methods to automatically identify cars by using their large scale
features—color, shape, light configuration, etc.

# Install Instructions
TODO

# Configuration
TODO

# Required Dependencies
TODO

# Documentation
TODO

# Turning Chips to Features
1. build the docker continer

docker build -f docker/Dockerfile.images2vecs -t i2v .

2. map folders with images and and output directory, and run.

docker run -v **/folder/with/chips/**:/pelops_root/INPUT_DIR -v **/dir/for/output/**:/pelops_root/OUTPUT_DIR i2v 

3. Advanced, bring your own model.

docker run -v **/folder/with/chips/**:/pelops_root/INPUT_DIR -v **/dir/for/output/**:/pelops_root/OUTPUT_DIR -v **/folder/with/model_files/**:/pelops_root/MODEL_DIR -e MODEL='/pelops_root/**MODELFILENAME**' -e WEIGHTS='/pelops_root/**WEIGHTSFILENAME**' -e LAYER='**layernameToUseAsOutput**' i2v 

# Tests
Tests are currently written in py.test for Python. The tests are automatically run when building the containers.

They can also be tested using:
TODO

# Contributing to Pelops

Want to contribute?  Awesome!

Please make sure you have [`pre-commit`](http://pre-commit.com/) installed so
that your code is checked for various issues.

After that, send us a pull request! We're happy to review them!
