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

- [Turning Chips into features](docs/chips_to_features.md)

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
