# Shobu - Python

This repository contains PyTorch implementations of neural network
architectures for playing Shobu, a 2-player abstract strategy board game. The
two projects in this repo are AlphaZero and NNUE.

The .pyd file in this repository is the move generator used by both projects,
and was compiled to run on Python 3.12. It can be found in the
[shobu-engine](https://github.com/TangilJ/shobu-engine) repository. The
inference code for running the NNUE network can also be found in that
repository.
