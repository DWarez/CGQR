# CGQR

_Conjugate Gradient, QR factorization_

This repository contains the source code and material of our project for the Computational Mathematics course
at University of Pisa, a.y. 2021/22.

Authors: [Dario Salvati](https://github.com/DWarez), [Andrea Zuppolini](https://github.com/AndreZupp)


Please do not contribute.


## How to build
If you wish to build from source you need [CMake](https://cmake.org/).
You'll also need to install [Armadillo](http://arma.sourceforge.net), the only dependency
of this project.

We suggest to use WLS in case you're running on Windows. From the WLS (Ubuntu) shell:
```bash
# install required packages
$ sudo apt install build-essential cmake liblapack-dev libblas-dev libboost-dev

# install Armadillo
sudo apt-get install libarmadillo-dev
```

Then, clone and compile from source:

```bash
# clone the repository
$ git clone https://github.com/DWarez/CGQR

# navigate into it
$ cd CGQR

# make
$ cmake -B build
$ cd build
$ make
```

To execute the entire experiment, run:

```bash
# from the CGQR directory navigate into build directory
$ cd build

# run the experiment
$ ./CGQR

# if you want to run a test, from the build directory run
$ ./cg_test # Conjugate Gradient test
# or
$ ./qr_test # QR factorization test
```
