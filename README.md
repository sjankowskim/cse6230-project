# CSE 6230 Project #
CSE 6230 Spring 2024 project by Sebastian Jankowski, Gabriel mister-has-too-many-last-names, and Ryan Ding

## Running General Purpose ##
`git clone https://github.com/sjankowskim/cse6230-project.git`

Change your directory to `cse6230-project`

Edit `general_purpose.cpp` or make a copy of it and edit it as necessary.

Run `make` or edit `Makefile` to change `SRCS = general_purpose.cpp` to whatever you named your file.

Use the scripts in `slurm/` to run your code on ICE-PACE.

`sbatch slurm/job-gen`

## Running CUDA ##
`git clone https://github.com/sjankowskim/cse6230-project.git`

Change your directory to `cse6230-project`

`module purge`

`module load cmake`

`module load cuda/11.7.0-7sdye3`

`module load gcc`

`mkdir build`

`cd build`

Edit `cuda.cu` as necessary or make a copy of it and go into `CMakeLists.txt` and change `cuda.cu` in `add_executable(gpu cuda.cu utils.hpp)` to be file name you made.

`cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=70 ..`

`make`

You can add parameters to the slurm call (e.g. `-t 0`)

`sbatch job-gpu -t 0`


