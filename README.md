# GPU-VAH
CUDA version of CPU-VAH (anisotropic hydrodynamics code for Heavy Ion Collisions)

To compile and run you will need to install libconfig and gtest files.
To compile simply type make. To run create an output directory and type
./gpu-vah --config rhic-conf -o output_directory_you_created -h
The directory rhic-conf is where all of the input files are located.
