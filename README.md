# Molecular Dynamics Simulation (C++/CUDA)

This is a simple molecular dynamics simulation written in C++ and CUDA. The simulation is based on the Lennard-Jones potential. And we try to optimize the simulation using CUDA to speed up the simulation.

## Lennard-Jones Potential

The Lennard-Jones potential is a simple mathematical model that describes how pairs of neutral atoms or molecules interact with each other. The potential is given by the following equation:

$$
V(r) = 4 \epsilon \left[ \left( \frac{\sigma}{r} \right)^{12} - \left( \frac{\sigma}{r} \right)^6 \right]
$$

where $r$ is the distance between the two atoms, $\epsilon$ is the depth of the potential well, and $\sigma$ is the finite distance at which the inter-particle potential is zero.

## Simulation

The simulation is based on the following algorithm:

1. Initialize the positions and velocities of the particles.

2. Calculate the forces between the particles using the Lennard-Jones potential.

3. Update the positions and velocities of the particles using the Verlet integration algorithm.


## Usage

```bash
# Compile the code
make

# Run the simulation
make run
# run the cuda only
make run-cuda
# run the cpu only
make run-cpu

# directly run the simulation
./ljp config.in [size]

# Clean up
make clean

# or you can run automatically
make all
```