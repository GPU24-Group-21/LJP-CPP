# Molecular Dynamics Simulation (C++/CUDA)
> Group 21 <br>
> Kamku Xue(yx3494@nyu.edu) <br>
> Ning Miao(nm4543@nyu.edu)

This is a simple implement of molecular dynamics simulation written in C++ and CUDA. The simulation is based on the Lennard-Jones potential. And we try to optimize the simulation using CUDA to speed up the simulation.

## Lennard-Jones Potential

The Lennard-Jones potential is a simple mathematical model that describes how pairs of neutral atoms or molecules interact with each other. The potential is given by the following equation:

$$
V(r) = 4 \epsilon \left[ \left( \frac{\sigma}{r} \right)^{12} - \left( \frac{\sigma}{r} \right)^6 \right]
$$

where $r$ is the distance between the two atoms, $\epsilon$ is the depth of the potential well, and $\sigma$ is the finite distance at which the inter-particle potential is zero.

In our version, we simplify the potential to:

$$ \epsilon = 1, \sigma = 1 $$
$$
V(r) = \left( \frac{1}{r^{12}} - \frac{1}{r^6} \right)
$$

Aslo, the temperature is set to constant 1 to simplify the simulation.

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
./ljp config.in [size] [0: cpu, 1: cuda]

# Clean up
make clean

# or you can run automatically
make all
```

## Results

All the results are stored in the `output` directory. The results include the positions and velocities of the particles at each time step, as well as the total energy of the system.