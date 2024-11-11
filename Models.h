//
// Created by KAMIKU on 11/9/24.
//

#ifndef MODELS_H
#define MODELS_H
#include <cstdint>
#include <cstdlib>

struct Config {
    float deltaT = 0;
    float density = 0;
    int initUcell_x = 0;
    int initUcell_y = 0;
    int stepAvg = 0;
    int stepEquil = 0;
    int stepLimit = 0;
    float temperature = 0;
};

static constexpr long IADD = 453806245;
static constexpr long IMUL = 314159269;
static constexpr long MASK = 2147483647;
static constexpr int NDIM = 2;
static constexpr double SCALE = 0.4656612873e-9;

// LJP parameters:
static double EPSILON = 1.0;
static double SIGMA = 1.0;

struct Molecule {
    // Molecule ID
    int id = 0;
    // Position
    double pos[2] = {0, 0};
    // Velocity
    double vel[2] = {0, 0};
    // Acceleration
    double acc[2] = {0, 0};

    void multiple_pos(const double a) {
        pos[0] *= a;
        pos[1] *= a;
    }

    void multiple_vel(const double a) {
        vel[0] *= a;
        vel[1] *= a;
    }

    void multiple_acc(const double a) {
        acc[0] *= a;
        acc[1] *= a;
    }

    void add_pos(const double a) {
        pos[0] += a;
        pos[1] += a;
    }

    void add_vel(const double a) {
        vel[0] += a;
        vel[1] += a;
    }

    void add_acc(const double a) {
        acc[0] += a;
        acc[1] += a;
    }
};

template<typename T>
struct Prop {
    T val = 0;
    T sum1 = 0;
    T sum2 = 0;
};

#endif //MODELS_H
