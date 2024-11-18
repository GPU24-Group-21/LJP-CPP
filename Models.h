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
  int stepAvg = 0;
  int stepLimit = 0;
  float temperature = 0;
};

struct Molecule {
  // Molecule ID
  int id = 0;
  // Position
  float pos[2] = {0, 0};
  // Velocity
  float vel[2] = {0, 0};
  // Acceleration
  float acc[2] = {0, 0};

  void multiple_pos(const float a) {
    pos[0] *= a;
    pos[1] *= a;
  }

  void multiple_vel(const float a) {
    vel[0] *= a;
    vel[1] *= a;
  }

  void multiple_acc(const float a) {
    acc[0] *= a;
    acc[1] *= a;
  }

  void add_pos(const float a) {
    pos[0] += a;
    pos[1] += a;
  }

  void add_vel(const float a) {
    vel[0] += a;
    vel[1] += a;
  }

  void add_acc(const float a) {
    acc[0] += a;
    acc[1] += a;
  }
};

struct BlockResult {
  float vSum[2];
  float vvSum;
};

struct PropertiesData {
  float vSum[2];
  float vvSum;
  float ke;
  float keSum;
  float keSum2;
  float totalEnergy;
  float totalEnergy2;
  float pressure;
  float pressure2;
  int cycleCount;
};

template <typename T> struct Prop {
  T val = 0;
  T sum1 = 0;
  T sum2 = 0;
};

// Global variables
int IADD = 453806245;
int IMUL = 314159269;
int MASK = 2147483647;
int NDIM = 2;
float SCALE = 0.4656612873e-9;
// LJP parameters:
float EPSILON = 1.0;
float SIGMA = 1.0;
#endif // MODELS_H
