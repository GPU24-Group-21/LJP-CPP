#include <iostream>
#include <string>
#include <fstream>
#include <chrono>
#include <cmath>
#include <iomanip>
#include "Models.h"

using namespace std;

// Constants
Config config;
bool debug = true;

// Statistical variables
// velocity sum
double vSum[2] = {0, 0};
// kinetic energy (Ek)
double keSum = 0;
double keSum2 = 0;
// total energy (E)
double totalEnergy = 0;
double totalEnergy2 = 0;
// pressure(P)
double pressure = 0;
double pressure2 = 0;

double rCut = 0;
double region[2] = {0, 0};
double velMag = 0;


//timer
auto start = chrono::high_resolution_clock::now();
auto start_time = start;
auto end_time = chrono::high_resolution_clock::now();

void breakPoint(const string &msg, const bool end = false) {
    end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end_time - (end ? start : start_time));
    cout << "[" << msg << "] " << duration.count() << " ms - " << fixed << setprecision(4) << duration.count() /
            1000000.0 <<
            " s" << endl;
    start_time = chrono::high_resolution_clock::now();
}


// Read input file
void readToken(std::ifstream &file, const string &token) {
    string str;
    file >> str;
    if (str != token) {
        std::cerr << "Error: token not found: " << token << std::endl;
        exit(1);
    }
}

void readConfig(const string &filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: file not found" << std::endl;
        exit(1);
    }

    // Read config
    readToken(file, "deltaT");
    file >> config.deltaT;
    readToken(file, "density");
    file >> config.density;
    readToken(file, "initUcell_x");
    file >> config.initUcell_x;
    readToken(file, "initUcell_y");
    file >> config.initUcell_y;
    readToken(file, "stepAvg");
    file >> config.stepAvg;
    readToken(file, "stepEquil");
    file >> config.stepEquil;
    readToken(file, "stepLimit");
    file >> config.stepLimit;
    readToken(file, "temperature");
    file >> config.temperature;

    if (debug) {
        cout << "=========== Config ===========" << endl;
        cout << "  deltaT: " << config.deltaT << endl;
        cout << "  density: " << config.density << endl;
        cout << "  initUcell_x: " << config.initUcell_x << endl;
        cout << "  initUcell_y: " << config.initUcell_y << endl;
        cout << "  stepAvg: " << config.stepAvg << endl;
        cout << "  stepEquil: " << config.stepEquil << endl;
        cout << "  stepLimit: " << config.stepLimit << endl;
        cout << "  temperature: " << config.temperature << endl;
    }
}

void readMoo(const string &filename, long N, Molecule *molecules) {
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: file not found" << std::endl;
        exit(1);
    }

    readToken(file, "rCut");
    file >> rCut;
    readToken(file, "region");
    file >> region[0] >> region[1];
    readToken(file, "velMag");
    file >> velMag;

    if (debug) {
        cout << "=========== Pre Defined Props ===========" << endl;
        cout << "  rCut: " << rCut << endl;
        cout << "  region: " << region[0] << " " << region[1] << endl;
        cout << "  velMag: " << velMag << endl;
    }

    // rest til the end of the file is the molecule
    for (int i = 0; i < N; i++) {
        Molecule m;
        m.id = i;
        file >> m.pos[0] >> m.pos[1] >> m.vel[0] >> m.vel[1] >> m.acc[0] >> m.acc[1];
        molecules[i] = m;
    }

    if (debug) {
        cout << "=========== Molecules ===========" << endl;
        for (int i = 0; i < N; i++) {
            cout << "  id: " << molecules[i].id << " pos: " << molecules[i].pos[0] << " " << molecules[i].pos[1] <<
                    " vel: "
                    << molecules[i].vel[0] << " " << molecules[i].vel[1] << endl;
        }
    }
}

void randVec(double vec[2]) {
    double r = sqrt(-2.0 * log(rand() / (RAND_MAX + 1.0)));
    double theta = 2.0 * M_PI * (rand() / (RAND_MAX + 1.0));
    vec[0] = r * cos(theta);
    vec[1] = r * sin(theta);
}

void outputResult(const string &filename, const long n, const Molecule *molecules, const int step, const double dTime) {
    ofstream file;
    file.open(filename);

    if (!file.is_open()) {
        std::cerr << "Error: file not found" << std::endl;
        exit(1);
    }

    file << "step: " << to_string(step) << endl;
    file << "ts: " << dTime << endl;
    file << "vSum0: " << vSum[0] / n << endl;
    file << "vSum1: " << vSum[1] / n << endl;
    file << "====================" << endl;
    file << setprecision(15) << fixed;
    const int mark1 = n / 2 + n / 8;
    const int mark2 = n / 2 + n / 8 + 1;
    for (int i = 0; i < n; i++) {
        if (i == mark1 || i == mark2) {
            file << "m-" << molecules[i].id << " ";
        } else {
            file << "o-" << molecules[i].id << " ";
        }
        file << molecules[i].pos[0] << " " << molecules[i].pos[1] << " " << endl;
    }
}

// ============= Core function =============
//Toroidal functions
void toroidal(double &x, double &y) {
    if (x < -0.5 * region[0]) x += region[0];
    if (x >= 0.5 * region[0]) x -= region[0];
    if (y < -0.5 * region[1]) y += region[1];
    if (y >= 0.5 * region[1]) y -= region[1];
}

// Calculate the force between two molecules
void leapfrog(const long n, Molecule *mols, const bool pre) {
    for (int i = 0; i < n; i++) {
        mols[i].vel[0] += 0.5 * config.deltaT * mols[i].acc[0];
        mols[i].vel[1] += 0.5 * config.deltaT * mols[i].acc[1];
        if (pre) {
            mols[i].pos[0] += config.deltaT * mols[i].vel[0];
            mols[i].pos[1] += config.deltaT * mols[i].vel[1];
        }
    }
}

void boundaryCondition(const long n, Molecule *mols) {
    for (int i = 0; i < n; i++) {
        toroidal(mols[i].pos[0], mols[i].pos[1]);
    }
}

void evaluateForce(const long n, Molecule *mols, double &uSum, double &virSum) {
    // reset the acceleration
    for (int i = 0; i < n; i++) {
        mols[i].acc[0] = 0;
        mols[i].acc[1] = 0;
    }

    for (int i = 0; i < n - 1; i++) {
        for (int j = i + 1; j < n; j++) {
            // Make DeltaRij: (sum of squared RJ1-RJ2)
            double dr[2] = {
                mols[i].pos[0] - mols[j].pos[0],
                mols[i].pos[1] - mols[j].pos[1]
            };
            toroidal(dr[0], dr[1]);
            const double rr = dr[0] * dr[0] + dr[1] * dr[1];
            const double r = sqrt(rr);

            // case dr2 < Rc^2
            if (rr < rCut * rCut) {
                const double r2i = SIGMA / rr;
                double rri3 = r2i * r2i * r2i;

                // Forces calculation by Lennard-Jones potential (original from Rapaport)
                // double fcVal = 48.0 * rri3 * (rri3 - 0.5) * r2i;
                // Forces calculated with the completed Lennard-Jones.
                const double fcVal = 48 * EPSILON * pow(SIGMA, 12) / pow(r, 13) - 24 * EPSILON * pow(SIGMA, 6) /
                                     pow(r, 7);

                // update the acc
                mols[i].acc[0] += fcVal * dr[0];
                mols[i].acc[1] += fcVal * dr[1];

                mols[j].acc[0] -= fcVal * dr[0];
                mols[j].acc[1] -= fcVal * dr[1];


                // # Lennard-Jones potential (original from Rapaport)
                // uSum += 4.0 * rri3 * (rri3 - 1.0) + 1.0;
                // The completed Lennard-Jones.

                // Lennard-Jones potential balanced
                uSum += 4.0 * EPSILON * pow(SIGMA / r, 12) / r - pow(SIGMA / r, 6);
                virSum += fcVal * rr;
            }
        }
    }
}

void evaluateProperties(const long n, const Molecule *mols, const double &uSum, const double &virSum) {
    vSum[0] = 0;
    vSum[1] = 0;

    double vvSum = 0;

    for (int i = 0; i < n; i++) {
        vSum[0] += mols[i].vel[0];
        vSum[1] += mols[i].vel[1];
        vvSum += mols[i].vel[0] * mols[i].vel[0] + mols[i].vel[1] * mols[i].vel[1];
    }

    const double ke = 0.5 * vvSum / n;
    const double energy = ke + uSum / n;
    const double p = config.density * (vvSum + virSum) / (n * 2);

    keSum += ke;
    totalEnergy += energy;
    pressure += p;

    keSum2 += ke * ke;
    totalEnergy2 += energy * energy;
    pressure2 += p * p;
}

void stepSummary(const long n, const int step, const double dTime) {
    // average and standard deviation of kinetic energy, total energy, and pressure
    double keAvg = keSum / config.stepAvg;
    double totalAvg = totalEnergy / config.stepAvg;
    double pressureAvg = pressure / config.stepAvg;

    double keStd = sqrt(max(0.0, keSum2 / config.stepAvg - keAvg * keAvg));
    double totalStd = sqrt(max(0.0, totalEnergy2 / config.stepAvg - totalAvg * totalAvg));
    double pressureStd = sqrt(max(0.0, pressure2 / config.stepAvg - pressureAvg * pressureAvg));

    cout << fixed << setprecision(4)
            << step << "\t"
            << dTime << "\t"
            << vSum[0] / n << "\t"
            << totalAvg << "\t"
            << totalStd << "\t"
            << keAvg << "\t"
            << keStd << "\t"
            << pressureAvg << "\t"
            << pressureStd << endl;

    // reset the sum
    keSum = 0;
    keSum2 = 0;
    totalEnergy = 0;
    totalEnergy2 = 0;
    pressure = 0;
    pressure2 = 0;
}

// Main function
int main(int argc, char *argv[]) {
    start = chrono::high_resolution_clock::now();
    start_time = start;

    // Parse arguments
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <config file> [init mol file]" << std::endl;
        return 1;
    }

    const string filename = argv[1];
    readConfig(filename);

    // create output folder '/output' if not exist
    system("mkdir -p output");

    const long mSize = config.initUcell_x * config.initUcell_y;
    Molecule molecules[mSize];
    if (argc == 3) {
        const string molFile = argv[2];
        readMoo(molFile, mSize, molecules);
    } else {
        rCut = pow(2.0, 1.0 / 6.0 * SIGMA);
        // Region size
        region[0] = 1.0 / sqrt(config.density) * config.initUcell_x;
        region[1] = 1.0 / sqrt(config.density) * config.initUcell_y;
        // Velocity magnitude
        velMag = sqrt(3.0 * config.temperature);
        const double gap[2] = {
            region[0] / config.initUcell_x,
            region[1] / config.initUcell_y
        };
        for (int i = 0; i < config.initUcell_x; i++) {
            for (int j = 0; j < config.initUcell_y; j++) {
                Molecule m;
                // assign molecule id
                m.id = i * config.initUcell_y + j;

                // assign position
                m.pos[0] = (i + 0.5) * gap[0] + region[0] * -0.5;
                m.pos[1] = (j + 0.5) * gap[1] + region[1] * -0.5;

                // assign velocity
                m.vel[0] = velMag * (rand() / (RAND_MAX + 1.0) - 0.5);
                m.vel[1] = velMag * (rand() / (RAND_MAX + 1.0) - 0.5);

                // add to list
                molecules[i * config.initUcell_y + j] = m;
            }
        }
    }

    int step = 0;
    while (step < config.stepLimit) {
        step++;
        double uSum = 0;
        double virSum = 0;
        double deltaT = (double) step * config.deltaT;
        leapfrog(mSize, molecules, true);
        boundaryCondition(mSize, molecules);
        evaluateForce(mSize, molecules, uSum, virSum);
        leapfrog(mSize, molecules, false);
        evaluateProperties(mSize, molecules, uSum, virSum);
        if (config.stepAvg > 0 && step % config.stepAvg == 0) {
            stepSummary(mSize, step, deltaT);
        }
        // output the result
        outputResult("output/" + to_string(step - 1) + ".out", mSize, molecules, step - 1, deltaT);
    }
    breakPoint("Total Time", true);
    return 0;
}
