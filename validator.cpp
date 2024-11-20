#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
using namespace std;

template <typename T>
void readToken(std::ifstream &file, const string &token, T &val) {
  string str;
  file >> str;
  if (str != token) {
    std::cerr << "Error: token not found. Expected: " << token
              << " Got: " << str << std::endl;
    exit(1);
  }
  file >> val;
}

bool checkMol(ifstream &file1, ifstream &file2) {
  string token1, token2;
  double x1, y1, x2, y2;
  if (!(file1 >> token1) || !(file2 >> token2)) {
    return false;
  }
  if (token1.find("=") != string::npos) {
    return true;
  }
  if (token1 != token2) {
    std::cerr << "Error: Mol_id mismatch: " << token1 << " " << token2
              << std::endl;
    exit(1);
  }
  if (!(file1 >> x1 >> y1) || !(file2 >> x2 >> y2)) {
    cerr << "Error: EOF while reading molecule coordinates" << endl;
    exit(1);
  }
  if (abs(x1 - x2) > 0.009 || abs(y1 - y2) > 0.009) {
    cout << "Check Mol[" << token1 << "]Failed: (" << x1 << "," << y1
         << ") != (" << x2 << "," << y2 << ")" << std::endl;
    exit(1);
  }
  return true;
}

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <mol_out_file1> <mol_out_file2>"
              << std::endl;
    return 1;
  }

  string filename1 = argv[1];
  string filename2 = argv[2];
  ifstream file1(filename1), file2(filename2);

  if (!file1.is_open() || !file2.is_open()) {
    std::cerr << "Error: could not open file" << std::endl;
    return 1;
  }

  int step1, step2;
  readToken(file1, "step", step1);
  readToken(file2, "step", step2);

  if (step1 != step2) {
    std::cerr << "Error: step mismatch: step1=" << step1 << " step2=" << step2
              << std::endl;
    return 1;
  }

  double ts1, ts2;
  readToken(file1, "ts", ts1);
  readToken(file2, "ts", ts2);
  double E_1, E_2;
  readToken(file1, "E", E_1);
  readToken(file2, "E", E_2);
  double KE_1, KE_2;
  readToken(file1, "KE", KE_1);
  readToken(file2, "KE", KE_2);
  double P_1, P_2;
  readToken(file1, "P", P_1);
  readToken(file2, "P", P_2);
  double sE_1, sE_2;
  readToken(file1, "sE", sE_1);
  readToken(file2, "sE", sE_2);
  double sKE_1, sKE_2;
  readToken(file1, "sKE", sKE_1);
  readToken(file2, "sKE", sKE_2);
  double sP_1, sP_2;
  readToken(file1, "sP", sP_1);
  readToken(file2, "sP", sP_2);

  while (!file1.eof() && !file2.eof()) {
    if (!checkMol(file1, file2)) {
      break;
    }
  }

  return 0;
}