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
  cout << "Check Mol[" << token1 << "]...";
  if (!(file1 >> x1 >> y1) || !(file2 >> x2 >> y2)) {
    cerr << "Error: EOF while reading molecule coordinates" << endl;
    exit(1);
  }
  if (abs(x1 - x2) > 0.009 || abs(y1 - y2) > 0.009) {
    cout << "Failed: (" << x1 << "," << y1 << ") != (" << x2 << "," << y2 << ")"
         << std::endl;
    exit(1);
  }
  cout << "Passed" << endl;
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

  // read first line: "step 0"
  int step1, step2;
  readToken(file1, "step", step1);
  readToken(file2, "step", step2);

  if (step1 != step2) {
    std::cerr << "Error: step mismatch: step1=" << step1 << " step2=" << step2
              << std::endl;
    return 1;
  }

  cout << "Validate Step " << step1 << endl;

  // read second line: "ts 0.005"
  double ts1, ts2;
  readToken(file1, "ts", ts1);
  readToken(file2, "ts", ts2);
  while (!file1.eof() && !file2.eof()) {
    if (!checkMol(file1, file2)) {
      break;
    }
  }
  return 0;
}