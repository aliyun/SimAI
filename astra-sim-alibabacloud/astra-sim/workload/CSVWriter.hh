/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __CSVWRITER_HH__
#define __CSVWRITER_HH__

#include <fcntl.h>
#include <math.h>
#include <sys/stat.h>
#include <unistd.h>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <fstream>
#include <iostream>
#include <list>
#include <map>
#include <tuple>
#include <vector>

namespace AstraSim {
class CSVWriter {
 public:
  // std::fstream inFile;
  std::fstream myFile;
  void initialize_csv(int rows, int cols);
  void finalize_csv(std::list<std::list<std::pair<uint64_t, double>>> dims);
  CSVWriter(std::string path, std::string name);
  void write_cell(int row, int column, std::string data);
  void write_line(std::string data);
  std::string path;
  std::string name;
  ~CSVWriter() {
    if (myFile.is_open()) {
      myFile.close();
    }
  }
  inline bool exists_test(const std::string& name) {
    struct stat buffer;
    return (stat(name.c_str(), &buffer) == 0);
  }
};
} // namespace AstraSim
#endif
