/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __ASTRAMEMORYAPI_HH__
#define __ASTRAMEMORYAPI_HH__
#include <cassert>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
namespace AstraSim {
class AstraMemoryAPI {
 public:
  virtual uint64_t mem_read(uint64_t size) = 0;
  virtual uint64_t mem_write(uint64_t size) = 0;
  virtual uint64_t npu_mem_read(uint64_t size) = 0;
  virtual uint64_t npu_mem_write(uint64_t size) = 0;
  virtual uint64_t nic_mem_read(uint64_t size) = 0;
  virtual uint64_t nic_mem_write(uint64_t size) = 0;
  virtual ~AstraMemoryAPI() = default;
};
} // namespace AstraSim
#endif
