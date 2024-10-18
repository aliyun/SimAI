/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __STATDATA_HH__
#define __STATDATA_HH__

#include <assert.h>
#include <math.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <fstream>
#include <list>
#include <map>
#include <sstream>
#include <tuple>
#include <vector>
#include "CallData.hh"
#include "Common.hh"
namespace AstraSim {
class StatData : public CallData {
 public:
  Tick start;
  Tick waiting;
  Tick end;
  //~StatData()= default;
  StatData() {
    start = 0;
    waiting = 0;
    end = 0;
  }
};
} // namespace AstraSim
#endif
