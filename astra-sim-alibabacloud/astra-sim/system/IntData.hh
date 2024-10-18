/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __INDATA_HH__
#define __INDATA_HH__
namespace AstraSim {
class IntData : public CallData {
 public:
  int data;
  IntData(int d) {
    data = d;
  }
};
} // namespace AstraSim
#endif
