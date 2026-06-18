// -*- c++ -*-

#include "ns3/core-module.h"

using namespace ns3;

namespace 
{

static inline Ptr<Object>
__dummy_function_to_force_template_instantiation (Ptr<Object> obj, TypeId typeId)
{
  return obj->GetObject<Object> (typeId);
}


static inline void
__dummy_function_to_force_template_instantiation_v2 ()
{
  Time t1 = Seconds (1), t2 = Seconds (2), t3 = Seconds (3);
  t1 = t2 + t3;
  t1 = t2 - t3;
  t1 < t2;
  t1 <= t2;
  t1 == t2;
  t1 != t2;
  t1 >= t2;
  t1 > t2;


  int64x64_t s1(2), s2(3), s3;
  s1 = s2 + s3;
  s1 = s2 - s3;
  s1 < s2;
  s1 <= s2;
  s1 == s2;
  s1 != s2;
  s1 >= s2;
  s1 > s2;
  
}

}

