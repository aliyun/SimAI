
/* We need to scan the complete network module so that all address
 * formats are pulled, otherwise just a few basic ones are seen, and
 * that affects implicit conversions detected by pybindgen. */
#include "ns3/network-module.h"

#include "ns3/applications-module.h"

