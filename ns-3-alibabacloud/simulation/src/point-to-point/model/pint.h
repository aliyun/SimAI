#ifndef PINT_H
#define PINT_H

#include <stdint.h>

namespace ns3{
class Pint{
public:
	static const uint32_t max_concurrent = 512; // max number of concurrent flows
	static double log_base, log_factor; // used for PINT
	static void set_log_base(double base);
	static int get_n_bits();
	static int get_n_bytes();
	static uint16_t encode_u(double u);
	static double decode_u(uint16_t p);
};
} /* namespace ns3 */

#endif /* PINT_H */
