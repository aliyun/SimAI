#include <cmath>
#include <cstdlib>
#include <cstdio>

#include "pint.h"

namespace ns3{

double Pint::log_base = 1.05;
double Pint::log_factor = 1 / log(log_base);

void Pint::set_log_base(double base){
	log_base = base;
	log_factor = 1 / log(log_base);
}

int Pint::get_n_bits(){
	double max_value = log(max_concurrent * max_concurrent) * log_factor; // the max value of the encoded u
	return (int)ceil(log2(max_value)); // get the number of bits for the max encoded u
}
int Pint::get_n_bytes(){
	int n_bits = get_n_bits();
	if (n_bits == 0)
		return 0;
	return (n_bits - 1) / 8 + 1;
}

uint16_t Pint::encode_u(double u){
	uint32_t u_toInt = ceil(u * max_concurrent); // convert u to int so that the minimum possible u value is mapped to 1
	if (u_toInt == 0) u_toInt = 1;
	double power = log(u_toInt) * log_factor;
	uint16_t p_upper = ceil(power), p_lower = floor(power);
	double upper = pow(log_base, p_upper), lower = pow(log_base, p_lower);
	if (p_upper == p_lower)
		upper *= log_base;
	uint16_t p = (rand() % 65536 < (u_toInt - lower) / (upper - lower) * 65536) ? p_upper : p_lower;
	return p;
}

double Pint::decode_u(uint16_t p){
	return pow(log_base, p) / max_concurrent;
}

} /* namespace ns3 */
