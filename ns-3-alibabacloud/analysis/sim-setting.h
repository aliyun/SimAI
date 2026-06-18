#ifndef SIM_SETTING_H
#define SIM_SETTING_H

#include <stdint.h>
#include <vector>
#include <cstdio>
#include <cassert>
#include <unordered_map>

class SimSetting{
public:
	std::unordered_map<uint16_t, std::unordered_map<uint8_t, uint64_t> > port_speed; // port_speed[i][j] is node i's j-th port's speed
	uint32_t win; // window bound

	void Serialize(FILE* file){
		// write port_speed
		uint32_t len = 0;
		for (auto i: port_speed)
			for (auto j : i.second)
				len++;
		fwrite(&len, sizeof(len), 1, file);
		for (auto i: port_speed)
			for (auto j : i.second){
				fwrite(&i.first, sizeof(i.first), 1, file);
				fwrite(&j.first, sizeof(j.first), 1, file);
				fwrite(&j.second, sizeof(j.second), 1, file);
			}
		// write win
		fwrite(&win, sizeof(win), 1, file);
	}
	void Deserialize(FILE *file){
		int ret;
		// read port_speed
		uint32_t len;
		ret = fread(&len, sizeof(len), 1, file);
		for (uint32_t i = 0; i < len; i++){
			uint16_t node;
			uint8_t intf;
			uint64_t bps;
			ret &= fread(&node, sizeof(node), 1, file);
			ret &= fread(&intf, sizeof(intf), 1, file);
			ret &= fread(&bps, sizeof(bps), 1, file);
			port_speed[node][intf] = bps;
		}
		// read win
		ret &= fread(&win, sizeof(win), 1, file);

		// make sure read successfully
		assert(ret != 0);
	}
};

#endif
