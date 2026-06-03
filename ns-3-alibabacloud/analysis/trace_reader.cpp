#include <cstdio>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include "trace-format.h"
#include "trace_filter.hpp"
#include "utils.hpp"
#include "sim-setting.h"

using namespace ns3;
using namespace std;

int main(int argc, char** argv){
	if (argc != 2 && argc != 3){
		printf("Usage: ./trace_reader <trace_file> [filter_expr]\n");
		return 0;
	}
	FILE* file = fopen(argv[1], "r");
	TraceFilter f;
	if (argc == 3){
		f.parse(argv[2]);
		if (f.root == NULL){
			printf("Invalid filter\n");
			return 0;
		}
	}
	//printf("filter: %s\n", f.str().c_str());

	// first read SimSetting
	SimSetting sim_setting;
	sim_setting.Deserialize(file);
	#if 0
	// print sim_setting
	for (auto i : sim_setting.port_speed)
		for (auto j : i.second)
			printf("%u,%u:%lu\n", i.first, j.first, j.second);
	#endif

	// read trace
	TraceFormat tr;
	while (tr.Deserialize(file) > 0){
		if (!f.test(tr))
			continue;
		print_trace(tr);
	}
}
