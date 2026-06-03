#include <cstdio>
#include <stdint.h>
#include <vector>
#include <utility>
#include <algorithm>
#include <unistd.h>
#include <string>
#include <string.h>

using namespace std;

string prefix="fct_fat";
uint32_t step = 5, type = 0;
uint64_t time_limit = 3000000000lu;
vector<pair<uint32_t, double> > steps;
vector<string> cc;

void parse_opt(int argc, char* argv[]){
	for (int opt=0; (opt = getopt(argc, argv, "p:s:S:t:T:c:")) != -1;) {
		switch (opt) {
			case 'p':
				prefix = optarg;
				break;
			case 's':
				step = atoi(optarg);
				break;
			case 'S':
				{
					FILE* step_file = fopen(optarg, "r");
					for (pair<uint32_t, double> p; fscanf(step_file, "%u%lf", &p.first, &p.second) != EOF;){
						steps.push_back(p);
					}
				}
				break;
			case 't':
				type = atoi(optarg);
				break;
			case 'T':
				time_limit = atoll(optarg);
				break;
			case 'c':
				for (char *tok = strtok(optarg, ","); tok != NULL; tok = strtok(NULL, ",")){
					cc.push_back(tok);
				}
				break;
			default: /* '?' */
				fprintf(stderr, 
						"usage: %s [-h] [-p PREFIX] [-s STEP] [-t TYPE] [-T TIME_LIMIT] [-c CC_LIST]\n"
						"\n"
						"optional arguments:\n"
						"  -h, --help     show this help message and exit\n"
						"  -p PREFIX      Specify the prefix of the fct file. Usually like\n"
						"                 fct_<topology>_<trace>\n"
						"  -s STEP\n"
						"  -S STEP_FILE   Specify the file of the steps\n"
						"  -t TYPE        0: normal, 1: incast, 2: all\n"
						"  -T TIME_LIMIT  only consider flows that finish before T\n"
						"  -c CC_LIST     Specify a list of cc\n",
						argv[0]);
				exit(EXIT_FAILURE);
		}
	}
}

bool compare(pair<uint32_t, float> a, pair<uint32_t, float> b){
	return a.first < b.first;
}

bool compare_second(pair<uint32_t, float> a, pair<uint32_t, float> b){
	return a.second < b.second;
}

int main(int argc, char* argv[]){
	parse_opt(argc, argv);
	vector<vector<double> > res;
	if (steps.size() > 0){
		for (auto p : steps)
			res.push_back(vector<double> (1, p.second / 100.));
	}else{
		for (int p = 0; p < 100; p += step)
			res.push_back(vector<double> (1, (p+step) / 100.));
	}
	for (int i = 0; i < cc.size(); i++){
		string c = cc[i];
		vector<pair<uint32_t, float> > tuples;
		FILE* file = fopen(("../simulation/mix/"+prefix+"_"+c+".txt").c_str(), "r");
		uint16_t port;
		uint32_t size;
		uint64_t start_time, fct, standalone_fct;
		while (fscanf(file, "%*s%*s%*s%hu%u%lu%lu%lu", &port, &size, &start_time, &fct, &standalone_fct) != EOF){
			if (((port == 100 && !(type & 1)) || (port == 200 && type > 0)) && start_time + fct < time_limit){
				float slowdown = double(fct) / standalone_fct;
				tuples.push_back(make_pair(size, slowdown < 1 ? 1.0 : slowdown));
			}
		}
		fclose(file);

		sort(tuples.begin(), tuples.end(), compare);

		if (steps.size() > 0){
			uint64_t l = 0, r = 0;
			int i = 0;
			for (auto p : steps){
				while (r < tuples.size() && tuples[r].first <= p.first)
					r++;
				sort(tuples.begin() + l, tuples.begin() + r, compare_second);
				res[i].push_back(p.first);
				res[i].push_back(tuples[l + uint64_t((r-l)*0.5)].second);
				res[i].push_back(tuples[l + uint64_t((r-l)*0.95)].second);
				res[i].push_back(tuples[l + uint64_t((r-l)*0.99)].second);
				l = r;
				i++;
			}
		}else{
			for (int p = 0; p < 100; p += step){
				uint64_t l = p * tuples.size() / 100;
				uint64_t r = (p + step) * tuples.size() / 100;
				uint32_t largest_size = tuples[r-1].first;
				sort(tuples.begin() + l, tuples.begin() + r, compare_second);
				res[p / step].push_back(largest_size);
				res[p / step].push_back(tuples[l + uint64_t((r-l)*0.5)].second);
				res[p / step].push_back(tuples[l + uint64_t((r-l)*0.95)].second);
				res[p / step].push_back(tuples[l + uint64_t((r-l)*0.99)].second);
			}
		}
	}

	for (int i = 0; i < res.size(); i++){
		printf("%.6lf %.0lf", res[i][0], res[i][1]);
		for (int j = 0; j < cc.size(); j++)
			printf("\t%.3f %.3f %.3f", res[i][j*4 + 2], res[i][j*4+3], res[i][j*4+4]);
		printf("\n");
	}
	return 0;
}
