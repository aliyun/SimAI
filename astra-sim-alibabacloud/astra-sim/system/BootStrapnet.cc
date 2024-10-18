/* 
*Copyright (c) 2024, Alibaba Group;
*Licensed under the Apache License, Version 2.0 (the "License");
*you may not use this file except in compliance with the License.
*You may obtain a copy of the License at

*   http://www.apache.org/licenses/LICENSE-2.0

*Unless required by applicable law or agreed to in writing, software
*distributed under the License is distributed on an "AS IS" BASIS,
*WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*See the License for the specific language governing permissions and
*limitations under the License.
*/

#include <map>
#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <arpa/inet.h>
#include <netdb.h>
#include <mpi.h>
#include "BootStrapnet.hh"


using namespace std;


int world_size,local_rank;
std::map<int,std::string> rank2addr;

static void 
initBootStrapNetRank(int argc,char*argv[]){
    std::string host_file = argv[1];
    std::string line;
    std::ifstream file(host_file);
    int rank = 0;

    if(file.is_open()){
        while (std::getline(file,line)){
            /* code */
           rank2addr[rank] = line;
           rank++; 
        }
        file.close();
    } else{
        std::cerr << "Failed to open the file" << std::endl;
    }
}

void BootStrapNet(int argc, char *argv[]){
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &local_rank);
    MPI_Barrier(MPI_COMM_WORLD);
    initBootStrapNetRank(argc,argv);
}
