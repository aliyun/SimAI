#include <stdlib.h>
#include <string>
#include <cstdlib>
#include <stdio.h>
#include "cuda_runtime.h"
#include "../build/include/nccl.h"
#include <map>
#include <vector>
#include <iostream>

typedef enum {
  ncclFuncBroadcast = 0,
  ncclFuncReduce = 1,
  ncclFuncAllGather = 2,
  ncclFuncReduceScatter = 3,
  ncclFuncAllReduce = 4,
  ncclFuncSendRecv = 5,
  ncclFuncSend = 6,
  ncclFuncRecv = 7,
  ncclNumFuncs = 8
} ncclFunc_t;
const char *name_coll[ncclNumFuncs] = {
  "Broadcast",
  "Reduce",
  "AllGather",
  "ReduceScatter",
  "AllReduce",
  "SendRecv",
  "Send",
  "Recv"
};
const char *name_algo[6] = {
  "NCCL_ALGO_TREE",
  "NCCL_ALGO_RING",
  "NCCL_ALGO_COLLNET_DIRECT",
  "NCCL_ALGO_COLLNET_CHAIN",
  "NCCL_ALGO_NVLS",
  "NCCL_ALGO_NVLS_TREE"
};
const char *name_proto[3] = {
  "NCCL_PROTO_LL",
  "NCCL_PROTO_LL128",
  "NCCL_PROTO_SIMPLE"
};

struct ncclMockOutput {
    std::string mockName;
    std::map<int, std::map<int, std::vector<int>>> ringChannels;
    struct CollInfo {
        int algo, protocol, nchannels;
        int chunkSize, chunkCount, chunkSteps;
        int sliceSteps, stepSize;
        size_t alignCount, workCount, lastChunkCount;
        uint32_t steps;
    };
    std::map<ncclFunc_t, std::map<size_t, CollInfo>> collInfos;
};


#define NCCLCHECK(cmd) do {                         \
  ncclResult_t res = cmd;                           \
  if (res != ncclSuccess) {                         \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(res)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

void printRingChannels(const std::map<int, std::map<int, std::vector<int>>>& ringChannels) {
    int nChannels = ringChannels.size();
    if (nChannels == 0) {
        std::cout << "ringChannels is empty." << std::endl;
        return;
    }

    int nRanks = ringChannels.begin()->second.size();

    for (int c = 0; c < nChannels; c++) {
        if (ringChannels.count(c) == 0) continue;
        const auto& channel = ringChannels.at(c);
        for (int r = 0; r < nRanks; r++) {
            if (channel.count(r) == 0) continue;
            const auto& vec = channel.at(r);
            if (vec.size() != 4) {
                std::cout << "Unexpected vector size for channel " << c << ", rank " << r << std::endl;
                continue;
            }
            std::cout << "nChannels: " << nChannels 
                      << ", nranks: " << nRanks 
                      << ", ringChannels[" << c << "][" << r << "]: " 
                      << vec[0] << " " << vec[1] << " " << vec[2] << " " << vec[3] << std::endl;
        }
    }
}

void printCollInfos(const std::map<ncclFunc_t, std::map<size_t, ncclMockOutput::CollInfo>>& collInfos) {
  std::cout << "Printing all values in collInfos:" << std::endl;
  for (const auto& coll_pair : collInfos) {
    std::cout << "Collective operation type: " << name_coll[coll_pair.first] << std::endl;
    for (const auto& size_pair : coll_pair.second) {
      std::cout << "  Size: " << size_pair.first << " bytes" << std::endl;
      const auto& info = size_pair.second;
      std::cout << "    algo: " << name_algo[info.algo] << std::endl;
      std::cout << "    protocol: " << name_proto[info.protocol] << std::endl;
      std::cout << "    nchannels: " << info.nchannels << std::endl;
      std::cout << "    chunkSize: " << info.chunkSize << std::endl;
      std::cout << "    chunkCount: " << info.chunkCount << std::endl;
      std::cout << "    chunkSteps: " << info.chunkSteps << std::endl;
      std::cout << "    sliceSteps: " << info.sliceSteps << std::endl;
      std::cout << "    stepSize: " << info.stepSize << std::endl;
      std::cout << "    alignCount: " << info.alignCount << std::endl;
      std::cout << "    workCount: " << info.workCount << std::endl;
      std::cout << "    lastChunkCount: " << info.lastChunkCount << std::endl;
      std::cout << "    steps: " << info.steps << std::endl;
      std::cout << std::endl;
    }
  }
}

// #define DOCC
int main(int argc, char* argv[])
{
  ncclComm_t comms[4];

  //managing 8 devices
  const int nDev = 1;
  size_t size = 4*1024*1024;

  int devs[nDev] = {0};

  struct ncclMockOutput mock_outp;

  //initializing NCCL
  NCCLCHECK(ncclCommInitAll(comms, nDev, nullptr, &mock_outp));
  // setenv("NCCL_NUM_MOCK_GPU", "32", 1);
  // setenv("NCCL_NUM_MOCK_NODE", "4", 1);
  // NCCLCHECK(ncclCommInitAll(comms+1, nDev, nullptr));

  // NCCLCHECK(ncclGroupStart());
  // for (int i = 0; i < nDev; ++i)
  //   NCCLCHECK(ncclReduceScatter(NULL, NULL, size, ncclFloat, ncclSum,
  //       comms[i], NULL));
  // NCCLCHECK(ncclGroupEnd());

  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < nDev; ++i)
    NCCLCHECK(ncclAllGather(NULL, &mock_outp, size, ncclFloat,
        comms[i], NULL));
  NCCLCHECK(ncclGroupEnd());

  // NCCLCHECK(ncclGroupStart());
  // for (int i = 0; i < nDev; ++i)
  //   // NCCLCHECK(ncclAllReduce((const void*)sendbuff[i], (void*)recvbuff[i], size, ncclFloat, ncclSum,
  //   //     comms[i], s[i]));
  //   NCCLCHECK(ncclAllReduce(NULL, NULL, size, ncclFloat, ncclSum,
  //       comms[i], NULL));
  // NCCLCHECK(ncclGroupEnd());

  //finalizing NCCL
  // for(int i = 0; i < nDev; ++i) {
  //   ncclCommDestroy(comms[i]);
  // }
  printCollInfos(mock_outp.collInfos);
  printf("Success mockoutput: %s, allgatherInfo: %zu\n", mock_outp.mockName.c_str(), mock_outp.collInfos.size());
  // std::cout << "Printing ringChannels:" << std::endl;
  // printRingChannels(mock_outp.ringChannels);
  return 0;
}
