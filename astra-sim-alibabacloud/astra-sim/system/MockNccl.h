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

#ifndef ASTRA_SIM_MOCKNCCL_MOCKNCCL_H
#define ASTRA_SIM_MOCKNCCL_MOCKNCCL_H

#define NCCL_NUM_ALGORITHMS 6 // Tree/Ring/CollNet*
#define NCCL_ALGO_UNDEF -1
#define NCCL_ALGO_TREE 0
#define NCCL_ALGO_RING 1
#define NCCL_ALGO_COLLNET_DIRECT 2
#define NCCL_ALGO_COLLNET_CHAIN 3
#define NCCL_ALGO_NVLS 4
#define NCCL_ALGO_NVLS_TREE 5

#define NCCL_NUM_PROTOCOLS 3 // Simple/LL/LL128
#define NCCL_PROTO_UNDEF -1
#define NCCL_PROTO_LL 0
#define NCCL_PROTO_LL128 1
#define NCCL_PROTO_SIMPLE 2
#define NCCL_WORK_SIZE 512

/* Array indexes used below */
#define VOLTA_COMPCAP_IDX 0
#define AMPERE_COMPCAP_IDX 1
#define HOPPER_COMPCAP_IDX 2
#define NCCL_TOPO_CPU_VENDOR_AMD 2


#define NCCL_TOPO_CPU_ARCH_X86 1
#define NCCL_TOPO_CPU_ARCH_POWER 2
#define NCCL_TOPO_CPU_VENDOR_INTEL 1
#define NCCL_TOPO_CPU_VENDOR_AMD 2

#define NCCL_TOPO_PATTERN_BALANCED_TREE 1   // 目前正在使用的 Spread NIC traffic between two GPUs (Tree parent + one child on first GPU, second child on second GPU)
#define NCCL_TOPO_PATTERN_SPLIT_TREE 2      // Spread NIC traffic between two GPUs (Tree parent on first GPU, tree children on the second GPU)
#define NCCL_TOPO_PATTERN_TREE 3            // All NIC traffic going to/from the same GPU
#define NCCL_TOPO_PATTERN_RING 4            // Ring
#define NCCL_TOPO_PATTERN_NVLS 5            // NVLS+SHARP and NVLS+Tree
// Latencies in us, Bandwidths in GB/s

#define NCCL_NUM_FUNCTIONS 5 // Send/Recv not included for now
typedef enum { ncclFuncBroadcast, ncclFuncReduce, ncclFuncAllGather, ncclFuncReduceScatter, ncclFuncAllReduce, ncclFuncSendRecv, ncclFuncSend, ncclFuncRecv, ncclNumFuncs} ncclFunc_t;

// LL128 max BW per channel
static const double llMaxBws[3][3] = {
    /* Volta-N1/Intel-N2/Intel-N4) */ {39.0, 39.0, 20.4},
    /* Ampere-N1/AMD-N2/AMD-N4) */ {87.7, 22.5 /*avg of ring & tree*/, 19.0},
    /* Hopper-N1/AMD-N2/AMD-N4) */ {87.7, 22.5 /*avg of ring & tree*/, 19.0}
};
// Latencies in us, Bandwidths in GB/s
// Tree { LL, LL128, Simple } , Ring { LL, LL128, Simple }
static const float baseLat  [NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS] = {
    {  6.8, 14.0,    0 }, {  6.6, 14.0,  8.4 }, // Tree, Ring
    {  6.8, 14.0,    0 }, {  6.8, 14.0,    0 },       // Collnet Direct, Chain
    {    0,    0, 23.0 }, {    0,    0, 23.0 }};     // NVLS, NVLS Tree

static const double perChMaxRingLL128Bws[3][3] = {
    /* Volta (N1/N2/N4) */  {20.0, 20.0, 20.0},
    /* Ampere (N1/N2/N4) */ {20.0, 20.0, 20.0},
    /* Hopper (N1/N2/N4) */ {36.7, 36.7, 36.7},
};
static const double perChMaxTreeLL128Bws[3][3] = {
    /* Volta (N1/N2/N4) */  {20.0, 20.0, 20.0},
    /* Ampere (N1/N2/N4) */ {20.0, 20.0, 20.0},
    /* Hopper (N1/N2/N4) */ {36.7, 36.7, 29.0},
};
static const double perChMaxTreeBws[3][3] = {
    /* Volta (N1/N2/N4) */  {26.5, 18.5, 10.0},
    /* Ampere (N1/N2/N4) */ {24.0, 23.6, 17.8},
    /* Hopper (N1/N2/N4) */ {38.7, 41.4, 36.0},
};

// NVLink, PCI, Network 硬件时延
#define NCCL_HW_NVLINK 0
#define NCCL_HW_PCI 1
#define NCCL_HW_NET 2
static float hwLat [3][NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS] =
    { /* NVLINK */
     { /* Tree (LL/LL128/Simple)*/ { .6, 1.25,  4 }, /* Ring (LL/LL128/Simple)*/ { .6, 1.9, 3.4 },
      /* CollNetDirect (Simple)*/ { 0, 0, 8.0 }, /* CollNetChain (Simple)*/ { 0, 0, 4.75 },
      /* NVLS */ { 0, 0, 0 }, /* NVLSTree */ { 0, 0, 0 } },
     /* PCI */
     { /* Tree (LL/LL128/Simple)*/ { 1.0, 1.9,  6 }, /* Ring (LL/LL128/Simple)*/ { 1.0, 2.5, 5.7 },
      /* CollNetDirect (Simple)*/ { 0, 0, 8.0 }, /* CollNetChain (Simple)*/ { 0, 0, 8.0 },
      /* NVLS */ { 0, 0, 0 }, /* NVLSTree */ { 0, 0, 0 } },
     /* NET */
     { /* Tree (LL/LL128/Simple)*/ { 5.0, 8.5, 14 }, /* Ring (LL/LL128/Simple)*/ { 2.7, 4.0, 14.0 },
      /* CollNetDirect (Simple)*/ { 0, 0, 10.7 }, /* CollNetChain (Simple)*/ { 0, 0, 14 },
      /* NVLS */ { 0, 0, 18 }, /* NVLSTree */ { 0, 0, 19 } }
};

// We want link types and path types to match as much as possible
#define LINK_LOC 0
#define LINK_NVL 1
// Skipping 2 for PATH_NVB
#define LINK_PCI 3
// Skipping 4 for PATH_PXB
// Skipping 5 for PATH_PXN
// Skipping 6 for PATH_PHB
#define LINK_SYS 7
#define LINK_NET 8
#define PCI_BW 12.0           // PCI Gen3 x16

// Local (myself)
#define PATH_LOC 0

// Connection traversing NVLink
#define PATH_NVL 1

// Connection through NVLink using an intermediate GPU
#define PATH_NVB 2

// Connection traversing at most a single PCIe bridge
#define PATH_PIX 3

// Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
#define PATH_PXB 4

// Connection between a GPU and a NIC using an intermediate GPU. Used to enable rail-local, aggregated network send/recv operations.
#define PATH_PXN 5

// Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
#define PATH_PHB 6

// Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
#define PATH_SYS 7

// Connection through the network
#define PATH_NET 8

// Disconnected
#define PATH_DIS 9 

#define DIVUP(x, y) \
    (((x)+(y)-1)/(y))
template<typename X, typename Z = decltype(X()+int())>
constexpr Z alignUp(X x, int a) {
  return (x+a-1) & Z(-a);
}

enum ncclWorkType : uint8_t {
  ncclWorkTypeUnused=0,
  ncclWorkTypeColl=1,
  ncclWorkTypeP2p=2,
  ncclWorkTypeRegColl=3
};

struct ncclWorkHeader {
  union {
    int32_t workNext;  // when isLast=0: Offset from kernel argument workHead
    uint32_t doneAcks; // when isLast=1: Monotonic (mod 1<<32) ack value to send back.
  };
  uint16_t funcIndex;
  uint8_t isLast:1; // last work for this kernel
  uint8_t inFifo:1; // is this work in the fifo
  enum ncclWorkType type;
};

struct ncclWorkElem {
  union {
    uint8_t flagBits;
    struct {
      uint8_t isUsed:1, redOpArgIsPtr:1, regUsed:1;
    };
  };
  uint8_t nWarps;
  uint8_t direct;

  const void * sendbuff;
  void * recvbuff;

  size_t count;
  size_t lastChunkSize;
  uint32_t root;
  uint8_t bid;
  uint8_t nChannels;
  uint64_t redOpArg;
};

#define NCCL_MAX_WORK_ELEMENTS ((NCCL_WORK_SIZE - alignUp(sizeof(ncclWorkHeader), alignof(ncclWorkElem)))/sizeof(ncclWorkElem))
// Trees are not perfectly sticking to the model for medium sizes. Applying a static correction
// factor is not ideal but works quite well. Powers of two, 64 B to 256MB.
static float treeCorrectionFactor[NCCL_NUM_PROTOCOLS][23] = {
    { 1.0, 1.0, 1.0, 1.0,  .9,  .8,  .7,  .7,  .7,  .7,  .6,  .5,  .4,  .4,  .5,  .6,  .7,  .8,  .9, 1.0, 1.0, 1.0, 1.0 },
    { 1.0, 1.0, 1.0, 1.0, 1.0,  .9,  .8,  .8,  .8,  .7,  .6,  .6,  .6,  .6,  .6,  .6,  .8,  .9,  .9,  .9,  .9, 1.0, 1.0 },
    {  .9,  .9,  .9,  .9,  .9,  .9,  .9,  .8,  .7,  .6,  .6,  .5,  .5,  .5,  .5,  .6,  .7,  .8,  .7,  .7,  .8,  .9,  .9 }
};
#define MAXCHANNELS 10
#define NCCL_TOPO_MAX_NODES 10
#define NCCL_MAX_TREE_ARITY 2
 struct ncclTopoGraph {
    // Input / output
    int id; // ring : 0, tree : 1, collnet : 2
    int pattern;
    int crossNic;
    int collNet;
    int minChannels;
    int maxChannels;
    // Output
    int nChannels;    //搜索到的Chanel数
    float bwIntra;    //节点内单个chnnel带宽
    float bwInter;    //节点间单个channel带宽
    float latencyInter;
    int typeIntra;    //节点内的channel路径类型
    int typeInter;    //节点间的channel路径类型
    int sameChannels; //channel是否一样
    int nHops;
    int intra[MAXCHANNELS*NCCL_TOPO_MAX_NODES];   //节点内每个channel路径
    int inter[MAXCHANNELS*2];     //  节点间每个channel路径
  };

#endif // ASTRA_SIM_MOCKNCCL_MOCKNCCL_H
