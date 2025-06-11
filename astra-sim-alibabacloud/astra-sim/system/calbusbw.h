#ifndef CALBUSBW_H
#define CALBUSBW_H
#include "astra-sim/system/AstraParamParse.hh"
#define SM80_NVLINK_BW 20.0
#define SM90_NVLINK_BW 20.6
#define H100_NVLINKS   18
#define H800_NVLINKS   8
#define A100_NVLINKS   12
#define A800_NVLINKS   8

#define CX6_BW 23.5 // 25
#define CX7_BW 48.5 // 50
#define BF3_BW 48.5 // 50

#define H100_NVLS_BW 475.0
#define H800_NVLS_BW 215.0

#define H800_PCIE_BW 51.2 // 64*0.8
#define H100_PCIE_BW 51.2 // 64*0.8
#define A100_PCIE_BW 25.6 // 32*0.8
#define A800_PCIE_BW 25.6 // 32*0.8
#define NIC_RATIO_PATH "astra-sim-alibabacloud/inputs/ratio/nic_ratio.csv"
#define NVLINK_RATIO_PATH "astra-sim-alibabacloud/inputs/ratio/nvlink_ratio.csv"
#define ATA_RATIO_PATH "astra-sim-alibabacloud/inputs/ratio/ata_ratio.csv"
typedef struct {
    GPUType node_type;
    int node_count;
    char* nic_type;
    char* coll_type;
    int cross_nic;
    char* nccl_algo;
    int gpus_pernode;
    float nics_pernode;
    float bw_per_nic;
    float bw_intra;
    int group_split_mask;
    float real_nics_pernode;
    bool is_nvlink;
} CalculationParameters;

typedef struct {
    float busbw;
    int is_nvlink;
} BusBwResult;

BusBwResult cal_busbw(GPUType node_type, float bw_intra, float bw_per_nic, float nics_pernode, int node_count, char* coll_type, int gpus_pernode, char* nic_type);
float cal_ratio(std::vector<std::vector<std::string>> nic_ratio_data,std::vector<std::vector<std::string>> nvlink_ratio_data,std::vector<std::vector<std::string>> ata_ratio_data,uint64_t data_size,int nranks,int tp_size,uint32_t gpus_per_server,char* group_type,char* coll_type,bool is_nvlink);
std::vector<std::vector<std::string>> readCSV(const std::string &filePath);
float getValue(double datasize, int _temp_nnode, const std::vector<std::vector<std::string>>& data);
#endif // CALBUSBW_H