#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "calbusbw.h"
#include <algorithm>
#include <ctype.h>
#include "astra-sim/system/AstraParamParse.hh"
char info[1024] = "Success!";
int retcode = 0;

float calculateAlgoBw(CalculationParameters params) {
    return 0.0; 
}

float getNvlinkBw(GPUType node_type) {
    float nvlink_bw = 0.0;
    if (node_type == GPUType::H100 || node_type == GPUType::H20 ) {
        nvlink_bw = SM90_NVLINK_BW * H100_NVLINKS;
    } else if (node_type == GPUType::H800) {
        nvlink_bw = SM90_NVLINK_BW * H800_NVLINKS;
    } else if (node_type == GPUType::A100) {
        nvlink_bw = SM80_NVLINK_BW * A100_NVLINKS;
    } else if (node_type == GPUType::A800) {
        nvlink_bw = SM80_NVLINK_BW * A800_NVLINKS;
    } else {
        strcpy(info, "Warning: unknown machine type. Please choose from H20, H100, H800, A100, A800.");
        retcode = 1;
        return -1;
    }
    return nvlink_bw;
}

float getNicBw(char* nic_type) {
    float nic_bw = 0.0;
    if (strcmp(nic_type, "CX6") == 0 || strcmp(nic_type, "cx6") == 0) {
        nic_bw = CX6_BW;
    } else if (strcmp(nic_type, "CX7") == 0 || strcmp(nic_type, "cx7") == 0) {
        nic_bw = CX7_BW;
    } else if (strcmp(nic_type, "BF3") == 0 || strcmp(nic_type, "bf3") == 0) {
        nic_bw = BF3_BW;
    } else {
        strcpy(info, "Warning: unknown NIC type. Please choose from CX6, CX7, BF3.");
        retcode = 1;
        return -1;
    }
    return nic_bw;
}

float calcTreeBusBw(int gpus_per_node, int node_count, float nvlink_bw, float nic_bw, float nics_per_node, float all_gather_bus_bw) {
    int nranks = gpus_per_node * node_count;
    if (nranks == 1) return 5000.0;
    if (node_count == 1) {
        return all_gather_bus_bw * (gpus_per_node-1) / gpus_per_node;
    } else {
        float algbw_nic = nic_bw * nics_per_node;
        if (node_count == 2) {
            algbw_nic *= 2;
        } else if (node_count == 3) {
            algbw_nic *= (4.0/3.0);
        }
        if (gpus_per_node == 1) {
            return algbw_nic * (nranks-1) / nranks;
        }
        float algbw_nvlink = nvlink_bw * gpus_per_node / (gpus_per_node-1);
        return (algbw_nic < algbw_nvlink) ? algbw_nic * (nranks-1) / nranks : algbw_nvlink * (nranks-1) / nranks;
    }
}

float calcNVLSBusBw(int gpus_per_node, int node_count, float NVLS_bw, float nic_bw, float nics_per_node) {
    int nranks = gpus_per_node * node_count;
    
    if (gpus_per_node != 8) return -1.0;
    float algo_nvls_busbw = NVLS_bw * gpus_per_node / (gpus_per_node-1);

    if (node_count == 1) {
        return algo_nvls_busbw * (nranks-1) / nranks;
    } else {
        float algbw_nic = nic_bw * nics_per_node;
        if (node_count == 2) {
            algbw_nic *= 2;
        } else if (node_count == 3) {
            algbw_nic *= (4.0/3.0);
        }
        if (gpus_per_node == 1) {
            return algbw_nic * (nranks-1) / nranks;
        }
        return (algbw_nic < algo_nvls_busbw) ? algbw_nic * (nranks-1) / nranks : algo_nvls_busbw * (nranks-1) / nranks;
    }
}

int lower_compare(char *coll_type, const char *lower_str) {
    //return strcasecmp(coll_type, lower_str);
    char temp_str[strlen(coll_type) + 1];
    
    for (int i = 0; i < strlen(coll_type); i++) {
        temp_str[i] = tolower((unsigned char)coll_type[i]);
    }
    temp_str[strlen(coll_type)] = '\0';  
    
    if (strcmp(temp_str, lower_str) == 0) {
        return 0; 
    }
    return 1; 
}

float calculateBusBw(CalculationParameters* params) {
    float nvlink_bw; 
    if (params->bw_intra > 0.0) {
        nvlink_bw = params->bw_intra;
    } else {
        nvlink_bw = getNvlinkBw(params->node_type);
    }
    float nic_bw;
    if (params->bw_per_nic > 0.0) {
        nic_bw = params->bw_per_nic;
    } else {
        nic_bw = getNicBw(params->nic_type);
    }
    float all_gather_bus_bw = 0.0;
    
    int gpus_per_node = params->gpus_pernode;
    int nics_per_node = params->nics_pernode;
    float real_nics_per_node = params->real_nics_pernode;
    int node_count = params->node_count;
    int nranks = node_count * gpus_per_node;
    params->is_nvlink = false; //nvlink or nic
    if (nvlink_bw <= 0 || nic_bw <= 0 || gpus_per_node < 1 || nics_per_node < 1 || node_count < 1) {
        return -1;
    }

    if (real_nics_per_node * nic_bw > nvlink_bw) {
        if (params->cross_nic == 2) params->cross_nic = 1;
    } else {
        if (params->cross_nic == 2) params->cross_nic = 0;
    }

    if (node_count == 1) {
        all_gather_bus_bw = nvlink_bw;
    } else {
        if (gpus_per_node == 1) {
            all_gather_bus_bw = nic_bw * real_nics_per_node;
        } else {
            all_gather_bus_bw = (nvlink_bw < nic_bw * real_nics_per_node) ? (params->is_nvlink = true, nvlink_bw) : nic_bw * real_nics_per_node;
            if (params->cross_nic == 1) {
                params->is_nvlink = false;
                all_gather_bus_bw = (nvlink_bw * gpus_per_node / (gpus_per_node-1) < nic_bw * real_nics_per_node) ? (params->is_nvlink = true, nvlink_bw * gpus_per_node / (gpus_per_node-1) ): nic_bw * real_nics_per_node;
            }
        }
    }

    float tree_bus_bw = 0.0;
    float nvls_bus_bw = 0.0;
    tree_bus_bw = calcTreeBusBw(gpus_per_node, node_count, nvlink_bw, nic_bw, real_nics_per_node, all_gather_bus_bw);
    if (params->node_type == GPUType::H100 || params->node_type == GPUType::H20) {
        nvls_bus_bw = calcNVLSBusBw(gpus_per_node, node_count, H100_NVLS_BW, nic_bw, real_nics_per_node);
    } else if (params->node_type == GPUType::H800) {
        nvls_bus_bw = calcNVLSBusBw(gpus_per_node, node_count, H800_NVLS_BW, nic_bw, real_nics_per_node);
    }

    
    if (lower_compare(params->coll_type, "allreduce") == 0) {
        if (lower_compare(params->nccl_algo, "ring") == 0) {
            return all_gather_bus_bw;
        } else if (lower_compare(params->nccl_algo, "tree") == 0) {
            return tree_bus_bw;
        } else if (lower_compare(params->nccl_algo, "nvls") == 0 || lower_compare(params->nccl_algo, "nvlstree") == 0) {
            if (lower_compare(params->nccl_algo, "nvls") == 0 && node_count > 1) params->nccl_algo = "nvlstree";
            if (lower_compare(params->nccl_algo, "nvlstree") == 0 && node_count == 1) params->nccl_algo = "nvls";
            if (gpus_per_node == 8) {
                if (params->node_type == GPUType::H100 || params->node_type == GPUType::H800|| params->node_type == GPUType::H20) {
                    return nvls_bus_bw;
                } else {
                    strcpy(info, "Warning: unsupported machine type for NVLS algorithm. Please choose from H20,H100,H800.");
                    retcode = 1;
                    return -1;
                }
            } else {
                strcpy(info, "Warning: unsupported GPU count for NVLS algorithm. Please use 8 GPUs per node.");
                retcode = 1;
                return -1;
            }
        } else {
            if (nvls_bus_bw > tree_bus_bw) {
                if (all_gather_bus_bw > nvls_bus_bw) {
                    params->nccl_algo = "Ring";
                    return all_gather_bus_bw;
                } else {
                     if (node_count > 1) {
                            params->nccl_algo = strdup("NVLSTree");
                        } else {
                            params->nccl_algo = strdup("NVLS");
                        }
                    return nvls_bus_bw;
                }
            } else {
                if (all_gather_bus_bw > tree_bus_bw) {
                    params->nccl_algo = "Ring";
                    return all_gather_bus_bw;
                } else {
                    params->nccl_algo = "Tree";
                    return tree_bus_bw;
                }
            }
        }
        

    } else if (lower_compare(params->coll_type, "allgather") == 0) {
        params->nccl_algo = "Ring";
        return all_gather_bus_bw;

    } else if (lower_compare(params->coll_type, "alltoall") == 0) {
        params->nccl_algo = "none";
        if (node_count == 1) {
            params->is_nvlink = true;
            return nvlink_bw;
        }
        return nic_bw * real_nics_per_node / gpus_per_node * (nranks-1) / ((node_count-1)*gpus_per_node) ;

    } else if (lower_compare(params->coll_type, "broadcast") == 0) {
        params->nccl_algo = "Ring";
        return all_gather_bus_bw;

    } else if (lower_compare(params->coll_type, "reducescatter") == 0) {
        params->nccl_algo = "Ring";
        return all_gather_bus_bw;

    } else if (lower_compare(params->coll_type, "reduce") == 0) {
        params->nccl_algo = "Ring";
        return all_gather_bus_bw;

    } else {
        strcpy(info, "Warning: unknown collective type. Please choose from allreduce, allgather, alltoall, broadcast, reducescatter, reduce, multiallreduce.");
        retcode = 1;
        return -1;
    }
    return -1;
}
std::vector<std::vector<std::string>> readCSV(const std::string &filePath) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file");
    }
    std::vector<std::vector<std::string>> data;
    std::string line;
    bool isFirstLine = true;


    while (std::getline(file, line)) {

        if (isFirstLine) {
            isFirstLine = false;
            continue;
        }

        std::stringstream lineStream(line);
        std::string cell;
        std::vector<std::string> rowData;
        while (std::getline(lineStream, cell, ',')) {
            cell.erase(0, cell.find_first_not_of(' ')); 
            cell.erase(cell.find_last_not_of(' ') + 1); 
            if (cell.empty()) {
                cell = "1";
            }
            rowData.push_back(cell);
        }
        
        if (!rowData.empty()) {
            data.push_back(rowData);
        }
    }
    
    return data;
}
void printData(const std::vector<std::vector<std::string>> &data) {
    for (const auto &row : data) {
        for (const auto &cell : row) {
            std::cout << cell << " "; 
        }
        std::cout << std::endl; 
    }
}


BusBwResult cal_busbw(GPUType node_type,float bw_intra,float bw_per_nic, float nics_pernode,int node_count,char* coll_type,int gpus_pernode,char* nic_type) {
    BusBwResult result;
    CalculationParameters params;
    memset(&params, 0, sizeof(params));
    retcode = 0;
    params.node_count = node_count;
    
    params.gpus_pernode = gpus_pernode;
    params.nics_pernode = nics_pernode;
    params.bw_per_nic = bw_per_nic;
    params.bw_intra = bw_intra;
    params.group_split_mask = 0;
    params.nccl_algo = "ring";
    params.cross_nic = 2;
    params.coll_type = coll_type;
    params.node_type = node_type;
    params.nic_type = nic_type;
    // if (argc > 1 && strcmp(argv[1], "--help") == 0) {
    //     print_usage(argv[0]);
    //     return 1;
    // }
    // for (int i = 1; i < argc; i++){
    //     parseParams(argc, argv, &i, &params);
    // }
    params.real_nics_pernode = (float)params.nics_pernode;

    if (params.node_count < 1) {
        strcpy(info, "Error: The number of nodes must be greater than 0.");
        retcode = 1;
    }
    if (lower_compare(params.nccl_algo, "none")) {
        if (lower_compare(params.nccl_algo, "ring") && lower_compare(params.nccl_algo, "tree") && lower_compare(params.nccl_algo, "nvls") && lower_compare(params.nccl_algo, "nvlstree")) {
            strcpy(info, "Warning: the selected algorithm is not supported.");
        }
    }

    if (params.group_split_mask != 0 && params.group_split_mask != 1 && params.group_split_mask != 3 && params.group_split_mask != 7) {
        strcpy(info, "Warning: the value of group_split_mask can only be 0, 1, 3, 7. Default is 0.");
        params.group_split_mask = 0;
    } else if (params.group_split_mask != 0 && params.gpus_pernode != 8) {
        // 当前只支持8GPU机型的multi- 测试
        strcpy(info, "Warning: currently, only 8GPU nodes are supported for split_mask testing.");
        params.group_split_mask = 0;
    }

    if (lower_compare(params.coll_type, "allreduce")  && lower_compare(params.nccl_algo, "none") && lower_compare(params.nccl_algo, "ring")) {
        strcpy(info, "Warning: only allreduce can use other algorithms except ring.");
        params.nccl_algo = "Ring";
    }

    if (lower_compare(params.coll_type, "multiallreduce") == 0 || lower_compare(params.coll_type, "multialltoall") == 0) {
        params.nccl_algo = "Ring";
        params.cross_nic = 2;
        if (params.gpus_pernode == 8) {
            params.group_split_mask = 7;
        } else {
            params.real_nics_pernode = (float)params.nics_pernode / params.gpus_pernode;
            params.gpus_pernode = 1;
        }
        params.coll_type += strlen("multi");
    }

    if (params.group_split_mask == 7) {
        params.gpus_pernode = 1;
        params.real_nics_pernode = (float)params.nics_pernode / 8.0;
    } else if (params.group_split_mask == 3) {
        params.gpus_pernode = 2;
        params.real_nics_pernode = (float)params.nics_pernode / 4.0;
    } else if (params.group_split_mask == 1) {
        params.gpus_pernode = 4;
        params.real_nics_pernode = (float)params.nics_pernode / 2.0;
    }
    
    if (params.gpus_pernode * params.node_count == 1) {
        strcpy(info, "Warning: collective communication requires the participation of at least two gpus.");
        retcode = 1;
    }

    float busBw = 0.0;

    if (retcode == 0){
        busBw = calculateBusBw(&params);
    }

    if (params.node_count == 1) {
        params.cross_nic = 0;
    }

    if (retcode == 1) {
        printf("{\"retcode\":%d, \"info\":\"%s\", \"theoretical_bus_bw\":\"-1\", \"nccl_algo\":\"none\", \"cross_nic\":2}\n", retcode, info);
    } else {
        printf("{\"retcode\":%d, \"info\":\"%s\", \"node_count\":%d, \"nic_type\":\"%s\", \"gpus_pernode\":%d, \"nics_pernode\":%.1f, \"coll_type\":\"%s\", \"cross_nic\":%d, \"nccl_algo\":\"%s\", \"theoretical_bus_bw_GBps\":%.3lf}\n", retcode, info, params.node_count, params.nic_type, params.gpus_pernode, params.real_nics_pernode, params.coll_type, params.cross_nic, params.nccl_algo, busBw);
    }
    result.busbw = busBw;
    
    result.is_nvlink = params.is_nvlink;
    return result;
}
struct DataRow {
    std::string size;
    std::vector<double> values;
};
double interpolate(double size, double size1, double size2, double value1, double value2) {
    return value1 + (value2 - value1) * (size - size1) / (size2 - size1);
}
float getValue(double datasize, int _temp_nnode, const std::vector<std::vector<std::string>>& data) {
    int colIndex = 0;

    if (_temp_nnode == 1) {
        colIndex = 1; 
    } else if (_temp_nnode == 2) {
        colIndex = 2;  
    } else if (_temp_nnode == 4) {
        colIndex = 3;  
    } else if (_temp_nnode == 8) {
        colIndex = 4;  
    } else if (_temp_nnode == 16) {
        colIndex = 5;  
    } else if (_temp_nnode == 32) {
        colIndex = 6;  
    } else if (_temp_nnode == 64) {
        colIndex = 7;  
    } else if (_temp_nnode == 128) {
        colIndex = 8;  
    } else if (_temp_nnode == 9) {
        colIndex = 9;  
    }
    else {
        colIndex = 5; 
    }
    if (datasize == 0) {
        return 1.0;
    }
    double minSize = std::stod(data.front()[0]);
    if (datasize < minSize) {
        return std::stod(data.front()[colIndex])/std::stod(data.back()[colIndex]);
    }

    for (size_t i = 0; i < data.size() - 1; ++i) {
        double size1 = std::stod(data[i][0]);
        double size2 = std::stod(data[i+1][0]);
        if (datasize >= size1 && datasize <= size2) {
            double value1 = std::stod(data[i][colIndex]);
            double value2 = std::stod(data[i+1][colIndex]);
            return interpolate(datasize, size1, size2, value1, value2)/std::stod(data.back()[colIndex]);
        }
    }
    throw std::runtime_error("Data size out of range");
}

float cal_ratio(std::vector<std::vector<std::string>> nic_ratio_data,std::vector<std::vector<std::string>> nvlink_ratio_data,std::vector<std::vector<std::string>> ata_ratio_data,uint64_t data_size,int nranks,int tp_size,uint32_t gpus_per_server,char* group_type,char* coll_type,bool is_nvlink){
    if ((strcmp(coll_type, "allgather") == 0 || strcmp(coll_type, "reducescatter") == 0 ) && strcmp(group_type, "tp") == 0 ){
        auto data = is_nvlink ? nvlink_ratio_data : nic_ratio_data;
        int _temp_nnode = (tp_size < gpus_per_server) ? 1 : tp_size / gpus_per_server ;
        return getValue(data_size, _temp_nnode, data);
    } else if (strcmp(coll_type, "alltoall") == 0 && strcmp(group_type, "ep") == 0){
        auto data = ata_ratio_data;
        if(tp_size * nranks <= gpus_per_server){
            return getValue(data_size, 1, data);
        }else if(tp_size >= gpus_per_server){    //multi
            return getValue(data_size, 9, data);
        } else {
            int _temp_nnode = (tp_size * nranks) / gpus_per_server;
            return getValue(data_size, _temp_nnode, data);
        }
    } else if (strcmp(coll_type, "alltoall") == 0 && strcmp(group_type, "tp") == 0){
        auto data = ata_ratio_data;
        if (tp_size <= gpus_per_server){
            return getValue(data_size, 1, data);
        } else {
            int _temp_nnode = tp_size / gpus_per_server;
            return getValue(data_size, _temp_nnode, data);
        }
    }
    else if(strcmp(group_type, "dp") == 0 || strcmp(group_type, "dp_ep") == 0){
        return 1; 
    }else{
        return 1;
    }
}