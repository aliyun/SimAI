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

#ifndef __COMMON_H__
#define __COMMON_H__

#undef PGO_TRAINING
#define PATH_TO_PGO_CONFIG "path_to_pgo_config"

#include <fstream>
#include <iostream>
#include <time.h>
#include <unordered_map>
#include "config.h"
#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/error-model.h"
#include "ns3/global-route-manager.h"
#include "ns3/internet-module.h"
#include "ns3/ipv4-static-routing-helper.h"
#include "ns3/packet.h"
#include "ns3/point-to-point-helper.h"
#include "ns3/qbb-helper.h"

#include <ns3/rdma-client-helper.h>
#include <ns3/rdma-client.h>
#include <ns3/rdma-driver.h>
#include <ns3/rdma.h>
#include <ns3/sim-setting.h>
#include <ns3/switch-node.h>
#include <ns3/nvswitch-node.h>
#include <atomic>

using namespace ns3;
using namespace std;

NS_LOG_COMPONENT_DEFINE("GENERIC_SIMULATION");
class ConfigBase;
extern std::unordered_map<std::string, std::unique_ptr<ConfigBase>> config_map;
extern std::unordered_map<std::string, std::unique_ptr<ConfigBase>> config_map_ns3; // 存储用户配置的ns3配置用于读取
EventId monitor_qlen_event;
EventId monitor_bw_event;
EventId monitor_qp_rate_event;
EventId monitor_qp_cnp_num_event;
double simulator_stop_time = 3.01;
std::string data_rate, link_delay, topology_file, flow_file, trace_file,
    trace_output_file;
std::string fct_output_file = "fct.txt";
std::string pfc_output_file = "pfc.txt";
std::string send_output_file = "send.txt";

double alpha_resume_interval = 55, rp_timer, ewma_gain = 1 / 16;
double rate_decrease_interval = 4;
uint32_t fast_recovery_times = 5;
std::string rate_ai, rate_hai, min_rate = "100Mb/s";
std::string dctcp_rate_ai = "1000Mb/s";

bool clamp_target_rate = false, l2_back_to_zero = false;
double error_rate_per_link = 0.0;
uint32_t has_win = 1;
uint32_t global_t = 1;
uint32_t mi_thresh = 5;
bool var_win = false, fast_react = true;
bool multi_rate = true;
bool sample_feedback = false;
double pint_log_base = 1.05;
double pint_prob = 1.0;
double u_target = 0.95;
uint32_t int_multi = 1;
bool rate_bound = true;
int nic_total_pause_time =
    0; 

uint32_t ack_high_prio = 0;
uint64_t link_down_time = 0;
uint32_t link_down_A = 0, link_down_B = 0;
vector<uint64_t> link_down; 
uint32_t enable_trace = 1;

uint32_t buffer_size = 16;
double sw_forward_delay = 0.0;

uint32_t node_num, switch_num, link_num, trace_num, nvswitch_num, gpus_per_server;
GPUType gpu_type;
std::vector<int>NVswitchs;

uint32_t qp_mon_interval = 100; 
uint32_t bw_mon_interval = 10000; 
uint32_t qlen_mon_interval = 10000; 
uint64_t mon_start = 0, mon_end = 2100000000;

string qlen_mon_file;
string bw_mon_file;
string rate_mon_file;
string cnp_mon_file;
string total_flow_file = "/root/astra-sim/extern/network_backend/ns3-interface/simulation/monitor_output/";
FILE* total_flow_output = nullptr;

unordered_map<uint64_t, uint32_t> rate2kmax, rate2kmin;
unordered_map<uint64_t, double> rate2pmax;

std::ifstream topof, flowf, tracef;

NodeContainer n;

uint64_t nic_rate;

uint64_t maxRtt, maxBdp;

std::vector<Ipv4Address> serverAddress;

std::unordered_map<uint32_t, unordered_map<uint32_t, uint16_t>> portNumber;

struct Interface {
  uint32_t idx;
  bool up;
  uint64_t delay;
  uint64_t bw;

  Interface() : idx(0), up(false) {}
};
map<Ptr<Node>, map<Ptr<Node>, Interface>> nbr2if;
map<Ptr<Node>, map<Ptr<Node>, vector<Ptr<Node>>>> nextHop;
map<Ptr<Node>, map<Ptr<Node>, uint64_t>> pairDelay;
map<Ptr<Node>, map<Ptr<Node>, uint64_t>> pairTxDelay;
map<uint32_t, map<uint32_t, uint64_t>> pairBw;
map<Ptr<Node>, map<Ptr<Node>, uint64_t>> pairBdp;
map<uint32_t, map<uint32_t, uint64_t>> pairRtt;

struct FlowInput {
  uint32_t src, dst, pg, maxPacketCount, port, dport;
  double start_time;
  uint32_t idx;
};

FlowInput flow_input = {0};
uint32_t flow_num;
Ipv4Address node_id_to_ip(uint32_t id) {
  return Ipv4Address(0x0b000001 + ((id / 256) * 0x00010000) +
                     ((id % 256) * 0x00000100));
}

uint32_t ip_to_node_id(Ipv4Address ip) { return (ip.Get() >> 8) & 0xffff; }

void get_pfc(FILE *fout, Ptr<QbbNetDevice> dev, uint32_t type) {
  fprintf(fout, "%lu %u %u %u %u\n", Simulator::Now().GetTimeStep(),
          dev->GetNode()->GetId(), dev->GetNode()->GetNodeType(),
          dev->GetIfIndex(), type);
}

struct QlenDistribution {
  vector<uint32_t>
      cnt;

  void add(uint32_t qlen) {
    uint32_t kb = qlen / 1000;
    if (cnt.size() < kb + 1)
      cnt.resize(kb + 1);
    cnt[kb]++;
  }
};

void monitor_qlen(FILE* qlen_output, NodeContainer *n){
	for (uint32_t i = 0; i < n->GetN(); i++){
		if(n->Get(i)->GetNodeType() == 1){ 
			Ptr<SwitchNode> sw = DynamicCast<SwitchNode>(n->Get(i));
			sw->PrintSwitchQlen(qlen_output);
		}else if(n->Get(i)->GetNodeType() == 2){ 
			Ptr<NVSwitchNode> sw = DynamicCast<NVSwitchNode>(n->Get(i));
			sw->PrintSwitchQlen(qlen_output);
		}
	}
	Simulator::Schedule(MicroSeconds(qlen_mon_interval), &monitor_qlen, qlen_output, n);
}
void monitor_bw(FILE* bw_output, NodeContainer *n){
	for (uint32_t i = 0; i < n->GetN(); i++){
		if(n->Get(i)->GetNodeType() == 1){ // is switch
			Ptr<SwitchNode> sw = DynamicCast<SwitchNode>(n->Get(i));
			sw->PrintSwitchBw(bw_output, bw_mon_interval);
		}else if(n->Get(i)->GetNodeType() == 2){ // is nvswitch
			Ptr<NVSwitchNode> sw = DynamicCast<NVSwitchNode>(n->Get(i));
			sw->PrintSwitchBw(bw_output, bw_mon_interval);
		}else{ // is host
			Ptr<Node> host = n->Get(i);
			host->GetObject<RdmaDriver>()->m_rdma->PrintHostBW(bw_output, bw_mon_interval);
		}
	}
	if (bw_mon_interval + Simulator::Now().GetMicroSeconds() < mon_end) {
    monitor_bw_event = Simulator::Schedule(MicroSeconds(bw_mon_interval), &monitor_bw, bw_output, n);
  }
}
void monitor_qp_rate(FILE* rate_output, NodeContainer *n){
	for(uint32_t i = 0; i < n->GetN(); i++){
		if(n->Get(i)->GetNodeType() == 0){ 
			Ptr<Node> host = n->Get(i);
			host->GetObject<RdmaDriver>()->m_rdma->PrintQPRate(rate_output);
		}
	}
	Simulator::Schedule(MicroSeconds(qp_mon_interval), &monitor_qp_rate, rate_output, n);
}
void monitor_qp_cnp_number(FILE* cnp_output, NodeContainer *n){
	for(uint32_t i = 0; i < n->GetN(); i++){
		if(n->Get(i)->GetNodeType() == 0){ 
			Ptr<Node> host = n->Get(i);
			host->GetObject<RdmaDriver>()->m_rdma->PrintQPCnpNumber(cnp_output);
		}
	}
	Simulator::Schedule(MicroSeconds(qp_mon_interval), &monitor_qp_cnp_number, cnp_output, n);
}
void schedule_monitor(){
	FILE* qlen_output = fopen(qlen_mon_file.c_str(), "w"); 
	assert(qlen_output != nullptr);
	fprintf(qlen_output, "%s, %s, %s, %s, %s, %s\n", "time", "sw_id", "port_id", "q_id", "q_len", "port_len");
	fflush(qlen_output);
	Simulator::Schedule(MicroSeconds(mon_start), &monitor_qlen, qlen_output, &n);
	FILE* bw_output = fopen(bw_mon_file.c_str(), "w");
	assert(bw_output != nullptr);
	fprintf(bw_output, "%s, %s, %s, %s\n", "time", "node_id", "port_id", "bandwidth");
	fflush(bw_output);
	Simulator::Schedule(MicroSeconds(mon_start), &monitor_bw, bw_output, &n);

	FILE* rate_output = fopen(rate_mon_file.c_str(), "w");
	assert(rate_output != nullptr);
	fprintf(rate_output, "%s, %s, %s, %s, %s, %s, %s\n", "time", "src", "dst", "sport", "dport", "size", "curr_rate");
	fflush(rate_output);
	Simulator::Schedule(MicroSeconds(mon_start), &monitor_qp_rate, rate_output, &n);

	FILE* cnp_output = fopen(cnp_mon_file.c_str(), "w");
	assert(cnp_output != nullptr);
	fprintf(cnp_output, "%s, %s, %s, %s, %s, %s, %s\n", "time", "src", "dst", "sport", "dport", "size", "cnp_number");
	fflush(cnp_output);
	Simulator::Schedule(MicroSeconds(mon_start), &monitor_qp_cnp_number, cnp_output, &n);
}

void CalculateRoute(Ptr<Node> host) {
  vector<Ptr<Node>> q;
  map<Ptr<Node>, int> dis;
  map<Ptr<Node>, uint64_t> delay;
  map<Ptr<Node>, uint64_t> txDelay;
  map<Ptr<Node>, uint64_t> bw;
  q.push_back(host);
  dis[host] = 0;
  delay[host] = 0;
  txDelay[host] = 0;
  bw[host] = 0xfffffffffffffffflu;
  for (int i = 0; i < (int)q.size(); i++) {
    Ptr<Node> now = q[i];
    int d = dis[now];
    for (auto it = nbr2if[now].begin(); it != nbr2if[now].end(); it++) {
      if (!it->second.up)
        continue;
      Ptr<Node> next = it->first;  
      if (dis.find(next) == dis.end()) {
        dis[next] = d + 1;
        delay[next] = delay[now] + it->second.delay;
        uint32_t payload = get_config_value_ns3<uint64_t>("ns3::RdmaHw::Mtu");
        txDelay[next] = txDelay[now] +
                        payload * 1000000000lu * 8 / it->second.bw;
        bw[next] = std::min(bw[now], it->second.bw);
        if (next->GetNodeType() == 1 || next->GetNodeType() == 2) {
          q.push_back(next);
        }
          
      }
      bool via_nvswitch = false;
      if (d + 1 == dis[next]) {
        for(auto x : nextHop[next][host]) {
          if(x->GetNodeType() == 2) via_nvswitch = true;
        }
        if(via_nvswitch == false) {
          if(now->GetNodeType() == 2) {
            while(nextHop[next][host].size() != 0) 
            nextHop[next][host].pop_back();
          }
          nextHop[next][host].push_back(now);
        } else if(via_nvswitch == true && now->GetNodeType() == 2) {
          nextHop[next][host].push_back(now);
        }
        if(next->GetNodeType() == 0 && nextHop[next][now].size() == 0) {
          nextHop[next][now].push_back(now);
          pairBw[next->GetId()][now->GetId()] = pairBw[now->GetId()][next->GetId()] = it->second.bw;
        }
      }
    }
  }
  for (auto it : delay) {
    pairDelay[it.first][host] = it.second;
  }
  for (auto it : txDelay)
    pairTxDelay[it.first][host] = it.second;
  for (auto it : bw) {
    pairBw[it.first->GetId()][host->GetId()] = it.second;
  }
}

void CalculateRoutes(NodeContainer &n) {
  for (int i = 0; i < (int)n.GetN(); i++) {
    Ptr<Node> node = n.Get(i);
    if (node->GetNodeType() == 0)
      CalculateRoute(node);
  }
}

void SetRoutingEntries() {
  for (auto i = nextHop.begin(); i != nextHop.end(); i++) {
    Ptr<Node> node = i->first;
    auto &table = i->second;
    for (auto j = table.begin(); j != table.end(); j++) {
      Ptr<Node> dst = j->first;
      Ipv4Address dstAddr = dst->GetObject<Ipv4>()->GetAddress(1, 0).GetLocal();
      vector<Ptr<Node>> nexts = j->second;
      for (int k = 0; k < (int)nexts.size(); k++) {
        Ptr<Node> next = nexts[k];
        uint32_t interface = nbr2if[node][next].idx;
        if (node->GetNodeType() == 1) {
          DynamicCast<SwitchNode>(node)->AddTableEntry(dstAddr, interface);
        } else if(node->GetNodeType() == 2){
					DynamicCast<NVSwitchNode>(node)->AddTableEntry(dstAddr, interface);
          node->GetObject<RdmaDriver>()->m_rdma->AddTableEntry(dstAddr, interface, true);
				} else {
          bool is_nvswitch = false;
					if(next->GetNodeType() == 2){ 
						is_nvswitch = true;
					}
					node->GetObject<RdmaDriver>()->m_rdma->AddTableEntry(dstAddr, interface, is_nvswitch);
          if(next->GetId() == dst->GetId())  {
            node->GetObject<RdmaDriver>()->m_rdma->add_nvswitch(dst->GetId());
          }
        }
      }
    }
  }
}

void printRoutingEntries() {
  map<uint32_t, string> types;
  types[0] = "HOST";
  types[1] = "SWITCH";
  types[2] = "NVSWITCH";
  map<Ptr<Node>, map<Ptr<Node>, vector<pair<Ptr<Node>, uint32_t> >>> NVSwitch, NetSwitch, Host; 
  for (auto i = nextHop.begin(); i != nextHop.end(); i++) {
    Ptr<Node> src = i -> first;
    auto &table = i->second;
    for (auto j = table.begin(); j != table.end(); j++) { 
      Ptr<Node> dst = j->first;
      Ipv4Address dstAddr = dst->GetObject<Ipv4>()->GetAddress(1, 0).GetLocal();
      vector<Ptr<Node>> nexts = j->second;
      for (int k = 0; k < (int)nexts.size(); k++) {
        Ptr<Node> firstHop = nexts[k];
        uint32_t interface = nbr2if[src][firstHop].idx;
        if(src->GetNodeType() == 0) {
          Host[src][dst].push_back(pair<Ptr<Node>, uint32_t>(firstHop, interface));
        } else if(src->GetNodeType() == 1) {
          NetSwitch[src][dst].push_back(pair<Ptr<Node>, uint32_t>(firstHop, interface));
        } else if(src->GetNodeType() == 2) {
          NVSwitch[src][dst].push_back(pair<Ptr<Node>, uint32_t>(firstHop, interface));
        }
      }
    }
  }

  cout << "*********************    PRINT SWITCH ROUTING TABLE    *********************" << endl << endl << endl;
  for(auto it = NetSwitch.begin(); it != NetSwitch.end(); ++ it) {
    Ptr<Node> src = it -> first;
    auto table = it -> second;
    cout << "SWITCH: " << src->GetId() << "'s routing entries are as follows:" << endl;
    for(auto j = table.begin(); j != table.end(); ++ j) {
      Ptr<Node> dst = j -> first;
      auto entries = j -> second;
      for(auto k = entries.begin(); k != entries.end(); ++ k) {
        Ptr<Node> nextHop = k->first;
        uint32_t interface = k->second;
        cout << "To " << dst->GetId() << "[" << types[dst->GetNodeType()] << "] via " << nextHop->GetId() << "[" << types[nextHop->GetNodeType()] << "]" << " from port: " << interface << endl;
      }
    }
  } 

  cout << "*********************    PRINT NVSWITCH ROUTING TABLE    *********************" << endl  << endl << endl;
  for(auto it = NVSwitch.begin(); it != NVSwitch.end(); ++ it) {
    Ptr<Node> src = it -> first;
    auto table = it -> second;
    cout << "NVSWITCH: " << src->GetId() << "'s routing entries are as follows:" << endl;
    for(auto j = table.begin(); j != table.end(); ++ j) {
      Ptr<Node> dst = j -> first;
      auto entries = j -> second;
      for(auto k = entries.begin(); k != entries.end(); ++ k) {
        Ptr<Node> nextHop = k->first;
        uint32_t interface = k->second;
        cout << "To " << dst->GetId() << "[" << types[dst->GetNodeType()] << "] via " << nextHop->GetId() << "[" << types[nextHop->GetNodeType()] << "]" << " from port: " << interface << endl;
      }
    }
  } 

  cout << "*********************    HOST ROUTING TABLE    *********************" << endl << endl << endl;
  for(auto it = Host.begin(); it != Host.end(); ++ it) {
    Ptr<Node> src = it -> first;
    auto table = it -> second;
    cout << "HOST: " << src->GetId() << "'s routing entries are as follows:" << endl;
    for(auto j = table.begin(); j != table.end(); ++ j) {
      Ptr<Node> dst = j -> first;
      auto entries = j -> second;
      for(auto k = entries.begin(); k != entries.end(); ++ k) {
        Ptr<Node> nextHop = k->first;
        uint32_t interface = k->second;
        cout << "To " << dst->GetId() << "[" << types[dst->GetNodeType()] << "] via " << nextHop->GetId() << "[" << types[nextHop->GetNodeType()] << "]" << " from port: " << interface << endl;
      }
    }
  } 

}

bool validateRoutingEntries() {
  return false;
}

void TakeDownLink(NodeContainer n, Ptr<Node> a, Ptr<Node> b) {
  if (!nbr2if[a][b].up)
    return;
  nbr2if[a][b].up = nbr2if[b][a].up = false;
  nextHop.clear();
  CalculateRoutes(n);
	for (uint32_t i = 0; i < n.GetN(); i++){
		if (n.Get(i)->GetNodeType() == 1)
			DynamicCast<SwitchNode>(n.Get(i))->ClearTable();
		else if(n.Get(i)->GetNodeType() == 2)
			DynamicCast<NVSwitchNode>(n.Get(i))->ClearTable();
		else
			n.Get(i)->GetObject<RdmaDriver>()->m_rdma->ClearTable();
	}
  DynamicCast<QbbNetDevice>(a->GetDevice(nbr2if[a][b].idx))->TakeDown();
  DynamicCast<QbbNetDevice>(b->GetDevice(nbr2if[b][a].idx))->TakeDown();
  SetRoutingEntries();

  for (uint32_t i = 0; i < n.GetN(); i++) {
    if (n.Get(i)->GetNodeType() == 0)
      n.Get(i)->GetObject<RdmaDriver>()->m_rdma->RedistributeQp();
  }
}

string get_output_file_name(string config_file, string output_file){
	auto idx = config_file.find_last_of('/');
	string ans = output_file.substr(0, output_file.length()-4) + config_file.substr(idx+7);
	return ans;
}

uint64_t get_nic_rate(NodeContainer &n) {
  for (uint32_t i = 0; i < n.GetN(); i++)
    if (n.Get(i)->GetNodeType() == 0)
      return DynamicCast<QbbNetDevice>(n.Get(i)->GetDevice(1))
          ->GetDataRate()
          .GetBitRate();
}
void InitConfigMap() {
  /************************************************
   * Value Config
   ***********************************************/
  config_map["SIMULATOR_STOP_TIME"] =
      std::make_unique<ConfigVar<double>>(simulator_stop_time);
  config_map["DATA_RATE"] = std::make_unique<ConfigVar<std::string>>(data_rate);
  config_map["LINK_DELAY"] =
      std::make_unique<ConfigVar<std::string>>(link_delay);
  config_map["TOPOLOGY_FILE"] =
      std::make_unique<ConfigVar<std::string>>(topology_file);
  config_map["FLOW_FILE"] = std::make_unique<ConfigVar<std::string>>(flow_file);
  config_map["TRACE_FILE"] =
      std::make_unique<ConfigVar<std::string>>(trace_file);
  config_map["TRACE_OUTPUT_FILE"] =
      std::make_unique<ConfigVar<std::string>>(trace_output_file);
  config_map["FCT_OUTPUT_FILE"] =
      std::make_unique<ConfigVar<std::string>>(fct_output_file);
  config_map["PFC_OUTPUT_FILE"] =
      std::make_unique<ConfigVar<std::string>>(pfc_output_file);
  config_map["SEND_OUTPUT_FILE"] =
      std::make_unique<ConfigVar<std::string>>(send_output_file);
  config_map["ERROR_RATE_PER_LINK"] =
      std::make_unique<ConfigVar<double>>(error_rate_per_link);
  config_map["HAS_WIN"] = std::make_unique<ConfigVar<uint32_t>>(has_win);
  config_map["GLOBAL_T"] = std::make_unique<ConfigVar<uint32_t>>(global_t);
  config_map["PINT_LOG_BASE"] =
      std::make_unique<ConfigVar<double>>(pint_log_base);
  config_map["INT_MULTI"] = std::make_unique<ConfigVar<uint32_t>>(int_multi);
  config_map["ACK_HIGH_PRIO"] =
      std::make_unique<ConfigVar<uint32_t>>(ack_high_prio);
  config_map["LINK_DOWN"] =
      std::make_unique<ConfigVar<vector<uint64_t>>>(link_down);
  config_map["ENABLE_TRACE"] =
      std::make_unique<ConfigVar<uint32_t>>(enable_trace);
  config_map["BUFFER_SIZE"] =
      std::make_unique<ConfigVar<uint32_t>>(buffer_size);
  config_map["SWITCH_FORWARD_DELAY"] =
      std::make_unique<ConfigVar<double>>(sw_forward_delay);
  config_map["QP_MON_INTERVAL"] =
      std::make_unique<ConfigVar<uint32_t>>(qp_mon_interval);
  config_map["QLEN_MON_INTERVAL"] =
      std::make_unique<ConfigVar<uint32_t>>(qlen_mon_interval);
  config_map["BW_MON_INTERVAL"] = 
      std::make_unique<ConfigVar<uint32_t>>(bw_mon_interval);
  config_map["MON_START"] = std::make_unique<ConfigVar<uint64_t>>(mon_start);
  config_map["MON_END"] = std::make_unique<ConfigVar<uint64_t>>(mon_end);
  config_map["QLEN_MON_FILE"] =
      std::make_unique<ConfigVar<string>>(qlen_mon_file);
  config_map["BW_MON_FILE"] = std::make_unique<ConfigVar<string>>(bw_mon_file);
  // config_map["TX_BW_MON_FILE"] = std::make_unique<ConfigVar<string>>(tx_bw_mon_file);
  // config_map["RX_BW_MON_FILE"] = std::make_unique<ConfigVar<string>>(rx_bw_mon_file);
  config_map["RATE_MON_FILE"] =
      std::make_unique<ConfigVar<string>>(rate_mon_file);
  config_map["CNP_MON_FILE"] =
      std::make_unique<ConfigVar<string>>(cnp_mon_file);
  config_map["KMAX_MAP"] =
      std::make_unique<ConfigVar<unordered_map<uint64_t, uint32_t>>>(rate2kmax);
  config_map["KMIN_MAP"] =
      std::make_unique<ConfigVar<unordered_map<uint64_t, uint32_t>>>(rate2kmin);
  config_map["PMAX_MAP"] =
      std::make_unique<ConfigVar<unordered_map<uint64_t, double>>>(rate2pmax);

  /************************************************
   * NS3 Config
   ***********************************************/
  // QbbNetDevice
  config_map["ENABLE_QCN"] =
      std::make_unique<ConfigNs3<bool>>("ns3::QbbNetDevice::QcnEnabled", true);
  config_map["USE_DYNAMIC_PFC_THRESHOLD"] = std::make_unique<ConfigNs3<bool>>(
      "ns3::QbbNetDevice::DynamicThreshold", true);
  config_map["PAUSE_TIME"] =
      std::make_unique<ConfigNs3<uint32_t>>("ns3::QbbNetDevice::PauseTime", 5);
  // RdmaHw
  config_map["CC_MODE"] =
      std::make_unique<ConfigNs3<uint32_t>>("ns3::RdmaHw::CcMode", 1);
  config_map["PACKET_PAYLOAD_SIZE"] =
      std::make_unique<ConfigNs3<uint32_t>>("ns3::RdmaHw::Mtu", 1000);
  config_map["L2_CHUNK_SIZE"] =
      std::make_unique<ConfigNs3<uint32_t>>("ns3::RdmaHw::L2ChunkSize", 0);
  config_map["L2_ACK_INTERVAL"] =
      std::make_unique<ConfigNs3<uint32_t>>("ns3::RdmaHw::L2AckInterval", 0);
  config_map["L2_BACK_TO_ZERO"] =
      std::make_unique<ConfigNs3<bool>>("ns3::RdmaHw::L2BackToZero", false);
  config_map["RATE_AI"] =
      std::make_unique<ConfigNs3<std::string>>("ns3::RdmaHw::RateAI");
  config_map["RATE_HAI"] =
      std::make_unique<ConfigNs3<std::string>>("ns3::RdmaHw::RateHAI");
  config_map["MIN_RATE"] = std::make_unique<ConfigNs3<std::string>>(
      "ns3::RdmaHw::MinRate", "100Mb/s");
  config_map["VAR_WIN"] =
      std::make_unique<ConfigNs3<bool>>("ns3::RdmaHw::VarWin", false);
  config_map["RATE_BOUND"] =
      std::make_unique<ConfigNs3<bool>>("ns3::RdmaHw::RateBound", true);
  config_map["NIC_TOTAL_PAUSE_TIME"] = std::make_unique<ConfigNs3<uint32_t>>(
      "ns3::RdmaHw::TotalPauseTime",
      0); // slightly less than finish time without inefficiency in us
  // MellanoxDcqcn
  config_map["ALPHA_RESUME_INTERVAL"] = std::make_unique<ConfigNs3<double>>(
      "ns3::MellanoxDcqcn::AlphaResumInterval", 55.0);
  config_map["RP_TIMER"] =
      std::make_unique<ConfigNs3<double>>("ns3::MellanoxDcqcn::RPTimer", 0.01);
  config_map["EWMA_GAIN"] = std::make_unique<ConfigNs3<double>>(
      "ns3::MellanoxDcqcn::EwmaGain", 1 / 16);
  config_map["RATE_DECREASE_INTERVAL"] = std::make_unique<ConfigNs3<double>>(
      "ns3::MellanoxDcqcn::RateDecreaseInterval", 4);
  config_map["FAST_RECOVERY_TIMES"] = std::make_unique<ConfigNs3<uint32_t>>(
      "ns3::MellanoxDcqcn::FastRecoveryTimes", 5);
  config_map["CLAMP_TARGET_RATE"] = std::make_unique<ConfigNs3<bool>>(
      "ns3::MellanoxDcqcn::ClampTargetRate", false);
  // Dctcp
  config_map["DCTCP_RATE_AI"] = std::make_unique<ConfigNs3<std::string>>(
      "ns3::Dctcp::DctcpRateAI", "1000Mb/s");
  // Hpcc
  config_map["MI_THRESH"] =
      std::make_unique<ConfigNs3<uint32_t>>("ns3::Hpcc::MiThresh", 5);
  config_map["FAST_REACT"] =
      std::make_unique<ConfigNs3<bool>>("ns3::Hpcc::FastReact", true);
  config_map["MULTI_RATE"] =
      std::make_unique<ConfigNs3<bool>>("ns3::Hpcc::MultiRate", true);
  config_map["SAMPLE_FEEDBACK"] =
      std::make_unique<ConfigNs3<bool>>("ns3::Hpcc::SampleFeedback", false);
  config_map["U_TARGET"] =
      std::make_unique<ConfigNs3<double>>("ns3::Hpcc::TargetUtil", 0.95);
  // HpccPint
  config_map["PINT_PROB"] =
      std::make_unique<ConfigNs3<double>>("ns3::HpccPint::PintProb", 1.0);
}
bool ReadConf(string network_topo,string network_conf) {

    std::ifstream conf;
    conf.open(network_conf);
    InitConfigMap();
    string line;
    topology_file = network_topo;
    while (!conf.eof()) {
      std::getline(conf, line);
      if(line == "" || line == "\n"){
        continue;
      }
      if(line[0] == '#'){ // comment. Now only support comment at the beginning
        continue;
      }
      std::istringstream iss(line);
      std::string key;
      if (!(iss >> key)) {
        continue;
      }
      std::string value;
      std::getline(iss, value);
      if (!value.empty() && value.front() == ' ') {
        value.erase(0, 1);
      }
      // 去除value结尾的\r
      if (value.back() == '\r') {
        value.pop_back();
      }
      if(config_map.find(key) != config_map.end()){
        config_map[key]->set_value(value);
      }else if(key.substr(0, 5) == "ns3::"){
        // Try to set ns3 config directly
        Ns3ConfigMethods::ParseAndSetConfigDefault(key, value);
      }else {
        std::cout << "Error: key not found: " << key << std::endl;
      }
      fflush(stdout);
    }
    std::cout << "read config done!" << std::endl;
    conf.close();
    return true;
}

void SetConfig() {
  IntHop::multi = int_multi;
  uint32_t cc_mode = get_config_value_ns3<uint64_t>("ns3::RdmaHw::CcMode");
  if (cc_mode == 7) 
    IntHeader::mode = IntHeader::TS;
  else if (cc_mode == 3) 
    IntHeader::mode = IntHeader::NORMAL;
  else if (cc_mode == 10) 
    IntHeader::mode = IntHeader::PINT;
  else
    IntHeader::mode = IntHeader::NONE;

  if (cc_mode == 10) {
    Pint::set_log_base(pint_log_base);
    IntHeader::pint_bytes = Pint::get_n_bytes();
    printf("PINT bits: %d bytes: %d\n", Pint::get_n_bits(),
           Pint::get_n_bytes());
  }
}

void SetupNetwork(
    void (*qp_finish)(FILE *, Ptr<RdmaQueuePair>),
    void (*message_finish)(FILE*, Ptr<RdmaQueuePair>, uint64_t, uint64_t),
    void (*send_finish)(FILE *, Ptr<RdmaQueuePair>, uint64_t, uint64_t)
    ) {

  topof.open(topology_file.c_str());
  flowf.open(flow_file.c_str());
  tracef.open(trace_file.c_str());
  string gpu_type_str;

  topof >> node_num >> gpus_per_server >> nvswitch_num >> switch_num >>
      link_num >> gpu_type_str;
  flowf >> flow_num;
  tracef >> trace_num;
  if(gpu_type_str == "A100"){
    gpu_type = GPUType::A100;
  } else if(gpu_type_str == "A800"){
    gpu_type = GPUType::A800;
  } else if(gpu_type_str == "H100"){
    gpu_type = GPUType::H100;
  } else if(gpu_type_str == "H800"){
    gpu_type = GPUType::H800;
  } else{
    gpu_type = GPUType::NONE;
  }

  std::vector<uint32_t> node_type(node_num, 0);
  for (uint32_t i = 0; i < nvswitch_num; i++) {
    uint32_t sid;
    topof >> sid;
    node_type[sid] = 2;
	}
	for (uint32_t i = 0; i < switch_num; i++)
	{
		uint32_t sid;
		topof >> sid;
		node_type[sid] = 1;
	}
	for (uint32_t i = 0; i < node_num; i++){
		if (node_type[i] == 0)
			n.Add(CreateObject<Node>());
		else if(node_type[i] == 1){
			Ptr<SwitchNode> sw = CreateObject<SwitchNode>();
			n.Add(sw);
			sw->SetAttribute("EcnEnabled", BooleanValue(get_config_value_ns3<bool>("ns3::QbbNetDevice::QcnEnabled")));
      sw->SetAttribute("ForwardDelay", DoubleValue(sw_forward_delay));
		}else if(node_type[i] == 2){
			Ptr<NVSwitchNode> sw = CreateObject<NVSwitchNode>();
			n.Add(sw);
		}
	}

  NS_LOG_INFO("Create nodes.");
  InternetStackHelper internet;
  internet.Install(n);

  for (uint32_t i = 0; i < node_num; i++) {
    if (n.Get(i)->GetNodeType() == 0) {
      serverAddress.resize(i + 1);
      serverAddress[i] = node_id_to_ip(i);
    } else if(n.Get(i)->GetNodeType() == 2) {
      serverAddress.resize(i + 1);
      serverAddress[i] = node_id_to_ip(i);
    }
  }

  NS_LOG_INFO("Create channels.");

  Ptr<RateErrorModel> rem = CreateObject<RateErrorModel>();
  Ptr<UniformRandomVariable> uv = CreateObject<UniformRandomVariable>();
  rem->SetRandomVariable(uv);
  uv->SetStream(50);
  rem->SetAttribute("ErrorRate", DoubleValue(error_rate_per_link));
  rem->SetAttribute("ErrorUnit", StringValue("ERROR_UNIT_PACKET"));

  FILE *pfc_file = fopen(pfc_output_file.c_str(), "w");

  QbbHelper qbb;
  Ipv4AddressHelper ipv4;
  for (uint32_t i = 0; i < link_num; i++) {
    uint32_t src, dst;
    std::string data_rate, link_delay;
    double error_rate;
    topof >> src >> dst >> data_rate >> link_delay >> error_rate;
    Ptr<Node> snode = n.Get(src), dnode = n.Get(dst);
    
    qbb.SetDeviceAttribute("DataRate", StringValue(data_rate));
    qbb.SetChannelAttribute("Delay", StringValue(link_delay));

    if (error_rate > 0) {
      Ptr<RateErrorModel> rem = CreateObject<RateErrorModel>();
      Ptr<UniformRandomVariable> uv = CreateObject<UniformRandomVariable>();
      rem->SetRandomVariable(uv);
      uv->SetStream(50);
      rem->SetAttribute("ErrorRate", DoubleValue(error_rate));
      rem->SetAttribute("ErrorUnit", StringValue("ERROR_UNIT_PACKET"));
      qbb.SetDeviceAttribute("ReceiveErrorModel", PointerValue(rem));
    } else {
      qbb.SetDeviceAttribute("ReceiveErrorModel", PointerValue(rem));
    }

    fflush(stdout);

    NetDeviceContainer d = qbb.Install(snode, dnode);
    if (snode->GetNodeType() == 0 || snode->GetNodeType() == 2) {
      Ptr<Ipv4> ipv4 = snode->GetObject<Ipv4>();
      ipv4->AddInterface(d.Get(0));
      ipv4->AddAddress(
          1, Ipv4InterfaceAddress(serverAddress[src], Ipv4Mask(0xff000000)));
    }
    if (dnode->GetNodeType() == 0 || dnode->GetNodeType() == 2) {
      Ptr<Ipv4> ipv4 = dnode->GetObject<Ipv4>();
      ipv4->AddInterface(d.Get(1));
      ipv4->AddAddress(
          1, Ipv4InterfaceAddress(serverAddress[dst], Ipv4Mask(0xff000000)));
    }

    nbr2if[snode][dnode].idx =
        DynamicCast<QbbNetDevice>(d.Get(0))->GetIfIndex();
    nbr2if[snode][dnode].up = true;
    nbr2if[snode][dnode].delay =
        DynamicCast<QbbChannel>(
            DynamicCast<QbbNetDevice>(d.Get(0))->GetChannel())
            ->GetDelay()
            .GetTimeStep();
    nbr2if[snode][dnode].bw =
        DynamicCast<QbbNetDevice>(d.Get(0))->GetDataRate().GetBitRate();
    nbr2if[dnode][snode].idx =
        DynamicCast<QbbNetDevice>(d.Get(1))->GetIfIndex();
    nbr2if[dnode][snode].up = true;
    nbr2if[dnode][snode].delay =
        DynamicCast<QbbChannel>(
            DynamicCast<QbbNetDevice>(d.Get(1))->GetChannel())
            ->GetDelay()
            .GetTimeStep();
    nbr2if[dnode][snode].bw =
        DynamicCast<QbbNetDevice>(d.Get(1))->GetDataRate().GetBitRate();

    char ipstring[16];
    sprintf(ipstring, "10.%d.%d.0", i / 254 + 1, i % 254 + 1);
    ipv4.SetBase(ipstring, "255.255.255.0");
    ipv4.Assign(d);

    DynamicCast<QbbNetDevice>(d.Get(0))->TraceConnectWithoutContext(
        "QbbPfc", MakeBoundCallback(&get_pfc, pfc_file,
                                    DynamicCast<QbbNetDevice>(d.Get(0))));
    DynamicCast<QbbNetDevice>(d.Get(1))->TraceConnectWithoutContext(
        "QbbPfc", MakeBoundCallback(&get_pfc, pfc_file,
                                    DynamicCast<QbbNetDevice>(d.Get(1))));
  }

  nic_rate = get_nic_rate(n);
  uint32_t payload = get_config_value_ns3<uint64_t>("ns3::RdmaHw::Mtu");
  for (uint32_t i = 0; i < node_num; i++) {
    if (n.Get(i)->GetNodeType() == 1) { 
      Ptr<SwitchNode> sw = DynamicCast<SwitchNode>(n.Get(i));
      uint32_t shift = 3; 

      for (uint32_t j = 1; j < sw->GetNDevices(); j++) {
        Ptr<QbbNetDevice> dev = DynamicCast<QbbNetDevice>(sw->GetDevice(j));
        uint64_t rate = dev->GetDataRate().GetBitRate();
        NS_ASSERT_MSG(rate2kmin.find(rate) != rate2kmin.end(),
                      "must set kmin for each link speed");
        NS_ASSERT_MSG(rate2kmax.find(rate) != rate2kmax.end(),
                      "must set kmax for each link speed");
        NS_ASSERT_MSG(rate2pmax.find(rate) != rate2pmax.end(),
                      "must set pmax for each link speed");
        sw->m_mmu->ConfigEcn(j, rate2kmin[rate], rate2kmax[rate],
                             rate2pmax[rate]);
        uint64_t delay = DynamicCast<QbbChannel>(dev->GetChannel())
                             ->GetDelay()
                             .GetTimeStep();
        uint32_t headroom = rate * delay / 8 / 1000000000 * 3  + payload * 2;  // BDP + 2 packet
        sw->m_mmu->ConfigHdrm(j, headroom);
        sw->m_mmu->pfc_a_shift[j] = shift;
        while (rate > nic_rate && sw->m_mmu->pfc_a_shift[j] > 0) {
          sw->m_mmu->pfc_a_shift[j]--;
          rate /= 2;
        }
      }
      sw->m_mmu->ConfigNPort(sw->GetNDevices() - 1);
      sw->m_mmu->ConfigBufferSize(buffer_size * 1024 * 1024);
      sw->m_mmu->node_id = sw->GetId();
    } else if(n.Get(i)->GetNodeType() == 2){ 
			Ptr<NVSwitchNode> sw = DynamicCast<NVSwitchNode>(n.Get(i));
      uint32_t shift = 3;   //by default 1/
      for (uint32_t j = 1; j < sw->GetNDevices(); j++) {
        Ptr<QbbNetDevice> dev = DynamicCast<QbbNetDevice>(sw->GetDevice(j));
        uint64_t rate = dev->GetDataRate().GetBitRate();
        uint64_t delay = DynamicCast<QbbChannel>(dev->GetChannel())
                             ->GetDelay()
                             .GetTimeStep();
        uint32_t headroom = rate * delay / 8 / 1000000000 * 3;
        sw->m_mmu->ConfigHdrm(j, headroom);
        sw->m_mmu->pfc_a_shift[j] = shift;
        while (rate > nic_rate && sw->m_mmu->pfc_a_shift[j] > 0) {
          sw->m_mmu->pfc_a_shift[j]--;
          rate /= 2;
        }
      }
			sw->m_mmu->ConfigNPort(sw->GetNDevices()-1);
			sw->m_mmu->ConfigBufferSize(buffer_size* 1024 * 1024);
			sw->m_mmu->node_id = sw->GetId();
		}
  }

#if ENABLE_QP
  FILE *fct_output = fopen(fct_output_file.c_str(), "w");
  FILE *send_output = fopen(send_output_file.c_str(), "w");
  for (uint32_t i = 0; i < node_num; i++) {
    if (n.Get(i)->GetNodeType() == 0 || n.Get(i)->GetNodeType() == 2) { 
      // create RdmaHw
      Ptr<RdmaHw> rdmaHw = CreateObject<RdmaHw>();
      // check if i is in rdmaHw_config_map, if so, set the group attribute
      if (rdmaHw_config_map.find(i) != rdmaHw_config_map.end()) {
        for(auto configEntry : *(rdmaHw_config_map[i])){
          std::cout<<"node "<<i<<" set attribute "<<configEntry.first << std::endl;
          if(configEntry.first.compare(0, 5, "CC::") == 0){
            rdmaHw->m_cc_configs.push_back(std::make_pair(configEntry.first.substr(5), configEntry.second));
          }
          rdmaHw->SetAttribute(configEntry.first, *(configEntry.second));
        }
      }
      Ptr<RdmaDriver> rdma = CreateObject<RdmaDriver>();
      Ptr<Node> node = n.Get(i);
      rdma->SetNode(node);
      rdma->SetRdmaHw(rdmaHw);

      node->AggregateObject(rdma);
      rdma->Init();
      rdma->TraceConnectWithoutContext(
          "QpComplete", MakeBoundCallback(qp_finish, fct_output));
      rdma->TraceConnectWithoutContext(
          "MessageComplete", MakeBoundCallback(message_finish, fct_output));
      rdma->TraceConnectWithoutContext("SendComplete",MakeBoundCallback(send_finish,send_output));
    }
  }
#endif

  if (ack_high_prio)
    RdmaEgressQueue::ack_q_idx = 0;
  else
    RdmaEgressQueue::ack_q_idx = 3;

  CalculateRoutes(n);
  SetRoutingEntries();

  maxRtt = maxBdp = 0;
  for (uint32_t i = 0; i < node_num; i++) {
    if (n.Get(i)->GetNodeType() != 0)
      continue;
    for (uint32_t j = 0; j < node_num; j++) {
      if (n.Get(j)->GetNodeType() != 0)
        continue;
      uint64_t delay = pairDelay[n.Get(i)][n.Get(j)];
      uint64_t txDelay = pairTxDelay[n.Get(i)][n.Get(j)];
      uint64_t rtt = delay * 2 + txDelay;
      uint64_t bw = pairBw[i][j];
      uint64_t bdp = rtt * bw / 1000000000 / 8;
      pairBdp[n.Get(i)][n.Get(j)] = bdp;
      pairRtt[i][j] = rtt;
      if (bdp > maxBdp)
        maxBdp = bdp;
      if (rtt > maxRtt)
        maxRtt = rtt;
    }
  }
  printf("maxRtt=%lu maxBdp=%lu\n", maxRtt, maxBdp);

  for (uint32_t i = 0; i < node_num; i++) {
    if (n.Get(i)->GetNodeType() == 1) { 
      Ptr<SwitchNode> sw = DynamicCast<SwitchNode>(n.Get(i));
      sw->SetAttribute("CcMode", UintegerValue(get_config_value_ns3<uint64_t>("ns3::RdmaHw::CcMode")));
      sw->SetAttribute("MaxRtt", UintegerValue(maxRtt));
    }
  }

  NodeContainer trace_nodes;
  for (uint32_t i = 0; i < trace_num; i++) {
    uint32_t nid;
    tracef >> nid;
    if (nid >= n.GetN()) {
      continue;
    }
    trace_nodes = NodeContainer(trace_nodes, n.Get(nid));
  }

  FILE *trace_output = fopen(trace_output_file.c_str(), "w");
  if (enable_trace){
    qbb.EnableTracing(trace_output, trace_nodes);

  
    SimSetting sim_setting;
    for (auto i : nbr2if) {
      for (auto j : i.second) {
        uint16_t node = i.first->GetId();
        uint8_t intf = j.second.idx;
        uint64_t bps =
            DynamicCast<QbbNetDevice>(i.first->GetDevice(j.second.idx))
                ->GetDataRate()
                .GetBitRate();
        sim_setting.port_speed[node][intf] = bps;
      }
    }
    sim_setting.win = maxBdp;
    sim_setting.Serialize(trace_output);
  }

  NS_LOG_INFO("Create Applications.");

  Time interPacketInterval = Seconds(0.0000005 / 2);
  for (uint32_t i = 0; i < node_num; i++) {
    if (n.Get(i)->GetNodeType() == 0 || n.Get(i)->GetNodeType() == 2)
      for (uint32_t j = 0; j < node_num; j++) {
        if (n.Get(j)->GetNodeType() == 0 || n.Get(j)->GetNodeType() == 2)
          portNumber[i][j] = 10000; 
      }
  }
  flow_input.idx = -1;

  topof.close();
  tracef.close();

  if (link_down[0] > 0) {
    Simulator::Schedule(Seconds(2) + MicroSeconds(link_down[0]),
                        &TakeDownLink, n, n.Get(link_down[1]),
                        n.Get(link_down[2]));        
  }
}
#endif
