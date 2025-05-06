#ifndef __CONFIG_H__
#define __CONFIG_H__
#include <time.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <variant>
#include <vector>

#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/error-model.h"
#include "ns3/global-route-manager.h"
#include "ns3/internet-module.h"
#include "ns3/ipv4-static-routing-helper.h"
#include "ns3/packet.h"
#include "ns3/point-to-point-helper.h"
#include "ns3/qbb-helper.h"

#include <ns3/nvswitch-node.h>
#include <ns3/rdma-client-helper.h>
#include <ns3/rdma-client.h>
#include <ns3/rdma-driver.h>
#include <ns3/rdma.h>
#include <ns3/sim-setting.h>
#include <ns3/switch-node.h>
#include <atomic>

using namespace ns3;

// 基类
class ConfigBase {
 public:
  virtual void set_value(const std::string& value) = 0;

  virtual ~ConfigBase() = default;
};

// 基础类型的变量模板类
template <typename T>
class ConfigVar : public ConfigBase {
 public:
  ConfigVar(T& var) : var_(var) {}

  void set_value(const std::string& value) override {
    std::istringstream iss(value);
    iss >> var_; // 使用流操作符进行不同类型的转换
  }

 private:
  T& var_;
};

// 数组类型的变量模板类
template <typename T>
class ConfigVar<std::vector<T>> : public ConfigBase {
 public:
  ConfigVar(std::vector<T>& var) : var_(var) {}

  void set_value(const std::string& value) override {
    var_.clear();
    std::istringstream iss(value);
    T element;
    while (iss >> element) {
      // std::cout << "Add element: " << element << std::endl;
      var_.push_back(element);
    }
  }

 private:
  std::vector<T>& var_;
};

// 哈希表类型的变量模板类
// 形如 "n_k(int) K V K V K V...(n_k对)"
template <typename K, typename V>
class ConfigVar<std::unordered_map<K, V>> : public ConfigBase {
 public:
  ConfigVar(std::unordered_map<K, V>& var) : var_(var) {}
  void set_value(const std::string& value) override {
    var_.clear();
    std::istringstream iss(value);
    int n_k;
    iss >> n_k;
    for (int i = 0; i < n_k; i++) {
      K key;
      V value;
      iss >> key >> value;
      // std::cout << "Add element: " << key << ", " << value << std::endl;
      var_[key] = value;
    }
  }

 private:
  std::unordered_map<K, V>& var_;
};

// 存储基础类型的变量模板类
template <typename T>
class StoreVar : public ConfigBase {
 public:
  StoreVar(T var) : var_(var) {}

  void set_value(const std::string& value) override {
    std::istringstream iss(value);
    iss >> var_; // 使用流操作符进行不同类型的转换
  }

  T get_var() {
    return var_;
  }

 private:
  T var_; //ConfigVar将变量存储在外部，StoreVar将变量存储在内部
};

static std::unordered_map<std::string, std::unique_ptr<ConfigBase>> config_map_ns3; // 存储用户配置的ns3配置用于读取

template <typename T>
T get_config_value_ns3(const std::string& name) {
  return dynamic_cast<StoreVar<T>*>(config_map_ns3[name].get())->get_var();
}

class Ns3ConfigMethods {
 public:
  static void SetConfigDefault(const std::string& name, const std::string& value) {
    Config::SetDefault(name, StringValue(value));
    config_map_ns3[name] = std::make_unique<StoreVar<std::string>>(value);
  }
  static void SetConfigDefault(const std::string& name, uint64_t value) {
    Config::SetDefault(name, UintegerValue(value));
    config_map_ns3[name] = std::make_unique<StoreVar<uint64_t>>(value);
  }
  static void SetConfigDefault(const std::string& name, uint32_t value) {
    Config::SetDefault(name, UintegerValue(value));
    config_map_ns3[name] = std::make_unique<StoreVar<uint64_t>>(value);
  }
  static void SetConfigDefault(const std::string& name, double value) {
    Config::SetDefault(name, DoubleValue(value));
    config_map_ns3[name] = std::make_unique<StoreVar<double>>(value);
  }
  static void SetConfigDefault(const std::string& name, bool value) {
    Config::SetDefault(name, BooleanValue(value));
    config_map_ns3[name] = std::make_unique<StoreVar<bool>>(value);
  }

  static bool isBooleanValue(const std::string& str, bool& value) {
    if (str == "true") {
      value = true;
      return true;
    } else if (str == "false") {
      value = false;
      return true;
    }
    return false;
  }

  static bool isUintegerValue(const std::string& str, uint64_t& value) {
    for(auto c : str){
      if(!isdigit(c)){
        return false;
      }
    }
    value = std::stoll(str);
    return true;
  }

  static bool isDoubleValue(const std::string& str, double& value) {
    if(str.find_first_not_of("0123456789.-\r") != std::string::npos){
      return false;
    }
    value = std::stod(str);
    return true;
  }

  static void ParseAndSetConfigDefault(
      const std::string& name,
      const std::string& value) {
    // Check value's type
    bool boolValue;
    uint64_t uintValue;
    double doubleValue;
    Ptr<AttributeValue> config;
    if (Ns3ConfigMethods::isBooleanValue(value, boolValue)) {
      config_map_ns3[name] = std::make_unique<StoreVar<bool>>(boolValue);
      config = BooleanValue(boolValue).Copy();
    } else if (Ns3ConfigMethods::isUintegerValue(value, uintValue)) {
      config_map_ns3[name] = std::make_unique<StoreVar<uint64_t>>(uintValue);
      config = UintegerValue(uintValue).Copy();
    } else if (Ns3ConfigMethods::isDoubleValue(value, doubleValue)) {
      config_map_ns3[name] = std::make_unique<StoreVar<double>>(doubleValue);
      config = DoubleValue(doubleValue).Copy();
    } else {
      config_map_ns3[name] = std::make_unique<StoreVar<std::string>>(value);
      config = StringValue(value).Copy();
    }
    Config::SetDefault(name, *config);
  }

  static Ptr<AttributeValue> ParseConfig(const std::string& value){
    // Check value's type
    bool boolValue;
    uint64_t uintValue;
    double doubleValue;
    if (Ns3ConfigMethods::isBooleanValue(value, boolValue)) {
      // std::cout << value << " is BooleanValue" << std::endl;
      return BooleanValue(boolValue).Copy();
    } else if (Ns3ConfigMethods::isUintegerValue(value, uintValue)) {
      // std::cout << value << " is UintegerValue"<<std::endl;
      return UintegerValue(uintValue).Copy();
    } else if (Ns3ConfigMethods::isDoubleValue(value, doubleValue)) {
      // std::cout << value << " is DoubleValue"<<std::endl;
      return DoubleValue(doubleValue).Copy();
    } else {
      // std::cout << value << " is StringValue"<<std::endl;
      return StringValue(value).Copy();
    }
  }
  
  using ConfigEntry = std::pair<std::string, Ptr<AttributeValue>>;
};
// NS3设置的模板类
template <typename T>
class ConfigNs3 : public ConfigBase {
 public:
  ConfigNs3(const std::string& place) : place_(place) {}
  ConfigNs3(const std::string& place, T var)
      : place_(place), var_(var) {} // 可以设置默认值

  void set_value(const std::string& value) override {
    std::istringstream iss(value);
    iss >> var_; // 使用流操作符进行不同类型的转换
    Ns3ConfigMethods::SetConfigDefault(place_, var_);
  }

  T get_var() {
    return var_;
  }

  std::string get_place() {
    return place_;
  }

 private:
  T var_;
  std::string place_;
};

std::unordered_map<std::string, std::unique_ptr<ConfigBase>> config_map;

template <typename T>
inline T get_config_ns3_value_from_map(const std::string& name) {
  return dynamic_cast<ConfigNs3<T>*>(config_map[name].get())->get_var();
}


std::unordered_map<uint32_t, std::shared_ptr<std::vector<Ns3ConfigMethods::ConfigEntry>>> rdmaHw_config_map; // key: node id, value: a list of configs to set RdmaHw's attributes

class GroupConfig{
  public:
    std::string type;
    std::string nodes_str;
    std::vector<std::pair<std::string, std::string>> configs;

    std::unordered_map<uint32_t, std::shared_ptr<std::vector<Ns3ConfigMethods::ConfigEntry>>> * my_config_map;
    std::vector<uint32_t> nodes;
    void Parse(){
      // parse type
      if(type == "RdmaHw"){
        my_config_map = &rdmaHw_config_map;
      }else{
        NS_FATAL_ERROR("group config: type not supported" << type);
      }
      // parse nodes
      std::istringstream iss(nodes_str);
      std::string nodes_range;
      while(iss >> nodes_range){
        // 普通数字 or 数字范围，例如4:6 (包含6)
        if(nodes_range.find(':') != std::string::npos){
          std::string start = nodes_range.substr(0, nodes_range.find(':'));
          std::string end = nodes_range.substr(nodes_range.find(':') + 1);
          for(int i = std::stoi(start); i <= std::stoi(end); i++){
            nodes.push_back(i);
          }
        }else{
          nodes.push_back(std::stoi(nodes_range));
        }
      }
      if(nodes.size() == 0){
        NS_FATAL_ERROR("group config: nodes is empty");
      }
      // parse configs
      if(my_config_map->find(nodes[0]) == my_config_map->end()){
        my_config_map->insert(std::make_pair(nodes[0], std::make_shared<std::vector<Ns3ConfigMethods::ConfigEntry>>()));
      }else{
        NS_FATAL_ERROR("group config: node " << nodes[0]<<" has been set ");
      }
      for(auto& config : configs){
        my_config_map->at(nodes[0])->push_back(make_pair(config.first, Ns3ConfigMethods::ParseConfig(config.second)));
      }
      for(int i = 1; i < nodes.size(); i++){
        // they have the same configs
        if(my_config_map->find(nodes[i]) == my_config_map->end()){
          my_config_map->insert(std::make_pair(nodes[i], std::make_shared<std::vector<Ns3ConfigMethods::ConfigEntry>>(*my_config_map->at(nodes[0]))));
        } else {
          NS_FATAL_ERROR("group config: node " << nodes[i]<<" has been set ");
        }
      }
    }

    void Clear(){
      type = "";
      nodes_str = "";
      configs.clear();

      nodes.clear();
      my_config_map = nullptr;
    }

    void Print(){
      std::cout << "================GroupConfig================\ntype: " << type << std::endl << "nodes: ";
      for(auto node : nodes){
        std::cout << node << ", ";
      }
      std::cout << std::endl << "configs: " << std::endl;
      for(auto config : configs){
        std::cout << config.first << " = " << config.second << std::endl;
      }
      std::cout << "===========================================" << std::endl;
    }
};

#endif