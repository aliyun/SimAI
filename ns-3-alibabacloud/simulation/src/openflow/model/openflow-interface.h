/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * Author: Blake Hurd  <naimorai@gmail.com>
 */
#ifndef OPENFLOW_INTERFACE_H
#define OPENFLOW_INTERFACE_H

#include <assert.h>
#include <errno.h>

// Include OFSI code
#include "ns3/simulator.h"
#include "ns3/log.h"
#include "ns3/net-device.h"
#include "ns3/packet.h"
#include "ns3/address.h"
#include "ns3/nstime.h"
#include "ns3/mac48-address.h"

#include <set>
#include <map>
#include <limits>

// Include main header and Vendor Extension files
#include "openflow/openflow.h"
#include "openflow/nicira-ext.h"
#include "openflow/ericsson-ext.h"

extern "C"
{
// Inexplicably, the OpenFlow implementation uses these two reserved words as member names.
#define private _private
#define delete _delete
#define list List

// Include OFSI Library files
#include "openflow/private/csum.h"
#include "openflow/private/poll-loop.h"
#include "openflow/private/rconn.h"
#include "openflow/private/stp.h"
#include "openflow/private/vconn.h"
#include "openflow/private/xtoxll.h"

// Include OFSI Switch files
#include "openflow/private/chain.h"
#include "openflow/private/table.h"
#include "openflow/private/datapath.h" // The functions below are defined in datapath.c
uint32_t save_buffer (ofpbuf *);
ofpbuf * retrieve_buffer (uint32_t id);
void discard_buffer (uint32_t id);
#include "openflow/private/dp_act.h" // The functions below are defined in dp_act.c
void set_vlan_vid (ofpbuf *buffer, sw_flow_key *key, const ofp_action_header *ah);
void set_vlan_pcp (ofpbuf *buffer, sw_flow_key *key, const ofp_action_header *ah);
void strip_vlan (ofpbuf *buffer, sw_flow_key *key, const ofp_action_header *ah);
void set_dl_addr (ofpbuf *buffer, sw_flow_key *key, const ofp_action_header *ah);
void set_nw_addr (ofpbuf *buffer, sw_flow_key *key, const ofp_action_header *ah);
void set_tp_port (ofpbuf *buffer, sw_flow_key *key, const ofp_action_header *ah);
void set_mpls_label (ofpbuf *buffer, sw_flow_key *key, const ofp_action_header *ah);
void set_mpls_exp (ofpbuf *buffer, sw_flow_key *key, const ofp_action_header *ah);
#include "openflow/private/pt_act.h"

#undef list
#undef private
#undef delete
}

// Capabilities supported by this implementation.
#define OFP_SUPPORTED_CAPABILITIES ( OFPC_FLOW_STATS            \
                                     | OFPC_TABLE_STATS         \
                                     | OFPC_PORT_STATS          \
                                     | OFPC_MULTI_PHY_TX        \
                                     | OFPC_VPORT_TABLE)

// Actions supported by this implementation.
#define OFP_SUPPORTED_ACTIONS ( (1 << OFPAT_OUTPUT)             \
                                | (1 << OFPAT_SET_VLAN_VID)     \
                                | (1 << OFPAT_SET_VLAN_PCP)     \
                                | (1 << OFPAT_STRIP_VLAN)       \
                                | (1 << OFPAT_SET_DL_SRC)       \
                                | (1 << OFPAT_SET_DL_DST)       \
                                | (1 << OFPAT_SET_NW_SRC)       \
                                | (1 << OFPAT_SET_NW_DST)       \
                                | (1 << OFPAT_SET_TP_SRC)       \
                                | (1 << OFPAT_SET_TP_DST)       \
                                | (1 << OFPAT_SET_MPLS_LABEL)   \
                                | (1 << OFPAT_SET_MPLS_EXP) )

#define OFP_SUPPORTED_VPORT_TABLE_ACTIONS ( (1 << OFPPAT_OUTPUT)                \
                                            | (1 << OFPPAT_POP_MPLS)            \
                                            | (1 << OFPPAT_PUSH_MPLS)           \
                                            | (1 << OFPPAT_SET_MPLS_LABEL)      \
                                            | (1 << OFPPAT_SET_MPLS_EXP) )      \

namespace ns3 {

class OpenFlowSwitchNetDevice;

namespace ofi {

/**
 * \ingroup openflow
 * \brief Port and its metadata.
 *
 * We need to store port metadata, because OpenFlow dictates that there
 * exists a type of request where the Controller asks for data about a
 * port, or multiple ports. Otherwise, we'd refer to it via Ptr<NetDevice>
 * everywhere.
 */
struct Port
{
  Port () : config (0),
            state (0),
            netdev (0),
            rx_packets (0),
            tx_packets (0),
            rx_bytes (0),
            tx_bytes (0),
            tx_dropped (0),
            mpls_ttl0_dropped (0)
  {
  }

  uint32_t config;            ///< Some subset of OFPPC_* flags.
  uint32_t state;             ///< Some subset of OFPPS_* flags.
  Ptr<NetDevice> netdev;      ///< NetDevice pointer
  unsigned long long int rx_packets;  //!< Counter of Rx packets
  unsigned long long int tx_packets;  //!< Counter of Tx packets
  unsigned long long int rx_bytes;    //!< Counter of Rx bytes
  unsigned long long int tx_bytes;    //!< Counter of Tx bytes
  unsigned long long int tx_dropped;  //!< Counter of Tx dropped packets
  unsigned long long int mpls_ttl0_dropped; //!< Counter of packets dropped due to MPLS TTL
};

/**
 * \ingroup openflow
 * OpenFlow statistics
 */
class Stats
{
public:
  /**
   * Constructor
   * \param _type OpenFlow stats type.
   * \param body_len Stat body length.
   */
  Stats (ofp_stats_types _type, size_t body_len);

  /**
   * \brief Prepares to dump some kind of statistics on the connected OpenFlowSwitchNetDevice.
   *
   * \param body Body member of the struct ofp_stats_request.
   * \param body_len Length of the body member.
   * \param state State information.
   * \return 0 if successful, otherwise a negative error code.
   */
  int DoInit (const void *body, int body_len, void **state);

  /**
   * \brief Appends statistics for OpenFlowSwitchNetDevice to 'buffer'.
   *
   * \param swtch The OpenFlowSwitchNetDevice this callback is associated with.
   * \param state State information.
   * \param buffer Buffer to append stats reply to.
   * \return 1 if it should be called again later with another buffer, 0 if it is done, or a negative errno value on failure.
   */
  int DoDump (Ptr<OpenFlowSwitchNetDevice> swtch, void *state, ofpbuf *buffer);

  /**
   * \brief Cleans any state created by the init or dump functions.
   *
   * May not be implemented if no cleanup is required.
   *
   * \param state State information to clear.
   */
  void DoCleanup (void *state);

  /**
   * \brief State of the FlowStats request/reply.
   */
  struct FlowStatsState
  {
    int table_idx;      //!< Table index
    sw_table_position position; //!< Table position
    ofp_flow_stats_request rq;  //!< Stats requests
    time_t now; //!< Actual time

    ofpbuf *buffer; //!< Buffer
  };

  /**
   * \ingroup openflow
   * \brief State of the PortStats request/reply.
   */
  struct PortStatsState
  {
    uint32_t num_ports; ///< Number of ports in host byte order
    uint32_t *ports;    ///< Array of ports in network byte order
  };

  ofp_stats_types type; //!< Status type
private:
  /**
   * Dumps the stats description
   * \param [in] state The state.
   * \param [out] buffer Output buffer.
   * \return zero on success
   */
  int DescStatsDump (void *state, ofpbuf *buffer);

  /**
   * @{
   * Initialize the stats.
   * \param body Body member of the struct ofp_stats_request.
   * \param body_len Length of the body member.
   * \param state State information.
   * \return 0 if successful, otherwise a negative error code.
   */
  int FlowStatsInit (const void *body, int body_len, void **state);
  int AggregateStatsInit (const void *body, int body_len, void **state);
  int PortStatsInit (const void *body, int body_len, void **state);
  /** @} */

  /// Flow dump callback functor
  int (*FlowDumpCallback)(sw_flow *flow, void *state);
  /// Aggregate dump callback functor
  int (*AggregateDumpCallback)(sw_flow *flow, void *state);

  /**
   * @{
   * Dump the stats.
   * \param dp OpenFlow NetDevice.
   * \param state State.
   * \param [out] buffer output buffer.
   * \return 0 if successful
   */
  int FlowStatsDump (Ptr<OpenFlowSwitchNetDevice> dp, FlowStatsState *state, ofpbuf *buffer);
  int AggregateStatsDump (Ptr<OpenFlowSwitchNetDevice> dp, ofp_aggregate_stats_request *state, ofpbuf *buffer);
  int TableStatsDump (Ptr<OpenFlowSwitchNetDevice> dp, void *state, ofpbuf *buffer);
  int PortStatsDump (Ptr<OpenFlowSwitchNetDevice> dp, PortStatsState *state, ofpbuf *buffer);
  int PortTableStatsDump (Ptr<OpenFlowSwitchNetDevice> dp, void *state, ofpbuf *buffer);
  /** @} */
};

/**
 * \ingroup openflow
 * \brief Class for handling flow table actions.
 */
struct Action
{
  /**
   * \param type Type of Flow Table Action.
   * \return true if the provided type is a type of flow table action.
   */
  static bool IsValidType (ofp_action_type type);

  /**
   * \brief Validates the action on whether its data is valid or not.
   *
   * \param type Type of action to validate.
   * \param len Length of the action data.
   * \param key Matching key for the flow that is tied to this action.
   * \param ah Action's data header.
   * \return ACT_VALIDATION_OK if the action checks out, otherwise an error type.
   */
  static uint16_t Validate (ofp_action_type type, size_t len, const sw_flow_key *key, const ofp_action_header *ah);

  /**
   * \brief Executes the action.
   *
   * \param type Type of action to execute.
   * \param buffer Buffer of the Packet if it's needed for the action.
   * \param key Matching key for the flow that is tied to this action.
   * \param ah Action's data header.
   */
  static void Execute (ofp_action_type type, ofpbuf *buffer, sw_flow_key *key, const ofp_action_header *ah);
};

/**
 * \ingroup openflow
 * \brief Class for handling virtual port table actions.
 */
struct VPortAction
{
  /**
   * \param type Type of virtual port table Action.
   * \return true if the provided type is a type of virtual port table action.
   */
  static bool IsValidType (ofp_vport_action_type type);

  /**
   * \brief Validates the action on whether its data is valid or not.
   *
   * \param type Type of action to validate.
   * \param len Length of the action data.
   * \param ah Action's data header.
   * \return ACT_VALIDATION_OK if the action checks out, otherwise an error type.
   */
  static uint16_t Validate (ofp_vport_action_type type, size_t len, const ofp_action_header *ah);

  /**
   * \brief Executes the action.
   *
   * \param type Type of action to execute.
   * \param buffer Buffer of the Packet if it's needed for the action.
   * \param key Matching key for the flow that is tied to this action.
   * \param ah Action's data header.
   */
  static void Execute (ofp_vport_action_type type, ofpbuf *buffer, const sw_flow_key *key, const ofp_action_header *ah);
};

/**
 * \ingroup openflow
 * \brief Class for handling Ericsson Vendor-defined actions.
 */
struct EricssonAction
{
  /**
   * \param type Type of Ericsson Vendor-defined Action.
   * \return true if the provided type is a type of Ericsson Vendor-defined action.
   */
  static bool IsValidType (er_action_type type);

  /**
   * \brief Validates the action on whether its data is valid or not.
   *
   * \param type Type of action to validate.
   * \param len Length of the action data.
   * \return ACT_VALIDATION_OK if the action checks out, otherwise an error type.
   */
  static uint16_t Validate (er_action_type type, size_t len);

  /**
   * \brief Executes the action.
   *
   * \param type Type of action to execute.
   * \param buffer Buffer of the Packet if it's needed for the action.
   * \param key Matching key for the flow that is tied to this action.
   * \param ah Action's data header.
   */
  static void Execute (er_action_type type, ofpbuf *buffer, const sw_flow_key *key, const er_action_header *ah);
};

/**
 * \ingroup openflow
 * \brief Callback for a stats dump request.
 */
struct StatsDumpCallback
{
  bool done;                            ///< Whether we are done requesting stats.
  ofp_stats_request *rq;                ///< Current stats request.
  Stats *s;                             ///< Handler of the stats request.
  void *state;                          ///< Stats request state data.
  Ptr<OpenFlowSwitchNetDevice> swtch;   ///< The switch that we're requesting data from.
};

/**
 * \ingroup openflow
 * \brief Packet Metadata, allows us to track the packet's metadata as it passes through the switch.
 */
struct SwitchPacketMetadata
{
  Ptr<Packet> packet; ///< The Packet itself.
  ofpbuf* buffer;               ///< The OpenFlow buffer as created from the Packet, with its data and headers.
  uint16_t protocolNumber;      ///< Protocol type of the Packet when the Packet is received
  Address src;             ///< Source Address of the Packet when the Packet is received
  Address dst;             ///< Destination Address of the Packet when the Packet is received.
};

/**
 * \ingroup openflow
 * \brief An interface for a Controller of OpenFlowSwitchNetDevices
 *
 * Follows the OpenFlow specification for a controller.
 */
class Controller : public Object
{
public:
  /**
   * Register this type.
   * \return The TypeId.
   */
  static TypeId GetTypeId (void);
  /** Destructor. */
  virtual ~Controller ();

  /**
   * Adds a switch to the controller.
   *
   * \param swtch The switch to register.
   */
  virtual void AddSwitch (Ptr<OpenFlowSwitchNetDevice> swtch);

  /**
   * A switch calls this method to pass a message on to the Controller.
   *
   * \param swtch The switch the message was received from.
   * \param buffer The message.
   */
  virtual void ReceiveFromSwitch (Ptr<OpenFlowSwitchNetDevice> swtch, ofpbuf* buffer)
  {
  }

  /**
   * \brief Starts a callback-based, reliable, possibly multi-message reply to a request made by the controller.
   *
   * If an incoming request needs to have a reliable reply that might
   * require multiple messages, it can use StartDump() to set up
   * a callback that will be called as buffer space for replies.
   *
   * A stats request made by the controller is processed by the switch,
   * the switch then calls this method to tell the controller to start
   * asking for information. By default (it can be overridden), the
   * controller stops all work to run through the callback. ReceiveFromSwitch
   * must be defined appropriately to handle the status reply messages
   * generated by the switch, or otherwise the status reply messages will be sent
   * and discarded.
   *
   * \param cb The callback data.
   */
  void StartDump (StatsDumpCallback* cb);

protected:
  /**
   * However the controller is implemented, this method is to
   * be used to pass a message on to a switch.
   *
   * \param swtch The switch to receive the message.
   * \param msg The message to send.
   * \param length The length of the message.
   */
  virtual void SendToSwitch (Ptr<OpenFlowSwitchNetDevice> swtch, void * msg, size_t length);

  /**
   * Construct flow data from a matching key to build a flow
   * entry for adding, modifying, or deleting a flow.
   *
   * \param key The matching key data; used to create a flow that matches the packet.
   * \param buffer_id The OpenFlow Buffer ID; used to run the actions on the packet if we add or modify the flow.
   * \param command Whether to add, modify, or delete this flow.
   * \param acts List of actions to execute.
   * \param actions_len Length of the actions buffer.
   * \param idle_timeout Flow expires if left inactive for this amount of time (specify OFP_FLOW_PERMANENT to disable feature).
   * \param hard_timeout Flow expires after this amount of time (specify OFP_FLOW_PERMANENT to disable feature).
   * \return Flow data that when passed to SetFlow will add, modify, or delete a flow it defines.
   */
  ofp_flow_mod* BuildFlow (sw_flow_key key, uint32_t buffer_id, uint16_t command, void* acts, size_t actions_len, int idle_timeout, int hard_timeout);

  /**
   * Get the packet type on the buffer, which can then be used
   * to determine how to handle the buffer.
   *
   * \param buffer The packet in OpenFlow buffer format.
   * \return The packet type, as defined in the ofp_type struct.
   */
  uint8_t GetPacketType (ofpbuf* buffer);

  /// OpenFlowSwitchNetDevice container type
  typedef std::set<Ptr<OpenFlowSwitchNetDevice> > Switches_t;
  Switches_t m_switches;  ///< The collection of switches registered to this controller.
};

/**
 * \ingroup openflow
 * Demonstration of a Drop controller. When a connected switch
 * passes it a packet the switch doesn't recognize, the controller
 * configures the switch to make a flow that drops alike packets.
 */
class DropController : public Controller
{
public:
  /**
   * Register this type.
   * \return The TypeId.
   */
  static TypeId GetTypeId (void);
  
  void ReceiveFromSwitch (Ptr<OpenFlowSwitchNetDevice> swtch, ofpbuf* buffer);
};

/**
 * \ingroup openflow
 * Demonstration of a Learning controller. When a connected switch
 * passes it a packet the switch doesn't recognize, the controller
 * delves into its learned states and figures out if we know what
 * port the packet is supposed to go to, flooding if unknown, and
 * adjusts the switch's flow table accordingly.
 */
class LearningController : public Controller
{
public:
  /**
   * Register this type.
   * \return The TypeId.
   */
  static TypeId GetTypeId (void);

  virtual ~LearningController ()
  {
    m_learnState.clear ();
  }

  void ReceiveFromSwitch (Ptr<OpenFlowSwitchNetDevice> swtch, ofpbuf* buffer);

protected:
  /// Learned state
  struct LearnedState
  {
    uint32_t port;                      ///< Learned port.
  };
  Time m_expirationTime;                ///< Time it takes for learned MAC state entry/created flow to expire.
  /// Learned state type
  typedef std::map<Mac48Address, LearnedState> LearnState_t;
  LearnState_t m_learnState;            ///< Learned state data.
};

/**
 * \brief Executes a list of flow table actions.
 *
 * \param swtch OpenFlowSwitchNetDevice these actions are being executed on.
 * \param packet_uid Packet UID; used to fetch the packet and its metadata.
 * \param buffer The Packet OpenFlow buffer.
 * \param key The matching key for the flow tied to this list of actions.
 * \param actions A buffer of actions.
 * \param actions_len Length of actions buffer.
 * \param ignore_no_fwd If true, during port forwarding actions, ports that are set to not forward are forced to forward.
 */
void ExecuteActions (Ptr<OpenFlowSwitchNetDevice> swtch, uint64_t packet_uid, ofpbuf* buffer, sw_flow_key *key, const ofp_action_header *actions, size_t actions_len, int ignore_no_fwd);

/**
 * \brief Validates a list of flow table actions.
 *
 * \param key The matching key for the flow tied to this list of actions.
 * \param actions A buffer of actions.
 * \param actions_len Length of actions buffer.
 * \return If the action list validates, ACT_VALIDATION_OK is returned. Otherwise, a code for the OFPET_BAD_ACTION error type is returned.
 */
uint16_t ValidateActions (const sw_flow_key *key, const ofp_action_header *actions, size_t actions_len);

/**
 * \brief Executes a list of virtual port table entry actions.
 *
 * \param swtch OpenFlowSwitchNetDevice these actions are being executed on.
 * \param packet_uid Packet UID; used to fetch the packet and its metadata.
 * \param buffer The Packet OpenFlow buffer.
 * \param key The matching key for the flow tied to this list of actions.
 * \param actions A buffer of actions.
 * \param actions_len Length of actions buffer.
 */
void ExecuteVPortActions (Ptr<OpenFlowSwitchNetDevice> swtch, uint64_t packet_uid, ofpbuf* buffer, sw_flow_key *key, const ofp_action_header *actions, size_t actions_len);

/**
 * \brief Validates a list of virtual port table entry actions.
 *
 * \param actions A buffer of actions.
 * \param actions_len Length of actions buffer.
 * \return If the action list validates, ACT_VALIDATION_OK is returned. Otherwise, a code for the OFPET_BAD_ACTION error type is returned.
 */
uint16_t ValidateVPortActions (const ofp_action_header *actions, size_t actions_len);

/**
 * \brief Executes a vendor-defined action.
 *
 * \param buffer The Packet OpenFlow buffer.
 * \param key The matching key for the flow tied to this list of actions.
 * \param ah Header of the action.
 */
void ExecuteVendor (ofpbuf *buffer, const sw_flow_key *key, const ofp_action_header *ah);

/**
 * \brief Validates a vendor-defined action.
 *
 * \param key The matching key for the flow tied to this list of actions.
 * \param ah Header of the action.
 * \param len Length of the action.
 * \return If the action list validates, ACT_VALIDATION_OK is returned. Otherwise, a code for the OFPET_BAD_ACTION error type is returned.
 */
uint16_t ValidateVendor (const sw_flow_key *key, const ofp_action_header *ah, uint16_t len);

/* 
 * From datapath.c
 * Buffers are identified to userspace by a 31-bit opaque ID.  We divide the ID
 * into a buffer number (low bits) and a cookie (high bits).  The buffer number
 * is an index into an array of buffers.  The cookie distinguishes between
 * different packets that have occupied a single buffer.  Thus, the more
 * buffers we have, the lower-quality the cookie...
 */
#define PKT_BUFFER_BITS 8
#define N_PKT_BUFFERS (1 << PKT_BUFFER_BITS)
#define PKT_BUFFER_MASK (N_PKT_BUFFERS - 1)
#define PKT_COOKIE_BITS (32 - PKT_BUFFER_BITS)

}

}

#endif /* OPENFLOW_INTERFACE_H */
