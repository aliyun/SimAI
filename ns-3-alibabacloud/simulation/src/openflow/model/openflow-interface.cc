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
#ifdef NS3_OPENFLOW

#include "openflow-interface.h"
#include "openflow-switch-net-device.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("OpenFlowInterface");

namespace ofi {

Stats::Stats (ofp_stats_types _type, size_t body_len)
{
  type = _type;
  size_t min_body = 0, max_body = 0;

  switch (type)
    {
    case OFPST_DESC:
      break;
    case OFPST_FLOW:
      min_body = max_body = sizeof(ofp_flow_stats_request);
      break;
    case OFPST_AGGREGATE:
      min_body = max_body = sizeof(ofp_aggregate_stats_request);
      break;
    case OFPST_TABLE:
      break;
    case OFPST_PORT:
      min_body = 0;
      max_body = std::numeric_limits<size_t>::max (); // Not sure about this one. This would guarantee that the body_len is always acceptable.
      break;
    case OFPST_PORT_TABLE:
      break;
    default:
      NS_LOG_ERROR ("received stats request of unknown type " << type);
      return; // -EINVAL;
    }

  if ((min_body != 0 || max_body != 0) && (body_len < min_body || body_len > max_body))
    {
      NS_LOG_ERROR ("stats request type " << type << " with bad body length " << body_len);
      return; // -EINVAL;
    }
}

int
Stats::DoInit (const void *body, int body_len, void **state)
{
  switch (type)
    {
    case OFPST_DESC:
      return 0;
    case OFPST_FLOW:
      return FlowStatsInit (body, body_len, state);
    case OFPST_AGGREGATE:
      return AggregateStatsInit (body, body_len, state);
    case OFPST_TABLE:
      return 0;
    case OFPST_PORT:
      return PortStatsInit (body, body_len, state);
    case OFPST_PORT_TABLE:
      return 0;
    case OFPST_VENDOR:
      return 0;
    }

  return 0;
}

int
Stats::DoDump (Ptr<OpenFlowSwitchNetDevice> swtch, void *state, ofpbuf *buffer)
{
  switch (type)
    {
    case OFPST_DESC:
      return DescStatsDump (state, buffer);
    case OFPST_FLOW:
      return FlowStatsDump (swtch, (FlowStatsState *)state, buffer);
    case OFPST_AGGREGATE:
      return AggregateStatsDump (swtch, (ofp_aggregate_stats_request *)state, buffer);
    case OFPST_TABLE:
      return TableStatsDump (swtch, state, buffer);
    case OFPST_PORT:
      return PortStatsDump (swtch, (PortStatsState *)state, buffer);
    case OFPST_PORT_TABLE:
      return PortTableStatsDump (swtch, state, buffer);
    case OFPST_VENDOR:
      return 0;
    }

  return 0;
}

void
Stats::DoCleanup (void *state)
{
  switch (type)
    {
    case OFPST_DESC:
      break;
    case OFPST_FLOW:
      free ((FlowStatsState *)state);
      break;
    case OFPST_AGGREGATE:
      free ((ofp_aggregate_stats_request *)state);
      break;
    case OFPST_TABLE:
      break;
    case OFPST_PORT:
      free (((PortStatsState *)state)->ports);
      free ((PortStatsState *)state);
      break;
    case OFPST_PORT_TABLE:
      break;
    case OFPST_VENDOR:
      break;
    }
}

int
Stats::DescStatsDump (void *state, ofpbuf *buffer)
{
  ofp_desc_stats *ods = (ofp_desc_stats*)ofpbuf_put_zeros (buffer, sizeof *ods);
  strncpy (ods->mfr_desc, OpenFlowSwitchNetDevice::GetManufacturerDescription (), sizeof ods->mfr_desc);
  strncpy (ods->hw_desc, OpenFlowSwitchNetDevice::GetHardwareDescription (), sizeof ods->hw_desc);
  strncpy (ods->sw_desc, OpenFlowSwitchNetDevice::GetSoftwareDescription (), sizeof ods->sw_desc);
  strncpy (ods->serial_num, OpenFlowSwitchNetDevice::GetSerialNumber (), sizeof ods->serial_num);
  return 0;
}

#define MAX_FLOW_STATS_BYTES 4096

int
Stats::FlowStatsInit (const void *body, int body_len, void **state)
{
  const ofp_flow_stats_request *fsr = (ofp_flow_stats_request*)body;
  FlowStatsState *s = (FlowStatsState*)xmalloc (sizeof *s);

  s->table_idx = fsr->table_id == 0xff ? 0 : fsr->table_id;
  memset (&s->position, 0, sizeof s->position);
  s->rq = *fsr;
  *state = s;
  return 0;
}

int
Stats_FlowDumpCallback (sw_flow *flow, void* state)
{
  Stats::FlowStatsState *s = (Stats::FlowStatsState*)state;

  // Fill Flow Stats
  ofp_flow_stats *ofs;
  int length = sizeof *ofs + flow->sf_acts->actions_len;
  ofs = (ofp_flow_stats*)ofpbuf_put_zeros (s->buffer, length);
  ofs->length          = htons (length);
  ofs->table_id        = s->table_idx;
  ofs->match.wildcards = htonl (flow->key.wildcards);
  ofs->match.in_port   = flow->key.flow.in_port;
  memcpy (ofs->match.dl_src, flow->key.flow.dl_src, ETH_ADDR_LEN);
  memcpy (ofs->match.dl_dst, flow->key.flow.dl_dst, ETH_ADDR_LEN);
  ofs->match.dl_vlan   = flow->key.flow.dl_vlan;
  ofs->match.dl_type   = flow->key.flow.dl_type;
  ofs->match.nw_src    = flow->key.flow.nw_src;
  ofs->match.nw_dst    = flow->key.flow.nw_dst;
  ofs->match.nw_proto  = flow->key.flow.nw_proto;
  ofs->match.tp_src    = flow->key.flow.tp_src;
  ofs->match.tp_dst    = flow->key.flow.tp_dst;
  ofs->duration        = htonl (s->now - flow->created);
  ofs->priority        = htons (flow->priority);
  ofs->idle_timeout    = htons (flow->idle_timeout);
  ofs->hard_timeout    = htons (flow->hard_timeout);
  ofs->packet_count    = htonll (flow->packet_count);
  ofs->byte_count      = htonll (flow->byte_count);
  memcpy (ofs->actions, flow->sf_acts->actions, flow->sf_acts->actions_len);

  return s->buffer->size >= MAX_FLOW_STATS_BYTES;
}

int
Stats::FlowStatsDump (Ptr<OpenFlowSwitchNetDevice> swtch, FlowStatsState* s, ofpbuf *buffer)
{
  sw_flow_key match_key;

  flow_extract_match (&match_key, &s->rq.match);

  s->buffer = buffer;
  s->now = time_now ();
  while (s->table_idx < swtch->GetChain ()->n_tables
         && (s->rq.table_id == 0xff || s->rq.table_id == s->table_idx))
    {
      sw_table *table = swtch->GetChain ()->tables[s->table_idx];

      if (table->iterate (table, &match_key, s->rq.out_port, &s->position, Stats::FlowDumpCallback, s))
        {
          break;
        }

      s->table_idx++;
      memset (&s->position, 0, sizeof s->position);
    }
  return s->buffer->size >= MAX_FLOW_STATS_BYTES;
}

int
Stats::AggregateStatsInit (const void *body, int body_len, void **state)
{
  //ofp_aggregate_stats_request *s = (ofp_aggregate_stats_request*)body;
  *state = (ofp_aggregate_stats_request*)body;
  return 0;
}

int
Stats_AggregateDumpCallback (sw_flow *flow, void *state)
{
  ofp_aggregate_stats_reply *s = (ofp_aggregate_stats_reply*)state;
  s->packet_count += flow->packet_count;
  s->byte_count += flow->byte_count;
  s->flow_count++;
  return 0;
}

int
Stats::AggregateStatsDump (Ptr<OpenFlowSwitchNetDevice> swtch, ofp_aggregate_stats_request *s, ofpbuf *buffer)
{
  ofp_aggregate_stats_request *rq = s;
  ofp_aggregate_stats_reply *rpy = (ofp_aggregate_stats_reply*)ofpbuf_put_zeros (buffer, sizeof *rpy);
  sw_flow_key match_key;
  flow_extract_match (&match_key, &rq->match);
  int table_idx = rq->table_id == 0xff ? 0 : rq->table_id;

  sw_table_position position;
  memset (&position, 0, sizeof position);

  while (table_idx < swtch->GetChain ()->n_tables
         && (rq->table_id == 0xff || rq->table_id == table_idx))
    {
      sw_table *table = swtch->GetChain ()->tables[table_idx];
      int error = table->iterate (table, &match_key, rq->out_port, &position, Stats::AggregateDumpCallback, rpy);
      if (error)
        {
          return error;
        }

      table_idx++;
      memset (&position, 0, sizeof position);
    }

  rpy->packet_count = htonll (rpy->packet_count);
  rpy->byte_count = htonll (rpy->byte_count);
  rpy->flow_count = htonl (rpy->flow_count);
  return 0;
}

int
Stats::TableStatsDump (Ptr<OpenFlowSwitchNetDevice> swtch, void *state, ofpbuf *buffer)
{
  sw_chain* ft = swtch->GetChain ();
  for (int i = 0; i < ft->n_tables; i++)
    {
      ofp_table_stats *ots = (ofp_table_stats*)ofpbuf_put_zeros (buffer, sizeof *ots);
      sw_table_stats stats;
      ft->tables[i]->stats (ft->tables[i], &stats);
      strncpy (ots->name, stats.name, sizeof ots->name);
      ots->table_id = i;
      ots->wildcards = htonl (stats.wildcards);
      ots->max_entries = htonl (stats.max_flows);
      ots->active_count = htonl (stats.n_flows);
      ots->lookup_count = htonll (stats.n_lookup);
      ots->matched_count = htonll (stats.n_matched);
    }
  return 0;
}

// stats for the port table which is similar to stats for the flow tables
int
Stats::PortTableStatsDump (Ptr<OpenFlowSwitchNetDevice> swtch, void *state, ofpbuf *buffer)
{
  ofp_vport_table_stats *opts = (ofp_vport_table_stats*)ofpbuf_put_zeros (buffer, sizeof *opts);
  opts->max_vports = htonl (swtch->GetVPortTable ().max_vports);
  opts->active_vports = htonl (swtch->GetVPortTable ().active_vports);
  opts->lookup_count = htonll (swtch->GetVPortTable ().lookup_count);
  opts->port_match_count = htonll (swtch->GetVPortTable ().port_match_count);
  opts->chain_match_count = htonll (swtch->GetVPortTable ().chain_match_count);

  return 0;
}

int
Stats::PortStatsInit (const void *body, int body_len, void **state)
{
  PortStatsState *s = (PortStatsState*)xmalloc (sizeof *s);

  // the body contains a list of port numbers
  s->ports = (uint32_t*)xmalloc (body_len);
  memcpy (s->ports, body, body_len);
  s->num_ports = body_len / sizeof(uint32_t);

  *state = s;
  return 0;
}

int
Stats::PortStatsDump (Ptr<OpenFlowSwitchNetDevice> swtch, PortStatsState *s, ofpbuf *buffer)
{
  ofp_port_stats *ops;
  uint32_t port;

  // port stats are different depending on whether port is physical or virtual
  for (size_t i = 0; i < s->num_ports; i++)
    {
      port = ntohl (s->ports[i]);
      // physical port?
      if (port <= OFPP_MAX)
        {
          Port p = swtch->GetSwitchPort (port);

          if (p.netdev == 0)
            {
              continue;
            }

          ops = (ofp_port_stats*)ofpbuf_put_zeros (buffer, sizeof *ops);
          ops->port_no = htonl (swtch->GetSwitchPortIndex (p));
          ops->rx_packets   = htonll (p.rx_packets);
          ops->tx_packets   = htonll (p.tx_packets);
          ops->rx_bytes     = htonll (p.rx_bytes);
          ops->tx_bytes     = htonll (p.tx_bytes);
          ops->rx_dropped   = htonll (-1);
          ops->tx_dropped   = htonll (p.tx_dropped);
          ops->rx_errors    = htonll (-1);
          ops->tx_errors    = htonll (-1);
          ops->rx_frame_err = htonll (-1);
          ops->rx_over_err  = htonll (-1);
          ops->rx_crc_err   = htonll (-1);
          ops->collisions   = htonll (-1);
          ops->mpls_ttl0_dropped = htonll (p.mpls_ttl0_dropped);
          ops++;
        } 
      else if (port >= OFPP_VP_START && port <= OFPP_VP_END) // virtual port?
        {
          // lookup the virtual port
          vport_table_t vt = swtch->GetVPortTable ();
          vport_table_entry *vpe = vport_table_lookup (&vt, port);
          if (vpe == 0)
            {
              NS_LOG_ERROR ("vport entry not found!");
              continue;
            }
          // only tx_packets and tx_bytes are really relevant for virtual ports
          ops = (ofp_port_stats*)ofpbuf_put_zeros (buffer, sizeof *ops);
          ops->port_no = htonl (vpe->vport);
          ops->rx_packets   = htonll (-1);
          ops->tx_packets   = htonll (vpe->packet_count);
          ops->rx_bytes     = htonll (-1);
          ops->tx_bytes     = htonll (vpe->byte_count);
          ops->rx_dropped   = htonll (-1);
          ops->tx_dropped   = htonll (-1);
          ops->rx_errors    = htonll (-1);
          ops->tx_errors    = htonll (-1);
          ops->rx_frame_err = htonll (-1);
          ops->rx_over_err  = htonll (-1);
          ops->rx_crc_err   = htonll (-1);
          ops->collisions   = htonll (-1);
          ops->mpls_ttl0_dropped = htonll (-1);
          ops++;
        }
    }
  return 0;
}

bool
Action::IsValidType (ofp_action_type type)
{
  switch (type)
    {
    case OFPAT_OUTPUT:
    case OFPAT_SET_VLAN_VID:
    case OFPAT_SET_VLAN_PCP:
    case OFPAT_STRIP_VLAN:
    case OFPAT_SET_DL_SRC:
    case OFPAT_SET_DL_DST:
    case OFPAT_SET_NW_SRC:
    case OFPAT_SET_NW_DST:
    case OFPAT_SET_TP_SRC:
    case OFPAT_SET_TP_DST:
    case OFPAT_SET_MPLS_LABEL:
    case OFPAT_SET_MPLS_EXP:
      return true;
    default:
      return false;
    }
}

uint16_t
Action::Validate (ofp_action_type type, size_t len, const sw_flow_key *key, const ofp_action_header *ah)
{
  size_t size = 0;

  switch (type)
    {
    case OFPAT_OUTPUT:
      {
        if (len != sizeof(ofp_action_output))
          {
            return OFPBAC_BAD_LEN;
          }

        ofp_action_output *oa = (ofp_action_output *)ah;

        // To prevent loops, make sure there's no action to send to the OFP_TABLE virtual port.

        // port is now 32-bit
        if (oa->port == OFPP_NONE || oa->port == key->flow.in_port) // htonl(OFPP_NONE);
          { // if (oa->port == htons(OFPP_NONE) || oa->port == key->flow.in_port)
            return OFPBAC_BAD_OUT_PORT;
          }

        return ACT_VALIDATION_OK;
      }
    case OFPAT_SET_VLAN_VID:
      size = sizeof(ofp_action_vlan_vid);
      break;
    case OFPAT_SET_VLAN_PCP:
      size = sizeof(ofp_action_vlan_pcp);
      break;
    case OFPAT_STRIP_VLAN:
      size = sizeof(ofp_action_header);
      break;
    case OFPAT_SET_DL_SRC:
    case OFPAT_SET_DL_DST:
      size = sizeof(ofp_action_dl_addr);
      break;
    case OFPAT_SET_NW_SRC:
    case OFPAT_SET_NW_DST:
      size = sizeof(ofp_action_nw_addr);
      break;
    case OFPAT_SET_TP_SRC:
    case OFPAT_SET_TP_DST:
      size = sizeof(ofp_action_tp_port);
      break;
    case OFPAT_SET_MPLS_LABEL:
      size = sizeof(ofp_action_mpls_label);
      break;
    case OFPAT_SET_MPLS_EXP:
      size = sizeof(ofp_action_mpls_exp);
      break;
    default:
      break;
    }

  if (len != size)
    {
      return OFPBAC_BAD_LEN;
    }
  return ACT_VALIDATION_OK;
}

void
Action::Execute (ofp_action_type type, ofpbuf *buffer, sw_flow_key *key, const ofp_action_header *ah)
{
  switch (type)
    {
    case OFPAT_OUTPUT:
      break;
    case OFPAT_SET_VLAN_VID:
      set_vlan_vid (buffer, key, ah);
      break;
    case OFPAT_SET_VLAN_PCP:
      set_vlan_pcp (buffer, key, ah);
      break;
    case OFPAT_STRIP_VLAN:
      strip_vlan (buffer, key, ah);
      break;
    case OFPAT_SET_DL_SRC:
    case OFPAT_SET_DL_DST:
      set_dl_addr (buffer, key, ah);
      break;
    case OFPAT_SET_NW_SRC:
    case OFPAT_SET_NW_DST:
      set_nw_addr (buffer, key, ah);
      break;
    case OFPAT_SET_TP_SRC:
    case OFPAT_SET_TP_DST:
      set_tp_port (buffer, key, ah);
      break;
    case OFPAT_SET_MPLS_LABEL:
      set_mpls_label (buffer, key, ah);
      break;
    case OFPAT_SET_MPLS_EXP:
      set_mpls_exp (buffer, key, ah);
      break;
    default:
      break;
    }
}

bool
VPortAction::IsValidType (ofp_vport_action_type type)
{
  switch (type)
    {
    case OFPPAT_POP_MPLS:
    case OFPPAT_PUSH_MPLS:
    case OFPPAT_SET_MPLS_LABEL:
    case OFPPAT_SET_MPLS_EXP:
      return true;
    default:
      return false;
    }
}

uint16_t
VPortAction::Validate (ofp_vport_action_type type, size_t len, const ofp_action_header *ah)
{
  size_t size = 0;

  switch (type)
    {
    case OFPPAT_POP_MPLS:
      size = sizeof(ofp_vport_action_pop_mpls);
      break;
    case OFPPAT_PUSH_MPLS:
      size = sizeof(ofp_vport_action_push_mpls);
      break;
    case OFPPAT_SET_MPLS_LABEL:
      size = sizeof(ofp_vport_action_set_mpls_label);
      break;
    case OFPPAT_SET_MPLS_EXP:
      size = sizeof(ofp_vport_action_set_mpls_exp);
      break;
    default:
      break;
    }

  if (len != size)
    {
      return OFPBAC_BAD_LEN;
    }
  return ACT_VALIDATION_OK;
}

void
VPortAction::Execute (ofp_vport_action_type type, ofpbuf *buffer, const sw_flow_key *key, const ofp_action_header *ah)
{
  switch (type)
    {
    case OFPPAT_POP_MPLS:
      {
        ofp_vport_action_pop_mpls *opapm = (ofp_vport_action_pop_mpls *)ah;
        pop_mpls_act (0, buffer, key, &opapm->apm);
        break;
      }
    case OFPPAT_PUSH_MPLS:
      {
        ofp_vport_action_push_mpls *opapm = (ofp_vport_action_push_mpls *)ah;
        push_mpls_act (0, buffer, key, &opapm->apm);
        break;
      }
    case OFPPAT_SET_MPLS_LABEL:
      {
        ofp_vport_action_set_mpls_label *oparml = (ofp_vport_action_set_mpls_label *)ah;
        set_mpls_label_act (buffer, key, oparml->label_out);
        break;
      }
    case OFPPAT_SET_MPLS_EXP:
      {
        ofp_vport_action_set_mpls_exp *oparme = (ofp_vport_action_set_mpls_exp *)ah;
        set_mpls_exp_act (buffer, key, oparme->exp);
        break;
      }
    default:
      break;
    }
}

bool
EricssonAction::IsValidType (er_action_type type)
{
  switch (type)
    {
    case ERXT_POP_MPLS:
    case ERXT_PUSH_MPLS:
      return true;
    default:
      return false;
    }
}

uint16_t
EricssonAction::Validate (er_action_type type, size_t len)
{
  size_t size = 0;

  switch (type)
    {
    case ERXT_POP_MPLS:
      size = sizeof(er_action_pop_mpls);
      break;
    case ERXT_PUSH_MPLS:
      size = sizeof(er_action_push_mpls);
      break;
    default:
      break;
    }

  if (len != size)
    {
      return OFPBAC_BAD_LEN;
    }
  return ACT_VALIDATION_OK;
}

void
EricssonAction::Execute (er_action_type type, ofpbuf *buffer, const sw_flow_key *key, const er_action_header *ah)
{
  switch (type)
    {
    case ERXT_POP_MPLS:
      {
        er_action_pop_mpls *erapm = (er_action_pop_mpls *)ah;
        pop_mpls_act (0, buffer, key, &erapm->apm);
        break;
      }
    case ERXT_PUSH_MPLS:
      {
        er_action_push_mpls *erapm = (er_action_push_mpls *)ah;
        push_mpls_act (0, buffer, key, &erapm->apm);
        break;
      }
    default:
      break;
    }
}

/* static */
TypeId
Controller::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::ofi::Controller")
    .SetParent<Object> ()
    .SetGroupName ("OpenFlow")
    .AddConstructor<Controller> ()
    ;
  return tid;
}

Controller::~Controller ()
{
  m_switches.clear ();
}

void
Controller::AddSwitch (Ptr<OpenFlowSwitchNetDevice> swtch)
{
  if (m_switches.find (swtch) != m_switches.end ())
    {
      NS_LOG_INFO ("This Controller has already registered this switch!");
    }
  else
    {
      m_switches.insert (swtch);
    }
}

void
Controller::SendToSwitch (Ptr<OpenFlowSwitchNetDevice> swtch, void * msg, size_t length)
{
  if (m_switches.find (swtch) == m_switches.end ())
    {
      NS_LOG_ERROR ("Can't send to this switch, not registered to the Controller.");
      return;
    }

  swtch->ForwardControlInput (msg, length);
}

ofp_flow_mod*
Controller::BuildFlow (sw_flow_key key, uint32_t buffer_id, uint16_t command, void* acts, size_t actions_len, int idle_timeout, int hard_timeout)
{
  ofp_flow_mod* ofm = (ofp_flow_mod*)malloc (sizeof(ofp_flow_mod) + actions_len);
  ofm->header.version = OFP_VERSION;
  ofm->header.type = OFPT_FLOW_MOD;
  ofm->header.length = htons (sizeof(ofp_flow_mod) + actions_len);
  ofm->command = htons (command);
  ofm->idle_timeout = htons (idle_timeout);
  ofm->hard_timeout = htons (hard_timeout);
  ofm->buffer_id = htonl (buffer_id);
  ofm->priority = OFP_DEFAULT_PRIORITY;
  memcpy (ofm->actions,acts,actions_len);

  ofm->match.wildcards = key.wildcards;                                 // Wildcard fields
  ofm->match.in_port = key.flow.in_port;                                // Input switch port
  memcpy (ofm->match.dl_src, key.flow.dl_src, sizeof ofm->match.dl_src); // Ethernet source address.
  memcpy (ofm->match.dl_dst, key.flow.dl_dst, sizeof ofm->match.dl_dst); // Ethernet destination address.
  ofm->match.dl_vlan = key.flow.dl_vlan;                                // Input VLAN OFP_VLAN_NONE;
  ofm->match.dl_type = key.flow.dl_type;                                // Ethernet frame type ETH_TYPE_IP;
  ofm->match.nw_proto = key.flow.nw_proto;                              // IP Protocol
  ofm->match.nw_src = key.flow.nw_src;                                  // IP source address
  ofm->match.nw_dst = key.flow.nw_dst;                                  // IP destination address
  ofm->match.tp_src = key.flow.tp_src;                                  // TCP/UDP source port
  ofm->match.tp_dst = key.flow.tp_dst;                                  // TCP/UDP destination port
  ofm->match.mpls_label1 = key.flow.mpls_label1;                        // Top of label stack htonl(MPLS_INVALID_LABEL);
  ofm->match.mpls_label2 = key.flow.mpls_label1;                        // Second label (if available) htonl(MPLS_INVALID_LABEL);

  return ofm;
}

uint8_t
Controller::GetPacketType (ofpbuf* buffer)
{
  ofp_header* hdr = (ofp_header*)ofpbuf_try_pull (buffer, sizeof (ofp_header));
  uint8_t type = hdr->type;
  ofpbuf_push_uninit (buffer, sizeof (ofp_header));
  return type;
}

void
Controller::StartDump (StatsDumpCallback* cb)
{
  if (cb != 0)
    {
      int error = 1;
      while (error > 0) // Switch's StatsDump returns 1 if the reply isn't complete.
        {
          error = cb->swtch->StatsDump (cb);
        }
	
      if (error != 0) // When the reply is complete, error will equal zero if there's no errors.
        {
          NS_LOG_WARN ("Dump Callback Error: " << strerror (-error));
        }
	
      // Clean up
      cb->swtch->StatsDone (cb);
    }
}

/* static */
TypeId
DropController::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::ofi::DropController")
    .SetParent<Controller> ()
    .SetGroupName ("OpenFlow")
    .AddConstructor<DropController> ()
    ;
  return tid;
}

void
DropController::ReceiveFromSwitch (Ptr<OpenFlowSwitchNetDevice> swtch, ofpbuf* buffer)
{
  if (m_switches.find (swtch) == m_switches.end ())
    {
      NS_LOG_ERROR ("Can't receive from this switch, not registered to the Controller.");
      return;
    }

  // We have received any packet at this point, so we pull the header to figure out what type of packet we're handling.
  uint8_t type = GetPacketType (buffer);

  if (type == OFPT_PACKET_IN) // The switch didn't understand the packet it received, so it forwarded it to the controller.
    {
      ofp_packet_in * opi = (ofp_packet_in*)ofpbuf_try_pull (buffer, offsetof (ofp_packet_in, data));
      int port = ntohs (opi->in_port);

      // Create matching key.
      sw_flow_key key;
      key.wildcards = 0;
      flow_extract (buffer, port != -1 ? port : OFPP_NONE, &key.flow);

      ofp_flow_mod* ofm = BuildFlow (key, opi->buffer_id, OFPFC_ADD, 0, 0, OFP_FLOW_PERMANENT, OFP_FLOW_PERMANENT);
      SendToSwitch (swtch, ofm, ofm->header.length);
    }
}

TypeId LearningController::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::ofi::LearningController")
    .SetParent <Controller> ()
    .SetGroupName ("Openflow")
    .AddConstructor<LearningController> ()
    .AddAttribute ("ExpirationTime",
                   "Time it takes for learned MAC state entry/created flow to expire.",
                   TimeValue (Seconds (0)),
                   MakeTimeAccessor (&LearningController::m_expirationTime),
                   MakeTimeChecker ())
  ;
  return tid;
}

void
LearningController::ReceiveFromSwitch (Ptr<OpenFlowSwitchNetDevice> swtch, ofpbuf* buffer)
{
  if (m_switches.find (swtch) == m_switches.end ())
    {
      NS_LOG_ERROR ("Can't receive from this switch, not registered to the Controller.");
      return;
    }

  // We have received any packet at this point, so we pull the header to figure out what type of packet we're handling.
  uint8_t type = GetPacketType (buffer);

  if (type == OFPT_PACKET_IN) // The switch didn't understand the packet it received, so it forwarded it to the controller.
    {
      ofp_packet_in * opi = (ofp_packet_in*)ofpbuf_try_pull (buffer, offsetof (ofp_packet_in, data));
      int port = ntohs (opi->in_port);

      // Create matching key.
      sw_flow_key key;
      key.wildcards = 0;
      flow_extract (buffer, port != -1 ? port : OFPP_NONE, &key.flow);

      uint16_t out_port = OFPP_FLOOD;
      uint16_t in_port = ntohs (key.flow.in_port);

      // If the destination address is learned to a specific port, find it.
      Mac48Address dst_addr;
      dst_addr.CopyFrom (key.flow.dl_dst);
      if (!dst_addr.IsBroadcast ())
        {
          LearnState_t::iterator st = m_learnState.find (dst_addr);
          if (st != m_learnState.end ())
            {
              out_port = st->second.port;
            }
          else
            {
              NS_LOG_INFO ("Setting to flood; don't know yet what port " << dst_addr << " is connected to");
            }
        }
      else
        {
          NS_LOG_INFO ("Setting to flood; this packet is a broadcast");
        }

      // Create output-to-port action
      ofp_action_output x[1];
      x[0].type = htons (OFPAT_OUTPUT);
      x[0].len = htons (sizeof(ofp_action_output));
      x[0].port = out_port;

      // Create a new flow that outputs matched packets to a learned port, OFPP_FLOOD if there's no learned port.
      ofp_flow_mod* ofm = BuildFlow (key, opi->buffer_id, OFPFC_ADD, x, sizeof(x), OFP_FLOW_PERMANENT, m_expirationTime.IsZero () ? OFP_FLOW_PERMANENT : m_expirationTime.GetSeconds ());
      SendToSwitch (swtch, ofm, ofm->header.length);

      // We can learn a specific port for the source address for future use.
      Mac48Address src_addr;
      src_addr.CopyFrom (key.flow.dl_src);
      LearnState_t::iterator st = m_learnState.find (src_addr);
      if (st == m_learnState.end ()) // We haven't learned our source MAC yet.
        {
          LearnedState ls;
          ls.port = in_port;
          m_learnState.insert (std::make_pair (src_addr,ls));
          NS_LOG_INFO ("Learned that " << src_addr << " can be found over port " << in_port);

          // Learn src_addr goes to a certain port.
          ofp_action_output x2[1];
          x2[0].type = htons (OFPAT_OUTPUT);
          x2[0].len = htons (sizeof(ofp_action_output));
          x2[0].port = in_port;

          // Switch MAC Addresses and ports to the flow we're modifying
          src_addr.CopyTo (key.flow.dl_dst);
          dst_addr.CopyTo (key.flow.dl_src);
          key.flow.in_port = out_port;
          ofp_flow_mod* ofm2 = BuildFlow (key, -1, OFPFC_MODIFY, x2, sizeof(x2), OFP_FLOW_PERMANENT, m_expirationTime.IsZero () ? OFP_FLOW_PERMANENT : m_expirationTime.GetSeconds ());
          SendToSwitch (swtch, ofm2, ofm2->header.length);
        }
    }
}

void
ExecuteActions (Ptr<OpenFlowSwitchNetDevice> swtch, uint64_t packet_uid, ofpbuf* buffer, sw_flow_key *key, const ofp_action_header *actions, size_t actions_len, int ignore_no_fwd)
{
  NS_LOG_FUNCTION_NOARGS ();
  /* Every output action needs a separate clone of 'buffer', but the common
   * case is just a single output action, so that doing a clone and then
   * freeing the original buffer is wasteful.  So the following code is
   * slightly obscure just to avoid that. */
  int prev_port;
  size_t max_len = 0;     // Initialize to make compiler happy
  uint16_t in_port = key->flow.in_port; // ntohs(key->flow.in_port);
  uint8_t *p = (uint8_t *)actions;

  prev_port = -1;

  if (actions_len == 0)
    {
      NS_LOG_INFO ("No actions set to this flow. Dropping packet.");
      return;
    }

  /* The action list was already validated, so we can be a bit looser
   * in our sanity-checking. */
  while (actions_len > 0)
    {
      ofp_action_header *ah = (ofp_action_header *)p;
      size_t len = htons (ah->len);

      if (prev_port != -1)
        {
          swtch->DoOutput (packet_uid, in_port, max_len, prev_port, ignore_no_fwd);
          prev_port = -1;
        }

      if (ah->type == htons (OFPAT_OUTPUT))
        {
          ofp_action_output *oa = (ofp_action_output *)p;

          // port is now 32-bits
          prev_port = oa->port; // ntohl(oa->port);
          // prev_port = ntohs(oa->port);
          max_len = ntohs (oa->max_len);
        }
      else
        {
          uint16_t type = ntohs (ah->type);
          if (Action::IsValidType ((ofp_action_type)type)) // Execute a built-in OpenFlow action against 'buffer'.
            {
              Action::Execute ((ofp_action_type)type, buffer, key, ah);
            }
          else if (type == OFPAT_VENDOR)
            {
              ExecuteVendor (buffer, key, ah);
            }
        }

      p += len;
      actions_len -= len;
    }

  if (prev_port != -1)
    {
      swtch->DoOutput (packet_uid, in_port, max_len, prev_port, ignore_no_fwd);
    }
}

uint16_t
ValidateActions (const sw_flow_key *key, const ofp_action_header *actions, size_t actions_len)
{
  uint8_t *p = (uint8_t *)actions;
  int err;

  while (actions_len >= sizeof(ofp_action_header))
    {
      ofp_action_header *ah = (ofp_action_header *)p;
      size_t len = ntohs (ah->len);
      uint16_t type;

      /* Make there's enough remaining data for the specified length
        * and that the action length is a multiple of 64 bits. */
      if ((actions_len < len) || (len % 8) != 0)
        {
          return OFPBAC_BAD_LEN;
        }

      type = ntohs (ah->type);
      if (Action::IsValidType ((ofp_action_type)type)) // Validate built-in OpenFlow actions.
        {
          err = Action::Validate ((ofp_action_type)type, len, key, ah);
          if (err != ACT_VALIDATION_OK)
            {
              return err;
            }
        }
      else if (type == OFPAT_VENDOR)
        {
          err = ValidateVendor (key, ah, len);
          if (err != ACT_VALIDATION_OK)
            {
              return err;
            }
        }
      else
        {
          return OFPBAC_BAD_TYPE;
        }

      p += len;
      actions_len -= len;
    }

  // Check if there's any trailing garbage.
  if (actions_len != 0)
    {
      return OFPBAC_BAD_LEN;
    }

  return ACT_VALIDATION_OK;
}

void
ExecuteVPortActions (Ptr<OpenFlowSwitchNetDevice> swtch, uint64_t packet_uid, ofpbuf* buffer, sw_flow_key *key, const ofp_action_header *actions, size_t actions_len)
{
  /* Every output action needs a separate clone of 'buffer', but the common
   * case is just a single output action, so that doing a clone and then
   * freeing the original buffer is wasteful.  So the following code is
   * slightly obscure just to avoid that. */
  int prev_port;
  size_t max_len = 0;     // Initialize to make compiler happy
  uint16_t in_port = ntohs (key->flow.in_port);
  uint8_t *p = (uint8_t *)actions;
  uint16_t type;
  ofp_action_output *oa;

  prev_port = -1;
  /* The action list was already validated, so we can be a bit looser
   * in our sanity-checking. */
  while (actions_len > 0)
    {
      ofp_action_header *ah = (ofp_action_header *)p;
      size_t len = htons (ah->len);
      if (prev_port != -1)
        {
          swtch->DoOutput (packet_uid, in_port, max_len, prev_port, false);
          prev_port = -1;
        }

      if (ah->type == htons (OFPAT_OUTPUT))
        {
          oa = (ofp_action_output *)p;
          prev_port = ntohl (oa->port);
          max_len = ntohs (oa->max_len);
        }
      else
        {
          type = ah->type; // ntohs(ah->type);
          VPortAction::Execute ((ofp_vport_action_type)type, buffer, key, ah);
        }

      p += len;
      actions_len -= len;
    }

  if (prev_port != -1)
    {
      swtch->DoOutput (packet_uid, in_port, max_len, prev_port, false);
    }
}

uint16_t
ValidateVPortActions (const ofp_action_header *actions, size_t actions_len)
{
  uint8_t *p = (uint8_t *)actions;
  int err;

  while (actions_len >= sizeof(ofp_action_header))
    {
      ofp_action_header *ah = (ofp_action_header *)p;
      size_t len = ntohs (ah->len);
      uint16_t type;

      /* Make there's enough remaining data for the specified length
        * and that the action length is a multiple of 64 bits. */
      if ((actions_len < len) || (len % 8) != 0)
        {
          return OFPBAC_BAD_LEN;
        }

      type = ntohs (ah->type);
      if (VPortAction::IsValidType ((ofp_vport_action_type)type)) // Validate "built-in" OpenFlow port table actions.
        {
          err = VPortAction::Validate ((ofp_vport_action_type)type, len, ah);
          if (err != ACT_VALIDATION_OK)
            {
              return err;
            }
        }
      else
        {
          return OFPBAC_BAD_TYPE;
        }

      p += len;
      actions_len -= len;
    }

  // Check if there's any trailing garbage.
  if (actions_len != 0)
    {
      return OFPBAC_BAD_LEN;
    }

  return ACT_VALIDATION_OK;
}

void
ExecuteVendor (ofpbuf *buffer, const sw_flow_key *key, const ofp_action_header *ah)
{
  ofp_action_vendor_header *avh = (ofp_action_vendor_header *)ah;

  switch (ntohl (avh->vendor))
    {
    case NX_VENDOR_ID:
      // Nothing to execute yet.
      break;
    case ER_VENDOR_ID:
      {
        const er_action_header *erah = (const er_action_header *)avh;
        EricssonAction::Execute ((er_action_type)ntohs (erah->subtype), buffer, key, erah);
        break;
      }
    default:
      // This should not be possible due to prior validation.
      NS_LOG_INFO ("attempt to execute action with unknown vendor: " << ntohl (avh->vendor));
      break;
    }
}

uint16_t
ValidateVendor (const sw_flow_key *key, const ofp_action_header *ah, uint16_t len)
{
  ofp_action_vendor_header *avh;
  int ret = ACT_VALIDATION_OK;

  if (len < sizeof(ofp_action_vendor_header))
    {
      return OFPBAC_BAD_LEN;
    }

  avh = (ofp_action_vendor_header *)ah;

  switch (ntohl (avh->vendor))
    {
    case NX_VENDOR_ID:   // Validate Nicara OpenFlow actions.
      ret = OFPBAC_BAD_VENDOR_TYPE;   // Nothing to validate yet.
      break;
    case ER_VENDOR_ID:   // Validate Ericsson OpenFlow actions.
      {
        const er_action_header *erah = (const er_action_header *)avh;
        ret = EricssonAction::Validate ((er_action_type)ntohs (erah->subtype), len);
        break;
      }
    default:
      return OFPBAC_BAD_VENDOR;
    }

  return ret;
}

}

}

#endif // NS3_OPENFLOW
