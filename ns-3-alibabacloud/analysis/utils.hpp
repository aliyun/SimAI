#ifndef UTILS_HPP
#define UTILS_HPP

#include "trace-format.h"

typedef uint64_t FlowInt;

static uint32_t GetDevInt(uint16_t node, uint8_t intf){
	return ((uint32_t)node << 8) | intf;
}
struct Device{
	uint16_t node;
	uint8_t intf;

	Device(uint16_t _node, uint8_t _intf): node(_node), intf(_intf) {}
	uint32_t GetDevInt(){
		return ::GetDevInt(node, intf);
	}
};

static inline bool IsFlow(ns3::TraceFormat &tr){
	return tr.l3Prot == 0x6 || tr.l3Prot == 0x11 || tr.l3Prot == 0xFC || tr.l3Prot == 0xFD || tr.l3Prot == 0x0;
}

static inline FlowInt GetFlowInt(uint32_t sip, uint32_t dip, uint16_t sport, uint16_t dport){
	FlowInt res;
	uint64_t src = (sip >> 8) & 0xffff, dst = (dip >> 8) & 0xffff;
	res = ((src << 16) | dst);
	res = (((res << 16) | sport) << 16) | dport;
	return res;
}
static inline FlowInt GetFlowInt(ns3::TraceFormat &tr){
	switch (tr.l3Prot){
		case 0x6:
		case 0x11:
			return GetFlowInt(tr.sip, tr.dip, tr.data.sport, tr.data.dport);
		case 0xFC: // ACK
		case 0xFD: // NACK
			return GetFlowInt(tr.sip, tr.dip, tr.ack.sport, tr.ack.dport);
		case 0x0: // QpAv
			return GetFlowInt(tr.sip, tr.dip, tr.qp.sport, tr.qp.dport);
		default:
			return GetFlowInt(tr.sip, tr.dip, 0, 0);
	}
}
static inline FlowInt GetReverseFlowInt(ns3::TraceFormat &tr){
	switch (tr.l3Prot){
		case 0x6:
		case 0x11:
			return GetFlowInt(tr.dip, tr.sip, tr.data.dport, tr.data.sport);
		case 0xFC: // ACK
		case 0xFD: // NACK
			return GetFlowInt(tr.dip, tr.sip, tr.ack.dport, tr.ack.sport);
		case 0x0: // QpAv
			return GetFlowInt(tr.dip, tr.sip, tr.qp.dport, tr.qp.sport);
		default:
			return GetFlowInt(tr.dip, tr.sip, 0, 0);
	}
}

// Return the forward direction FlowInt for data, and reverse FlowInt for ACK
static inline FlowInt GetStandardFlowInt(ns3::TraceFormat &tr){
	if (tr.l3Prot == 0xFC || tr.l3Prot == 0xFD)
		return GetReverseFlowInt(tr);
	else
		return GetFlowInt(tr);
}

static inline char l3ProtToChar(uint8_t p){
	switch (p){
		case 0x6:
			return 'T';
		case 0x11:
			return 'U';
		case 0xFC: // ACK
			return 'A';
		case 0xFD: // NACK
			return 'N';
		case 0xFE: // PFC
			return 'P';
		case 0xFF:
			return 'C';
		default:
			return 'X';
	}
}

static inline void print_trace(ns3::TraceFormat &tr){
	switch (tr.l3Prot){
		case 0x6:
		case 0x11:
			printf("%lu n:%u %u:%u %u %s ecn:%x %08x %08x %hu %hu %c %u %lu %u %hu(%hu)", tr.time, tr.node, tr.intf, tr.qidx, tr.qlen, EventToStr((ns3::Event)tr.event), tr.ecn, tr.sip, tr.dip, tr.data.sport, tr.data.dport, l3ProtToChar(tr.l3Prot), tr.data.seq, tr.data.ts, tr.data.pg, tr.size, tr.data.payload);
			break;
		case 0xFC: // ACK
			printf("%lu n:%u %u:%u %u %s ecn:%x %08x %08x %u %u %c 0x%02X %u %u %lu %hu", tr.time, tr.node, tr.intf, tr.qidx, tr.qlen, EventToStr((ns3::Event)tr.event), tr.ecn, tr.sip, tr.dip, tr.ack.sport, tr.ack.dport, l3ProtToChar(tr.l3Prot), tr.ack.flags, tr.ack.pg, tr.ack.seq, tr.ack.ts, tr.size);
			break;
		case 0xFD: // NACK
			printf("%lu n:%u %u:%u %u %s ecn:%x %08x %08x %u %u %c 0x%02X %u %u %lu %hu", tr.time, tr.node, tr.intf, tr.qidx, tr.qlen, EventToStr((ns3::Event)tr.event), tr.ecn, tr.sip, tr.dip, tr.ack.sport, tr.ack.dport, l3ProtToChar(tr.l3Prot), tr.ack.flags, tr.ack.pg, tr.ack.seq, tr.ack.ts, tr.size);
			break;
		case 0xFE: // PFC
			printf("%lu n:%u %u:%u %u %s ecn:%x %08x %08x %c %u %u %u %hu", tr.time, tr.node, tr.intf, tr.qidx, tr.qlen, EventToStr((ns3::Event)tr.event), tr.ecn, tr.sip, tr.dip, l3ProtToChar(tr.l3Prot), tr.pfc.time, tr.pfc.qlen, tr.pfc.qIndex, tr.size);
			break;
		case 0xFF: // CNP
			printf("%lu n:%u %u:%u %u %s ecn:%x %08x %08x %c %u %u %u %u %u", tr.time, tr.node, tr.intf, tr.qidx, tr.qlen, EventToStr((ns3::Event)tr.event), tr.ecn, tr.sip, tr.dip, l3ProtToChar(tr.l3Prot), tr.cnp.fid, tr.cnp.qIndex, tr.cnp.ecnBits, tr.cnp.seq, tr.size);
			break;
		case 0x0: // QpAv
			printf("%lu n:%u %u:%u %s %08x %08x %u %u", tr.time, tr.node, tr.intf, tr.qidx, EventToStr((ns3::Event)tr.event), tr.sip, tr.dip, tr.qp.sport, tr.qp.dport);
			break;
		default:
			printf("%lu n:%u %u:%u %u %s ecn:%x %08x %08x %x %u", tr.time, tr.node, tr.intf, tr.qidx, tr.qlen, EventToStr((ns3::Event)tr.event), tr.ecn, tr.sip, tr.dip, tr.l3Prot, tr.size);
			break;
	}
	printf("\n");
}

#endif /* UTILS_HPP */
