#ifndef TRACE_FORMAT_H
#define TRACE_FORMAT_H
#include <stdint.h>
#include <cstdio>
#include <cassert>
#include <vector>
#include <stddef.h>
#include <cstring>

namespace ns3{

enum Event{
	Recv = 0,
	Enqu = 1,
	Dequ = 2,
	Drop = 3
};

struct TraceFormat{
	uint64_t time;
	uint16_t node;
	uint8_t intf, qidx;
	uint32_t qlen;
	uint32_t sip, dip;
	uint16_t size;
	uint8_t l3Prot;
	uint8_t event;
	uint8_t ecn; // this is the ip ECN bits
	uint8_t nodeType; // 0: host, 1: switch
	union{
		struct {
			uint16_t sport, dport;
			uint32_t seq;
			uint64_t ts;
			uint16_t pg;
			uint16_t payload; // this does not include SeqTsHeader's size, diff from udp's payload size.
		} data;
		struct {
			uint16_t fid;
			uint8_t qIndex;
			uint8_t ecnBits; // this is the ECN bits in the CNP
			union{
				struct {
					uint16_t qfb;
					uint16_t total;
				};
				uint32_t seq;
			};
		} cnp;
		struct {
			uint16_t sport, dport;
			uint16_t flags;
			uint16_t pg;
			uint32_t seq;
			uint64_t ts;
		} ack;
		struct {
			uint32_t time;
			uint32_t qlen;
			uint8_t qIndex;
		} pfc;
		struct {
			uint16_t sport, dport;
		}qp;
	};

	void Serialize(FILE *file){
		fwrite(this, sizeof(TraceFormat), 1, file);
	}
	int Deserialize(FILE *file){
		int ret = fread(this, sizeof(TraceFormat), 1, file);
		return ret;
	}
};

static inline const char* EventToStr(enum Event e){
	switch (e){
		case Recv:
			return "Recv";
		case Enqu:
			return "Enqu";
		case Dequ:
			return "Dequ";
		case Drop:
			return "Drop";
		default:
			return "????";
	}
}

}
#endif
