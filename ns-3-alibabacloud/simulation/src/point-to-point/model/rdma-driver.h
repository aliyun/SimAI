#ifndef RDMA_DRIVER_H
#define RDMA_DRIVER_H

#include <ns3/node.h>
#include <ns3/qbb-net-device.h>
#include <ns3/rdma.h>
#include <ns3/rdma-queue-pair.h>
#include <ns3/rdma-hw.h>
#include <vector>
#include <unordered_map>

namespace ns3 {

class RdmaDriver : public Object {
public:
	Ptr<Node> m_node;
	Ptr<RdmaHw> m_rdma;

	// trace
	TracedCallback<Ptr<RdmaQueuePair> > m_traceQpComplete;
    TracedCallback<Ptr<RdmaQueuePair> > m_traceSendComplete;

    static TypeId GetTypeId (void);
	RdmaDriver();

	// This function init the m_nic according to the NetDevice
	// So this must be called after all NICs are installed
	void Init(void);

	// Set Node
	void SetNode(Ptr<Node> node);

	// Set RdmaHw
	void SetRdmaHw(Ptr<RdmaHw> rdma);

	// add a queue pair
	void AddQueuePair(uint32_t src, uint32_t dest, uint64_t tag, uint64_t size, uint16_t pg, Ipv4Address _sip, Ipv4Address _dip, uint16_t _sport, uint16_t _dport, uint32_t win, uint64_t baseRtt, Callback<void> notifyAppFinish, Callback<void> notifyAppSent);

	// enable NVLS
	void EnbaleNVLS();
	void DisableNVLS();
	
	// callback when qp completes
	void QpComplete(Ptr<RdmaQueuePair> q);
    void SendComplete(Ptr<RdmaQueuePair> q);
};

} // namespace ns3

#endif /* RDMA_DRIVER_H */
