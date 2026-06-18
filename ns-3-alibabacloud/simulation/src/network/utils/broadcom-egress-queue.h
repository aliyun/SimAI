/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
* Copyright (c) 2006 Georgia Tech Research Corporation, INRIA
*
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
*/

#ifndef BROADCOM_EGRESS_H
#define BROADCOM_EGRESS_H

#include <queue>
#include "ns3/packet.h"
#include "queue.h"
#include "drop-tail-queue.h"
#include "ns3/point-to-point-net-device.h"
#include "ns3/event-id.h"
#include "ns3/ptr.h"
#include "packet-queue.h"
#include "simple-drop-tail-queue.h"

namespace ns3 {

	class TraceContainer;

	class BEgressQueue : public PacketQueue
	{
	public:
		static TypeId GetTypeId(void);
		static const unsigned fCnt = 128; //max number of queues, 128 for NICs
		static const unsigned qCnt = 8; //max number of queues, 8 for switches
		BEgressQueue();
		virtual ~BEgressQueue();
		bool Enqueue(Ptr<Packet> p, uint32_t qIndex);
		Ptr<Packet> DequeueRR(bool paused[]);
		uint32_t GetNBytes(uint32_t qIndex) const;
		uint32_t GetNBytesTotal() const;
		uint32_t GetLastQueue();

		TracedCallback<Ptr<const Packet>, uint32_t> m_traceBeqEnqueue;
		TracedCallback<Ptr<const Packet>, uint32_t> m_traceBeqDequeue;

	private:
		  /**
   		  * Place a packet into the rear of the Queue
   		  * \param p packet to enqueue
   		  * \return True if the operation was successful; false otherwise
   		  */  
		 bool Enqueue (Ptr<Packet> p); 
		 /**
   		 * Remove a packet from the front of the Queue
   		 * \return 0 if the operation was not successful; the packet otherwise.
   		 */
  		 Ptr<Packet> Dequeue (void);
  		 /**
   		 * Get a copy of the item at the front of the queue without removing it
   		 * \return 0 if the operation was not successful; the packet otherwise.
   		 */
  		Ptr<const Packet> Peek (void) const;
		bool DoEnqueue(Ptr<Packet> p, uint32_t qIndex);
		Ptr<Packet> DoDequeueRR(bool paused[]);
		//for compatibility
		virtual bool DoEnqueue(Ptr<Packet> p);
		virtual Ptr<Packet> DoDequeue(void);
		virtual Ptr<const Packet> DoPeek(void) const;
		double m_maxBytes; //total bytes limit
		uint32_t m_bytesInQueue[fCnt];
		uint32_t m_bytesInQueueTotal;
		uint32_t m_rrlast;
		uint32_t m_qlast;
		std::vector<Ptr<SimpleDropTailQueue>> m_queues; // uc queues
		
		TracedCallback<Ptr<const Packet> > m_traceEnqueue;
  		TracedCallback<Ptr<const Packet> > m_traceDequeue;
  		TracedCallback<Ptr<const Packet> > m_traceDrop;

  		uint32_t m_nBytes;
  		uint32_t m_nTotalReceivedBytes;
  		uint32_t m_nPackets;
  		uint32_t m_nTotalReceivedPackets;
  		uint32_t m_nTotalDroppedBytes;
  		uint32_t m_nTotalDroppedPackets;

		NS_LOG_TEMPLATE_DECLARE;
			

	};
/*
bool
BEgressQueue::Enqueue (Ptr<Packet> p)
{
  //NS_LOG_FUNCTION (this << p);

  //
  // If DoEnqueue fails, Queue::Drop is called by the subclass
  //
  bool retval = DoEnqueue (p);
  if (retval)
    {
     // NS_LOG_LOGIC ("m_traceEnqueue (p)");
      m_traceEnqueue (p);

      uint32_t size = p->GetSize ();
      m_nBytes += size;
      m_nTotalReceivedBytes += size;

      m_nPackets++;
      m_nTotalReceivedPackets++;
    }
  return retval;
}

Ptr<Packet>
BEgressQueue::Dequeue (void)
{
  //NS_LOG_FUNCTION (this);

        			//printf("dequeue!!\n");
			//fflush(stdout);

  Ptr<Packet> packet = DoDequeue ();

  if (packet != 0)
    {
      NS_ASSERT (m_nBytes >= packet->GetSize ());
      NS_ASSERT (m_nPackets > 0);

      m_nBytes -= packet->GetSize ();
      m_nPackets--;

   //   NS_LOG_LOGIC ("m_traceDequeue (packet)");
      m_traceDequeue (packet);
    }
  return packet;
}
*/
} // namespace ns3
#endif /* DROPTAIL_H */
