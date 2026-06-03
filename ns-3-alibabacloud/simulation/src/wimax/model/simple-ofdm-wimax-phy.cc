/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 *  Copyright (c) 2007,2008, 2009 INRIA, UDcast
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
 *
 * Author: Mohamed Amine Ismail <amine.ismail@sophia.inria.fr>
 *                              <amine.ismail@udcast.com>
 */

#include "ns3/simulator.h"
#include "ns3/packet.h"
#include "ns3/node.h"
#include "ns3/uinteger.h"
#include "ns3/double.h"
#include "ns3/string.h"
#include "wimax-net-device.h"
#include "simple-ofdm-wimax-phy.h"
#include "wimax-channel.h"
#include "ns3/packet-burst.h"
#include "wimax-mac-header.h"
#include "simple-ofdm-wimax-channel.h"
#include "ns3/trace-source-accessor.h"
#include <string>
#include <cmath>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("SimpleOfdmWimaxPhy");

NS_OBJECT_ENSURE_REGISTERED (SimpleOfdmWimaxPhy);

TypeId SimpleOfdmWimaxPhy::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::SimpleOfdmWimaxPhy")
    .SetParent<WimaxPhy> ()
    .SetGroupName ("Wimax")
    
    .AddConstructor<SimpleOfdmWimaxPhy> ()

    .AddAttribute ("NoiseFigure",
                   "Loss (dB) in the Signal-to-Noise-Ratio due to non-idealities in the receiver.",
                   DoubleValue (5),
                   MakeDoubleAccessor (&SimpleOfdmWimaxPhy::SetNoiseFigure, &SimpleOfdmWimaxPhy::GetNoiseFigure),
                   MakeDoubleChecker<double> ())

    .AddAttribute ("TxPower",
                   "Transmission power (dB).",
                   DoubleValue (30),
                   MakeDoubleAccessor (&SimpleOfdmWimaxPhy::SetTxPower, &SimpleOfdmWimaxPhy::GetTxPower),
                   MakeDoubleChecker<double> ())

    .AddAttribute ("G",
                   "This is the ratio of CP time to useful time.",
                   DoubleValue (0.25),
                   MakeDoubleAccessor (&SimpleOfdmWimaxPhy::DoSetGValue, &SimpleOfdmWimaxPhy::DoGetGValue),
                   MakeDoubleChecker<double> ())

    .AddAttribute ("TxGain",
                   "Transmission gain (dB).",
                   DoubleValue (0),
                   MakeDoubleAccessor (&SimpleOfdmWimaxPhy::SetTxGain, &SimpleOfdmWimaxPhy::GetTxGain),
                   MakeDoubleChecker<double> ())

    .AddAttribute ("RxGain",
                   "Reception gain (dB).",
                   DoubleValue (0),
                   MakeDoubleAccessor (&SimpleOfdmWimaxPhy::SetRxGain, &SimpleOfdmWimaxPhy::GetRxGain),
                   MakeDoubleChecker<double> ())

    .AddAttribute ("Nfft",
                   "FFT size",
                   UintegerValue (256),
                   MakeUintegerAccessor (&SimpleOfdmWimaxPhy::DoSetNfft, &SimpleOfdmWimaxPhy::DoGetNfft),
                   MakeUintegerChecker<uint16_t> (256, 1024))

    .AddAttribute ("TraceFilePath",
                   "Path to the directory containing SNR to block error rate files",
                   StringValue (""),
                   MakeStringAccessor (&SimpleOfdmWimaxPhy::GetTraceFilePath,
                                       &SimpleOfdmWimaxPhy::SetTraceFilePath),
                   MakeStringChecker ())

    .AddTraceSource ("Rx", "Receive trace",
                     MakeTraceSourceAccessor (&SimpleOfdmWimaxPhy::m_traceRx),
                     "ns3::PacketBurst::TracedCallback")
    .AddTraceSource ("Tx", "Transmit trace",
                     MakeTraceSourceAccessor (&SimpleOfdmWimaxPhy::m_traceTx),
                     "ns3::PacketBurst::TracedCallback")

    .AddTraceSource ("PhyTxBegin",
                     "Trace source indicating a packet has begun transmitting over the channel medium",
                     MakeTraceSourceAccessor (&SimpleOfdmWimaxPhy::m_phyTxBeginTrace),
                     "ns3::PacketBurst::TracedCallback")

    .AddTraceSource ("PhyTxEnd",
                     "Trace source indicating a packet has been completely transmitted over the channel",
                     MakeTraceSourceAccessor (&SimpleOfdmWimaxPhy::m_phyTxEndTrace),
                     "ns3::PacketBurst::TracedCallback")

    .AddTraceSource ("PhyTxDrop",
                     "Trace source indicating a packet has been dropped by the device during transmission",
                     MakeTraceSourceAccessor (&SimpleOfdmWimaxPhy::m_phyTxDropTrace),
                     "ns3::PacketBurst::TracedCallback")

    .AddTraceSource ("PhyRxBegin",
                     "Trace source indicating a packet has begun being received from the channel medium by the device",
                     MakeTraceSourceAccessor (&SimpleOfdmWimaxPhy::m_phyRxBeginTrace),
                     "ns3::PacketBurst::TracedCallback")

    .AddTraceSource ("PhyRxEnd",
                     "Trace source indicating a packet has been completely received from the channel medium by the device",
                     MakeTraceSourceAccessor (&SimpleOfdmWimaxPhy::m_phyRxEndTrace),
                     "ns3::PacketBurst::TracedCallback")

    .AddTraceSource ("PhyRxDrop",
                     "Trace source indicating a packet has been dropped by the device during reception",
                     MakeTraceSourceAccessor (&SimpleOfdmWimaxPhy::m_phyRxDropTrace),
                     "ns3::PacketBurst::TracedCallback");
  return tid;
}

void
SimpleOfdmWimaxPhy::InitSimpleOfdmWimaxPhy (void)
{
  m_fecBlockSize = 0;
  m_nrFecBlocksSent = 0;
  m_dataRateBpsk12 = 0;
  m_dataRateQpsk12 = 0;
  m_dataRateQpsk34 = 0;
  m_dataRateQam16_12 = 0;

  m_dataRateQam16_34 = 0;
  m_dataRateQam64_23 = 0;
  m_dataRateQam64_34 = 0;

  m_nrBlocks = 0;
  m_blockSize = 0;
  m_paddingBits = 0;
  m_rxGain = 0;
  m_txGain = 0;
  m_nfft = 256;
  m_g = (double) 1 / 4;
  SetNrCarriers (192);
  m_fecBlocks = new std::list<bvec>;
  m_receivedFecBlocks = new std::list<bvec>;
  m_currentBurstSize = 0;
  m_noiseFigure = 5; // dB
  m_txPower = 30; // dBm
  SetBandwidth (10000000); // 10Mhz
  m_nbErroneousBlock = 0;
  m_nrRecivedFecBlocks = 0;
  m_snrToBlockErrorRateManager = new SNRToBlockErrorRateManager ();
}

SimpleOfdmWimaxPhy::SimpleOfdmWimaxPhy (void)
{
  m_URNG = CreateObject<UniformRandomVariable> ();

  InitSimpleOfdmWimaxPhy ();
  m_snrToBlockErrorRateManager->SetTraceFilePath ((char*) "");
  m_snrToBlockErrorRateManager->LoadTraces ();
}

SimpleOfdmWimaxPhy::SimpleOfdmWimaxPhy (char * tracesPath)
{
  InitSimpleOfdmWimaxPhy ();
  m_snrToBlockErrorRateManager->SetTraceFilePath (tracesPath);
  m_snrToBlockErrorRateManager->LoadTraces ();
}

SimpleOfdmWimaxPhy::~SimpleOfdmWimaxPhy (void)
{

}

void
SimpleOfdmWimaxPhy::ActivateLoss (bool loss)
{
  m_snrToBlockErrorRateManager->ActivateLoss (loss);
}

void
SimpleOfdmWimaxPhy::SetSNRToBlockErrorRateTracesPath (char * tracesPath)
{
  m_snrToBlockErrorRateManager->SetTraceFilePath (tracesPath);
  m_snrToBlockErrorRateManager->ReLoadTraces ();
}

uint32_t
SimpleOfdmWimaxPhy::GetBandwidth (void) const
{
  return WimaxPhy::GetChannelBandwidth ();
}

void
SimpleOfdmWimaxPhy::SetBandwidth (uint32_t BW)
{
  WimaxPhy::SetChannelBandwidth (BW);
}

double
SimpleOfdmWimaxPhy::GetTxPower (void) const
{
  return m_txPower;
}
void
SimpleOfdmWimaxPhy::SetTxPower (double txPower)
{
  m_txPower = txPower;
}

double
SimpleOfdmWimaxPhy::GetNoiseFigure (void) const
{
  return m_noiseFigure;
}
void
SimpleOfdmWimaxPhy::SetNoiseFigure (double noiseFigure)
{
  m_noiseFigure = noiseFigure;
}

void
SimpleOfdmWimaxPhy::DoDispose (void)
{
  delete m_receivedFecBlocks;
  delete m_fecBlocks;
  m_receivedFecBlocks = 0;
  m_fecBlocks = 0;
  delete m_snrToBlockErrorRateManager;
  WimaxPhy::DoDispose ();
}

void
SimpleOfdmWimaxPhy::DoAttach (Ptr<WimaxChannel> channel)
{
  GetChannel ()->Attach (this);
}

void
SimpleOfdmWimaxPhy::Send (SendParams *params)
{
  OfdmSendParams *o_params = dynamic_cast<OfdmSendParams*> (params);
  NS_ASSERT (o_params !=0);
  Send (o_params->GetBurst (),
        (WimaxPhy::ModulationType) o_params->GetModulationType (),
        o_params->GetDirection ());

}

WimaxPhy::PhyType
SimpleOfdmWimaxPhy::GetPhyType (void) const
{
  return WimaxPhy::simpleOfdmWimaxPhy;
}

void
SimpleOfdmWimaxPhy::Send (Ptr<PacketBurst> burst,
                          WimaxPhy::ModulationType modulationType,
                          uint8_t direction)
{

  if (GetState () != PHY_STATE_TX)
    {
      m_currentBurstSize = burst->GetSize ();
      m_nrFecBlocksSent = 0;
      m_currentBurst = burst;
      SetBlockParameters (burst->GetSize (), modulationType);
      NotifyTxBegin (m_currentBurst);
      StartSendDummyFecBlock (true, modulationType, direction);
      m_traceTx (burst);
    }
}

void
SimpleOfdmWimaxPhy::StartSendDummyFecBlock (bool isFirstBlock,
                                            WimaxPhy::ModulationType modulationType,
                                            uint8_t direction)
{
  SetState (PHY_STATE_TX);
  bool isLastFecBlock = 0;
  if (isFirstBlock)
    {
      m_blockTime = GetBlockTransmissionTime (modulationType);
    }

  SimpleOfdmWimaxChannel *channel = dynamic_cast<SimpleOfdmWimaxChannel*> (PeekPointer (GetChannel ()));
  NS_ASSERT (channel != 0);

  if (m_nrRemainingBlocksToSend==1)
    {
      isLastFecBlock = true;
    }
  else
    {
      isLastFecBlock = false;
    }
  channel->Send (m_blockTime,
                 m_currentBurstSize,
                 this,
                 isFirstBlock,
                 isLastFecBlock,
                 GetTxFrequency (),
                 modulationType,
                 direction,
                 m_txPower,
                 m_currentBurst);

  m_nrRemainingBlocksToSend--;
  Simulator::Schedule (m_blockTime, &SimpleOfdmWimaxPhy::EndSendFecBlock, this, modulationType, direction);
}


void
SimpleOfdmWimaxPhy::EndSendFecBlock (WimaxPhy::ModulationType modulationType,
                                     uint8_t direction)
{
  m_nrFecBlocksSent++;
  SetState (PHY_STATE_IDLE);

  if (m_nrFecBlocksSent * m_blockSize == m_currentBurstSize * 8 + m_paddingBits)
    {
      // this is the last FEC block of the burst
      NS_ASSERT_MSG (m_nrRemainingBlocksToSend == 0, "Error while sending a burst");
      NotifyTxEnd (m_currentBurst);
    }
  else
    {
      StartSendDummyFecBlock (false,modulationType,direction);
    }
}

void
SimpleOfdmWimaxPhy::EndSend (void)
{
  SetState (PHY_STATE_IDLE);
}

void
SimpleOfdmWimaxPhy::StartReceive (uint32_t burstSize,
                                  bool isFirstBlock,
                                  uint64_t frequency,
                                  WimaxPhy::ModulationType modulationType,
                                  uint8_t direction,
                                  double rxPower,
                                  Ptr<PacketBurst> burst)
{

  uint8_t drop = 0;
  double Nwb = -114 + m_noiseFigure + 10 * std::log (GetBandwidth () / 1000000000.0) / 2.303;
  double SNR = rxPower - Nwb;

  SNRToBlockErrorRateRecord * record = m_snrToBlockErrorRateManager->GetSNRToBlockErrorRateRecord (SNR, modulationType);
  double I1 = record->GetI1 ();
  double I2 = record->GetI2 ();

  double blockErrorRate = m_URNG->GetValue (I1, I2);

  double rand = m_URNG->GetValue (0.0, 1.0);

  if (rand < blockErrorRate)
    {
      drop = 1;
    }
  if (rand > blockErrorRate)
    {
      drop = 0;
    }

  if (blockErrorRate == 1.0)
    {
      drop = 1;
    }
  if (blockErrorRate == 0.0)
    {
      drop = 0;
    }
  delete record;

  NS_LOG_INFO ("PHY: Receive rxPower=" << rxPower << ", Nwb=" << Nwb << ", SNR=" << SNR << ", Modulation="
                                       << modulationType << ", BlocErrorRate=" << blockErrorRate << ", drop=" << (int) drop);

  switch (GetState ())
    {
    case PHY_STATE_SCANNING:
      if (frequency == GetScanningFrequency ())
        {
          Simulator::Cancel (GetChnlSrchTimeoutEvent ());
          SetScanningCallback ();
          SetSimplex (frequency);
          SetState (PHY_STATE_IDLE);
        }
      break;
    case PHY_STATE_IDLE:
      if (frequency == GetRxFrequency ())
        {
          if (isFirstBlock)
            {
              NotifyRxBegin (burst);
              m_receivedFecBlocks->clear ();
              m_nrRecivedFecBlocks=0;
              SetBlockParameters (burstSize, modulationType);
              m_blockTime = GetBlockTransmissionTime (modulationType);
            }

          Simulator::Schedule (m_blockTime,
                               &SimpleOfdmWimaxPhy::EndReceiveFecBlock,
                               this,
                               burstSize,
                               modulationType,
                               direction,
                               drop,
                               burst);

          SetState (PHY_STATE_RX);
        }
      break;
    case PHY_STATE_RX:
      // drop
      break;
    case PHY_STATE_TX:
      if (IsDuplex () && frequency == GetRxFrequency ())
        {

        }
      break;
    }
}

void
SimpleOfdmWimaxPhy::EndReceiveFecBlock (uint32_t burstSize,
                                        WimaxPhy::ModulationType modulationType,
                                        uint8_t direction,
                                        uint8_t drop,
                                        Ptr<PacketBurst> burst)
{
  SetState (PHY_STATE_IDLE);
  m_nrRecivedFecBlocks++;

  if (drop == true)
    {
      m_nbErroneousBlock++;
    }

  if ((uint32_t) m_nrRecivedFecBlocks * m_blockSize == burstSize * 8 + m_paddingBits)
    {
      NotifyRxEnd (burst);
      if (m_nbErroneousBlock == 0)
        {
          Simulator::Schedule (Seconds (0),
                               &SimpleOfdmWimaxPhy::EndReceive,
                               this,
                               burst);
        }
      else
        {
          NotifyRxDrop (burst);
        }
      m_nbErroneousBlock = 0;
      m_nrRecivedFecBlocks = 0;
    }
}

void
SimpleOfdmWimaxPhy::EndReceive (Ptr<const PacketBurst> burst)
{
  Ptr<PacketBurst> b = burst->Copy ();
  GetReceiveCallback () (b);
  m_traceRx (burst);
}

bvec
SimpleOfdmWimaxPhy::ConvertBurstToBits (Ptr<const PacketBurst> burst)
{
  bvec buffer (burst->GetSize () * 8, 0);

  std::list<Ptr<Packet> > packets = burst->GetPackets ();

  uint32_t j = 0;
  for (std::list<Ptr<Packet> >::iterator iter = packets.begin (); iter != packets.end (); ++iter)
    {
      Ptr<Packet> packet = *iter;
      uint8_t *pstart = (uint8_t*) std::malloc (packet->GetSize ());
      std::memset (pstart, 0, packet->GetSize ());
      packet->CopyData (pstart, packet->GetSize ());
      bvec temp (8);
      temp.resize (0, 0);
      temp.resize (8, 0);
      for (uint32_t i = 0; i < packet->GetSize (); i++)
        {
          for (uint8_t l = 0; l < 8; l++)
            {
              temp[l] = (bool)((((uint8_t) pstart[i]) >> (7 - l)) & 0x01);
              buffer.at (j * 8 + l) = temp[l];
            }
          j++;
        }
      std::free (pstart);
    }

  return buffer;
}

/*
 Converts back the bit buffer (bvec) to the actual burst.
 Actually creates byte buffer from the bvec and resets the buffer
 of each packet in the copy of the orifinal burst stored before transmitting.
 By doing this it preserves the metadata and tags in the packet.
 Function could also be named DeserializeBurst because actually it
 copying to the burst's byte buffer.
 */
Ptr<PacketBurst>
SimpleOfdmWimaxPhy::ConvertBitsToBurst (bvec buffer)
{
  uint8_t init[buffer.size () / 8];
  uint8_t *pstart = init;
  uint8_t temp;
  int32_t j = 0;
  // recreating byte buffer from bit buffer (bvec)
  for (uint32_t i = 0; i < buffer.size (); i += 8)
    {

      temp = 0;
      for (int l = 0; l < 8; l++)
        {
          bool bin = buffer.at (i + l);
          temp += (uint8_t)(bin * std::pow (2.0, (7 - l)));
        }

      *(pstart + j) = temp;
      j++;
    }
  uint16_t bufferSize = buffer.size () / 8;
  uint16_t pos = 0;
  Ptr<PacketBurst> RecvBurst = Create<PacketBurst> ();
  while (pos < bufferSize)
    {
      uint16_t packetSize = 0;
      // Get the header type: first bit
      uint8_t ht = (pstart[pos] >> 7) & 0x01;
      if (ht == 1)
        {
          // BW request header. Size is always 8 bytes
          packetSize = 6;
        }
      else
        {
          // Read the size
          uint8_t Len_MSB = pstart[pos + 1] & 0x07;
          packetSize = (uint16_t)((uint16_t)(Len_MSB << 8) | (uint16_t)(pstart[pos + 2]));
          if (packetSize == 0)
            {
              break; // padding
            }
        }

      Ptr<Packet> p = Create<Packet> (&(pstart[pos]), packetSize);
      RecvBurst->AddPacket (p);
      pos += packetSize;
    }
  return RecvBurst;
}

void
SimpleOfdmWimaxPhy::CreateFecBlocks (const bvec &buffer, WimaxPhy::ModulationType modulationType)
{

  bvec fecBlock (m_blockSize);
  for (uint32_t i = 0, j = m_nrBlocks; j > 0; i += m_blockSize, j--)
    {

      if (j == 1 && m_paddingBits > 0) // last block can be smaller than block size
        {
          fecBlock = bvec (buffer.begin () + i, buffer.end ());
          fecBlock.resize (m_blockSize, 0);
        }
      else
        {
          fecBlock = bvec (buffer.begin () + i, buffer.begin () + i + m_blockSize);
        }

      m_fecBlocks->push_back (fecBlock);
    }
}

bvec
SimpleOfdmWimaxPhy::RecreateBuffer ()
{

  bvec buffer (m_blockSize * (unsigned long)m_nrBlocks);
  bvec block (m_blockSize);
  uint32_t i = 0;
  for (uint32_t j = 0; j < m_nrBlocks; j++)
    {
      bvec tmpRecFecBloc = m_receivedFecBlocks->front ();
      buffer.insert (buffer.begin () + i, tmpRecFecBloc.begin (), tmpRecFecBloc.end ());
      m_receivedFecBlocks->pop_front ();
      i += m_blockSize;
    }
  return buffer;
}

void
SimpleOfdmWimaxPhy::DoSetDataRates (void)
{
  m_dataRateBpsk12 = CalculateDataRate (MODULATION_TYPE_BPSK_12); // 6912000 bps
  m_dataRateQpsk12 = CalculateDataRate (MODULATION_TYPE_QPSK_12); // 13824000
  m_dataRateQpsk34 = CalculateDataRate (MODULATION_TYPE_QPSK_34); // 20736000
  m_dataRateQam16_12 = CalculateDataRate (MODULATION_TYPE_QAM16_12); // 27648000
  m_dataRateQam16_34 = CalculateDataRate (MODULATION_TYPE_QAM16_34); // 41472000
  m_dataRateQam64_23 = CalculateDataRate (MODULATION_TYPE_QAM64_23); // 55224000
  m_dataRateQam64_34 = CalculateDataRate (MODULATION_TYPE_QAM64_34); // 62208000
}

void
SimpleOfdmWimaxPhy::GetModulationFecParams (WimaxPhy::ModulationType modulationType,
                                            uint8_t &bitsPerSymbol,
                                            double &fecCode) const
{
  switch (modulationType)
    {
    case MODULATION_TYPE_BPSK_12:
      bitsPerSymbol = 1;
      fecCode = (double) 1 / 2;
      break;
    case MODULATION_TYPE_QPSK_12:
      bitsPerSymbol = 2;
      fecCode = (double) 1 / 2;
      break;
    case MODULATION_TYPE_QPSK_34:
      bitsPerSymbol = 2;
      fecCode = (double) 3 / 4;
      break;
    case MODULATION_TYPE_QAM16_12:
      bitsPerSymbol = 4;
      fecCode = (double) 1 / 2;
      break;
    case MODULATION_TYPE_QAM16_34:
      bitsPerSymbol = 4;
      fecCode = (double) 3 / 4;
      break;
    case MODULATION_TYPE_QAM64_23:
      bitsPerSymbol = 6;
      fecCode = (double) 2 / 3;
      break;
    case MODULATION_TYPE_QAM64_34:
      bitsPerSymbol = 6;
      fecCode = 0.75;
      break;
    }
}

uint32_t
SimpleOfdmWimaxPhy::CalculateDataRate (WimaxPhy::ModulationType modulationType) const
{
  uint8_t bitsPerSymbol = 0;
  double fecCode = 0;
  GetModulationFecParams (modulationType, bitsPerSymbol, fecCode);
  double symbolsPerSecond = 1 / GetSymbolDuration ().GetSeconds ();
  uint16_t bitsTransmittedPerSymbol = (uint16_t)(bitsPerSymbol * GetNrCarriers () * fecCode);
  // 96, 192, 288, 384, 576, 767 and 864 bits per symbol for the seven modulations, respectively

  return (uint32_t) symbolsPerSecond * bitsTransmittedPerSymbol;
}

uint32_t
SimpleOfdmWimaxPhy::DoGetDataRate (WimaxPhy::ModulationType modulationType) const
{
  switch (modulationType)
    {
    case MODULATION_TYPE_BPSK_12:
      return m_dataRateBpsk12;
      break;
    case MODULATION_TYPE_QPSK_12:
      return m_dataRateQpsk12;
      break;
    case MODULATION_TYPE_QPSK_34:
      return m_dataRateQpsk34;
      break;
    case MODULATION_TYPE_QAM16_12:
      return m_dataRateQam16_12;
      break;
    case MODULATION_TYPE_QAM16_34:
      return m_dataRateQam16_34;
      break;
    case MODULATION_TYPE_QAM64_23:
      return m_dataRateQam64_23;
      break;
    case MODULATION_TYPE_QAM64_34:
      return m_dataRateQam64_34;
      break;
    }
  NS_FATAL_ERROR ("Invalid modulation type");
  return 0;
}

Time
SimpleOfdmWimaxPhy::GetBlockTransmissionTime (WimaxPhy::ModulationType modulationType) const
{
  return Seconds ((double) GetFecBlockSize (modulationType) / DoGetDataRate (modulationType));
}

Time
SimpleOfdmWimaxPhy::DoGetTransmissionTime (uint32_t size, WimaxPhy::ModulationType modulationType) const
{
  /*adding 3 extra nano second to cope with the loss of precision problem.
   the time is internally stored in a 64 bit hence a floating-point time would loss
   precision, e.g., 0.00001388888888888889 seconds will become 13888888888 femtoseconds.*/
  return Seconds (DoGetNrSymbols (size, modulationType) * GetSymbolDuration ().GetSeconds ()) + NanoSeconds (3);
}

uint64_t
SimpleOfdmWimaxPhy::DoGetNrSymbols (uint32_t size, WimaxPhy::ModulationType modulationType) const
{
  Time transmissionTime = Seconds ((double)(GetNrBlocks (size, modulationType) * GetFecBlockSize (modulationType))
                                   / DoGetDataRate (modulationType));
  return (uint64_t) std::ceil (transmissionTime.GetSeconds () / GetSymbolDuration ().GetSeconds ());
}

uint64_t
SimpleOfdmWimaxPhy::DoGetNrBytes (uint32_t symbols, WimaxPhy::ModulationType modulationType) const
{
  Time transmissionTime = Seconds (symbols * GetSymbolDuration ().GetSeconds ());
  return (uint64_t) std::floor ((transmissionTime.GetSeconds () * DoGetDataRate (modulationType)) / 8);
}

uint32_t
SimpleOfdmWimaxPhy::GetFecBlockSize (WimaxPhy::ModulationType modulationType) const
{
  uint32_t blockSize = 0;
  switch (modulationType)
    {
    case MODULATION_TYPE_BPSK_12:
      blockSize = 12;
      break;
    case MODULATION_TYPE_QPSK_12:
      blockSize = 24;
      break;
    case MODULATION_TYPE_QPSK_34:
      blockSize = 36;
      break;
    case MODULATION_TYPE_QAM16_12:
      blockSize = 48;
      break;
    case MODULATION_TYPE_QAM16_34:
      blockSize = 72;
      break;
    case MODULATION_TYPE_QAM64_23:
      blockSize = 96;
      break;
    case MODULATION_TYPE_QAM64_34:
      blockSize = 108;
      break;
    default:
      NS_FATAL_ERROR ("Invalid modulation type");
      break;
    }
  return blockSize * 8; // in bits
}

// Channel coding block size, Table 215, page 434
uint32_t
SimpleOfdmWimaxPhy::GetCodedFecBlockSize (WimaxPhy::ModulationType modulationType) const
{
  uint32_t blockSize = 0;
  switch (modulationType)
    {
    case MODULATION_TYPE_BPSK_12:
      blockSize = 24;
      break;
    case MODULATION_TYPE_QPSK_12:
      blockSize = 48;
      break;
    case MODULATION_TYPE_QPSK_34:
      blockSize = 48;
      break;
    case MODULATION_TYPE_QAM16_12:
      blockSize = 96;
      break;
    case MODULATION_TYPE_QAM16_34:
      blockSize = 96;
      break;
    case MODULATION_TYPE_QAM64_23:
      blockSize = 144;
      break;
    case MODULATION_TYPE_QAM64_34:
      blockSize = 144;
      break;
    default:
      NS_FATAL_ERROR ("Invalid modulation type");
      break;
    }
  return blockSize * 8; // in bits
}

void
SimpleOfdmWimaxPhy::SetBlockParameters (uint32_t burstSize, WimaxPhy::ModulationType modulationType)
{
  m_blockSize = GetFecBlockSize (modulationType);
  m_nrBlocks = GetNrBlocks (burstSize, modulationType);
  m_paddingBits = (m_nrBlocks * m_blockSize) - (burstSize * 8);
  m_nrRemainingBlocksToSend = m_nrBlocks;
  NS_ASSERT_MSG (static_cast<uint32_t> (m_nrBlocks * m_blockSize) >= (burstSize * 8), "Size of padding bytes < 0");
}

uint16_t
SimpleOfdmWimaxPhy::DoGetTtg (void) const
{
  // assumed equal to 2 symbols
  return 2 * GetPsPerSymbol ();
}

uint16_t
SimpleOfdmWimaxPhy::DoGetRtg (void) const
{
  // assumed equal to 2 symbols
  return 2 * GetPsPerSymbol ();
}

uint8_t
SimpleOfdmWimaxPhy::DoGetFrameDurationCode (void) const
{
  uint16_t duration = 0;
  duration = (uint16_t)(GetFrameDuration ().GetSeconds () * 10000);
  uint8_t retval = 0;
  switch (duration)
    {
    case 25:
      {
        retval = FRAME_DURATION_2_POINT_5_MS;
        break;
      }
    case 40:
      {
        retval = FRAME_DURATION_4_MS;
        break;
      }
    case 50:
      {
        retval = FRAME_DURATION_5_MS;
        break;
      }
    case 80:
      {
        retval = FRAME_DURATION_8_MS;
        break;
      }
    case 100:
      {
        retval = FRAME_DURATION_10_MS;
        break;
      }
    case 125:
      {
        retval = FRAME_DURATION_12_POINT_5_MS;
        break;
      }
    case 200:
      {
        retval = FRAME_DURATION_20_MS;
        break;
      }
    default:
      {
        NS_FATAL_ERROR ("Invalid frame duration = " << duration);
        retval = 0;
      }
    }
  return retval;
}

Time
SimpleOfdmWimaxPhy::DoGetFrameDuration (uint8_t frameDurationCode) const
{
  switch (frameDurationCode)
    {
    case FRAME_DURATION_2_POINT_5_MS:
      return Seconds (2.5);
      break;
    case FRAME_DURATION_4_MS:
      return Seconds (4);
      break;
    case FRAME_DURATION_5_MS:
      return Seconds (5);
      break;
    case FRAME_DURATION_8_MS:
      return Seconds (8);
      break;
    case FRAME_DURATION_10_MS:
      return Seconds (10);
      break;
    case FRAME_DURATION_12_POINT_5_MS:
      return Seconds (12.5);
      break;
    case FRAME_DURATION_20_MS:
      return Seconds (20);
      break;
    default:
      NS_FATAL_ERROR ("Invalid modulation type");
    }
  return Seconds (0);
}

/*
 Retruns number of blocks (FEC blocks) the burst will be split in.
 The size of the block is specific for each modulation type.
 */
uint16_t
SimpleOfdmWimaxPhy::GetNrBlocks (uint32_t burstSize, WimaxPhy::ModulationType modulationType) const
{
  uint32_t blockSize = GetFecBlockSize (modulationType);
  uint16_t nrBlocks = (burstSize * 8) / blockSize;

  if ((burstSize * 8) % blockSize > 0)
    {
      nrBlocks += 1;
    }

  return nrBlocks;
}
/*---------------------PHY parameters functions-----------------------*/

void
SimpleOfdmWimaxPhy::DoSetPhyParameters (void)
{
  /*Calculations as per section 8.3.2.
   Currently assuming license-exempt 5 GHz band. For channel bandwidth 20 MHz (Table B.28, page 812) and frame duration 10 ms
   (Table 232, page 460) i.e, 100 frames per second, sampling frequency is 23040000, symbol (OFDM symbol) duration is
   1.388888888888889e-05 seconds, PS duration is 1.7361111111111112e-07 seconds. Hence PSs per frame is 57600, symbols per frame
   is 720 and PSs per symbol is 80. Note that defining these parameters (symbol and PS duration) as Time may not result in exaclty
   these values therefore lrint has been used (otherwise should be defined as double).
   For licensed bands set channel bandwidth according to Table B.26, page 810.*/

  double samplingFrequency = DoGetSamplingFrequency ();
  Time psDuration = Seconds ((double) 4 / samplingFrequency);

  SetPsDuration (psDuration);
  uint16_t psPerFrame = (uint16_t)(GetFrameDuration ().GetSeconds () / psDuration.GetSeconds ());
  SetPsPerFrame (psPerFrame);
  double subcarrierSpacing = samplingFrequency / DoGetNfft ();
  double tb = (double) 1 / subcarrierSpacing; // Tb (useful symbol time)
  double tg = DoGetGValue () * tb; // Tg (cyclic prefix time)
  Time symbolDuration = Seconds (tb + tg); // OFDM Symbol Time
  SetSymbolDuration (symbolDuration);
  uint16_t psPerSymbol = lrint (symbolDuration.GetSeconds () / psDuration.GetSeconds ());
  SetPsPerSymbol (psPerSymbol);
  uint32_t symbolsPerFrame = lrint (GetFrameDuration ().GetSeconds () / symbolDuration.GetSeconds ());
  SetSymbolsPerFrame (symbolsPerFrame);
}

void
SimpleOfdmWimaxPhy::DoSetNfft (uint16_t nfft)
{
  m_nfft = nfft;

}

uint16_t
SimpleOfdmWimaxPhy::DoGetNfft (void) const
{
  return m_nfft;

}

double
SimpleOfdmWimaxPhy::DoGetSamplingFactor (void) const
{
  // sampling factor (n), see Table 213, page 429

  uint32_t channelBandwidth = GetChannelBandwidth ();

  if (channelBandwidth % 1750000 == 0)
    {
      return (double) 8 / 7;
    }
  else if (channelBandwidth % 1500000 == 0)
    {
      return (double) 86 / 75;
    }
  else if (channelBandwidth % 1250000 == 0)
    {
      return (double) 144 / 125;
    }
  else if (channelBandwidth % 2750000 == 0)
    {
      return (double) 316 / 275;
    }
  else if (channelBandwidth % 2000000 == 0)
    {
      return (double) 57 / 50;
    }
  else
    {
      NS_LOG_DEBUG ("Oops may be wrong channel bandwidth for OFDM PHY!");
      NS_FATAL_ERROR ("wrong channel bandwidth for OFDM PHY");
    }

  return (double) 8 / 7;
}

double
SimpleOfdmWimaxPhy::DoGetSamplingFrequency (void) const
{
  // sampling frequency (Fs), see 8.3.2.2

  return (DoGetSamplingFactor () * GetChannelBandwidth () / 8000) * 8000;
}

double
SimpleOfdmWimaxPhy::DoGetGValue (void) const
{

  return m_g;
}

void
SimpleOfdmWimaxPhy::DoSetGValue (double g)
{
  m_g = g;

}

void
SimpleOfdmWimaxPhy::SetTxGain (double txGain)
{
  m_txGain = txGain;
}

void
SimpleOfdmWimaxPhy::SetRxGain (double txRain)
{
  m_rxGain = txRain;
}

double
SimpleOfdmWimaxPhy::GetTxGain (void) const
{
  return m_txGain;
}

double
SimpleOfdmWimaxPhy::GetRxGain (void) const
{
  return m_rxGain;
}

std::string
SimpleOfdmWimaxPhy::GetTraceFilePath (void) const
{
  return (m_snrToBlockErrorRateManager->GetTraceFilePath ());
}

void
SimpleOfdmWimaxPhy::SetTraceFilePath (std::string path)
{

  m_snrToBlockErrorRateManager->SetTraceFilePath ((char*) path.c_str ());
  m_snrToBlockErrorRateManager->LoadTraces ();
}

void
SimpleOfdmWimaxPhy::NotifyTxBegin (Ptr<PacketBurst> burst)
{
  m_phyTxBeginTrace (burst);
}

void
SimpleOfdmWimaxPhy::NotifyTxEnd (Ptr<PacketBurst> burst)
{
  m_phyTxEndTrace (burst);
}

void
SimpleOfdmWimaxPhy::NotifyTxDrop (Ptr<PacketBurst> burst)
{
  m_phyTxDropTrace (burst);
}

void
SimpleOfdmWimaxPhy::NotifyRxBegin (Ptr<PacketBurst> burst)
{
  m_phyRxBeginTrace (burst);
}

void
SimpleOfdmWimaxPhy::NotifyRxEnd (Ptr<PacketBurst> burst)
{
  m_phyRxEndTrace (burst);
}

void
SimpleOfdmWimaxPhy::NotifyRxDrop (Ptr<PacketBurst> burst)
{
  m_phyRxDropTrace (burst);
}

int64_t
SimpleOfdmWimaxPhy::AssignStreams (int64_t stream)
{
  NS_LOG_FUNCTION (this << stream);
  m_URNG->SetStream (stream);
  return 1;
}

} // namespace ns3
