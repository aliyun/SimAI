/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2012 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
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
 * Author: Marco Miozzo  <marco.miozzo@cttc.es>
 */


#include <ns3/lte-harq-phy.h>
#include <ns3/log.h>
#include <ns3/assert.h>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("LteHarqPhy");

//NS_OBJECT_ENSURE_REGISTERED (LteHarqPhy)
//  ;


LteHarqPhy::LteHarqPhy ()
{
  // Create DL Decodification HARQ buffers
  std::vector <HarqProcessInfoList_t> dlHarqLayer0;
  dlHarqLayer0.resize (8);
  std::vector <HarqProcessInfoList_t> dlHarqLayer1;
  dlHarqLayer1.resize (8);
  m_miDlHarqProcessesInfoMap.push_back (dlHarqLayer0);
  m_miDlHarqProcessesInfoMap.push_back (dlHarqLayer1);
}


LteHarqPhy::~LteHarqPhy ()
{
  m_miDlHarqProcessesInfoMap.clear ();
  m_miUlHarqProcessesInfoMap.clear ();
}


void
LteHarqPhy::SubframeIndication (uint32_t frameNo, uint32_t subframeNo)
{
  NS_LOG_FUNCTION (this);

  // left shift UL HARQ buffers
  std::map <uint16_t, std::vector <HarqProcessInfoList_t> >::iterator it;
  for (it = m_miUlHarqProcessesInfoMap.begin (); it != m_miUlHarqProcessesInfoMap.end (); it++)
    {
      (*it).second.erase ((*it).second.begin ());
      HarqProcessInfoList_t h;
      (*it).second.push_back (h);      
    }

}


double
LteHarqPhy::GetAccumulatedMiDl (uint8_t harqProcId, uint8_t layer)
{
  NS_LOG_FUNCTION (this << (uint32_t)harqProcId << (uint16_t)layer);
  HarqProcessInfoList_t list = m_miDlHarqProcessesInfoMap.at (layer).at (harqProcId);
  double mi = 0.0;
  for (uint8_t i = 0; i < list.size (); i++)
    {
      mi += list.at (i).m_mi;
    }
  return (mi);
}

HarqProcessInfoList_t
LteHarqPhy::GetHarqProcessInfoDl (uint8_t harqProcId, uint8_t layer)
{
  NS_LOG_FUNCTION (this << (uint32_t)harqProcId << (uint16_t)layer);
  return (m_miDlHarqProcessesInfoMap.at (layer).at (harqProcId));   
}


double
LteHarqPhy::GetAccumulatedMiUl (uint16_t rnti)
{
  NS_LOG_FUNCTION (this << rnti);

  std::map <uint16_t, std::vector <HarqProcessInfoList_t> >::iterator it;
  it = m_miUlHarqProcessesInfoMap.find (rnti);
  NS_ASSERT_MSG (it!=m_miUlHarqProcessesInfoMap.end (), " Does not find MI for RNTI");
  HarqProcessInfoList_t list = (*it).second.at (0);
  double mi = 0.0;
  for (uint8_t i = 0; i < list.size (); i++)
    {
      mi += list.at (i).m_mi;
    }
  return (mi);
}

HarqProcessInfoList_t
LteHarqPhy::GetHarqProcessInfoUl (uint16_t rnti, uint8_t harqProcId)
{
  NS_LOG_FUNCTION (this << rnti << (uint16_t)harqProcId);
  std::map <uint16_t, std::vector <HarqProcessInfoList_t> >::iterator it;
  it = m_miUlHarqProcessesInfoMap.find (rnti);
  if (it==m_miUlHarqProcessesInfoMap.end ())
    {
      // new entry
      std::vector <HarqProcessInfoList_t> harqList;
      harqList.resize (8);
      m_miUlHarqProcessesInfoMap.insert (std::pair <uint16_t, std::vector <HarqProcessInfoList_t> > (rnti, harqList));
      return (harqList.at (harqProcId));
    }
  else
    {
      return ((*it).second.at (harqProcId));
    }
}



void
LteHarqPhy::UpdateDlHarqProcessStatus (uint8_t id, uint8_t layer, double mi, uint16_t infoBytes, uint16_t codeBytes)
{
  NS_LOG_FUNCTION (this << (uint16_t) id << mi);
  if (m_miDlHarqProcessesInfoMap.at (layer).at (id).size () == 3)  // MAX HARQ RETX
    {
      // HARQ should be disabled -> discard info
      return;
    }
  HarqProcessInfoElement_t el;
  el.m_mi = mi;
  el.m_infoBits = infoBytes * 8;
  el.m_codeBits = codeBytes * 8;
  m_miDlHarqProcessesInfoMap.at (layer).at (id).push_back (el);
}


void
LteHarqPhy::ResetDlHarqProcessStatus (uint8_t id)
{
  NS_LOG_FUNCTION (this << (uint16_t) id);
  for (uint8_t i = 0; i < m_miDlHarqProcessesInfoMap.size (); i++)
    {
      m_miDlHarqProcessesInfoMap.at (i).at (id).clear ();
    }
  
}


void
LteHarqPhy::UpdateUlHarqProcessStatus (uint16_t rnti, double mi, uint16_t infoBytes, uint16_t codeBytes)
{
  NS_LOG_FUNCTION (this << rnti << mi);
  std::map <uint16_t, std::vector <HarqProcessInfoList_t> >::iterator it;
  it = m_miUlHarqProcessesInfoMap.find (rnti);
  if (it==m_miUlHarqProcessesInfoMap.end ())
    {
      // new entry
      std::vector <HarqProcessInfoList_t> harqList;
      harqList.resize (8);
      HarqProcessInfoElement_t el;
      el.m_mi = mi;
      el.m_infoBits = infoBytes * 8;
      el.m_codeBits = codeBytes * 8;
      harqList.at (7).push_back (el);
      m_miUlHarqProcessesInfoMap.insert (std::pair <uint16_t, std::vector <HarqProcessInfoList_t> > (rnti, harqList));
    }
  else
    {
      if ((*it).second.at (0).size () == 3) // MAX HARQ RETX
        {
          // HARQ should be disabled -> discard info
          return;
        }
      
//       move current status back at the end to maintain full history
      HarqProcessInfoList_t list = (*it).second.at (0);
      for (uint8_t i = 0; i < list.size (); i++)
        {
          (*it).second.at (7).push_back (list.at (i));
        }
      
      HarqProcessInfoElement_t el;
      el.m_mi = mi;
      el.m_infoBits = infoBytes * 8;
      el.m_codeBits = codeBytes * 8;
      (*it).second.at (7).push_back (el);
    }
}

void
LteHarqPhy::ResetUlHarqProcessStatus (uint16_t rnti, uint8_t id)
{
  NS_LOG_FUNCTION (this << rnti << (uint16_t)id);
  std::map <uint16_t, std::vector <HarqProcessInfoList_t> >::iterator it;
  it = m_miUlHarqProcessesInfoMap.find (rnti);
  if (it==m_miUlHarqProcessesInfoMap.end ())
    {
      // new entry
      std::vector <HarqProcessInfoList_t> harqList;
      harqList.resize (8);
      m_miUlHarqProcessesInfoMap.insert (std::pair <uint16_t, std::vector <HarqProcessInfoList_t> > (rnti, harqList));
    }
  else
    {
      (*it).second.at (id).clear ();
    }
}

void
LteHarqPhy::ClearDlHarqBuffer (uint16_t rnti)
{
  NS_LOG_FUNCTION (this << rnti);
  // flush the DL harq buffers
  m_miDlHarqProcessesInfoMap.clear ();
  // Recreate DL Decodification HARQ buffers
  std::vector<HarqProcessInfoList_t> dlHarqLayer0;
  dlHarqLayer0.resize (8);
  std::vector<HarqProcessInfoList_t> dlHarqLayer1;
  dlHarqLayer1.resize (8);
  m_miDlHarqProcessesInfoMap.push_back (dlHarqLayer0);
  m_miDlHarqProcessesInfoMap.push_back (dlHarqLayer1);
}





} // end namespace
