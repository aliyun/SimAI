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
 * Author: Nicola Baldo <nbaldo@cttc.es>
 */

#include "lte-rrc-sap.h"

namespace ns3 {

LteRrcSap::~LteRrcSap ()
{
}

LteRrcSap::ReportConfigEutra::ReportConfigEutra ()
{
  triggerType = EVENT;
  eventId = EVENT_A1;
  threshold1.choice = ThresholdEutra::THRESHOLD_RSRP;
  threshold1.range = 0;
  threshold2.choice = ThresholdEutra::THRESHOLD_RSRP;
  threshold2.range = 0;
  reportOnLeave = false;
  a3Offset = 0;
  hysteresis = 0;
  timeToTrigger = 0;
  purpose = REPORT_STRONGEST_CELLS;
  triggerQuantity = RSRP;
  reportQuantity = BOTH;
  maxReportCells = MaxReportCells;
  reportInterval = MS480;
  reportAmount = 255;
}


} // namespace ns3
