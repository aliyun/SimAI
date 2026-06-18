/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2017
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
 * Author: Sébastien Deronne <sebastien.deronne@gmail.com>
 */

#ifndef SIMPLE_FRAME_CAPTURE_MODEL_H
#define SIMPLE_FRAME_CAPTURE_MODEL_H

#include "frame-capture-model.h"

namespace ns3 {
/**
 * \ingroup wifi
 *
 * A simple threshold-based model for frame capture effect.
 * If the new incoming frame arrives while the receiver is
 * receiving the preamble of another frame and the SIR of
 * the new incoming frame is above a fixed margin, then
 * the current frame is dropped and the receiver locks
 * onto the new incoming frame.
 */
class SimpleFrameCaptureModel : public FrameCaptureModel
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  SimpleFrameCaptureModel ();
  ~SimpleFrameCaptureModel ();

  /**
   * Sets the frame capture margin (dB).
   *
   * \param margin the frame capture margin in dB
   */
  void SetMargin (double margin);
  /**
   * Return the frame capture margin (dB).
   *
   * \return the frame capture margin in dB
   */
  double GetMargin (void) const;

  /**
   * This method returns whether the reception should be switched to a
   * new incoming frame.
   *
   * \param currentEvent the event of the current frame
   * \param newEvent the event of the new incoming frame
   *
   * \return true if the reception should be switched to a new incoming frame,
   *         false otherwise
   */
  bool CaptureNewFrame (Ptr<Event> currentEvent, Ptr<Event> newEvent) const override;


private:
  double m_margin; ///< margin for determining if a new frame (dB)
};

} //namespace ns3

#endif /* SIMPLE_FRAME_CAPTURE_MODEL_H */

