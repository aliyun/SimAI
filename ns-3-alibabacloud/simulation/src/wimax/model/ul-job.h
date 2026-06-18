/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c)
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
 * Author:  Juliana Freitag Borin, Flavio Kubota and Nelson L.
 * S. da Fonseca - wimaxgroup@lrc.ic.unicamp.br
 */

#ifndef UL_JOB_H
#define UL_JOB_H

#include <stdint.h>
#include "ns3/header.h"
#include "ss-record.h"
#include "service-flow.h"
#include "service-flow-record.h"

namespace ns3 {

class SSRecord;
class ServiceFlow;

/// Request type enumeration
enum ReqType
{
  DATA,
  UNICAST_POLLING
};

/**
 * \ingroup wimax
 * \brief this class implements a structure to compute the priority of service flows
 */
class UlJob : public Object
{
public:
  /// Job priority enumeration
  enum JobPriority
  {
    LOW,
    INTERMEDIATE,
    HIGH
  };
  UlJob (void);
  virtual ~UlJob (void);
  /**
   * Get SS record
   * \returns the SS record
   */
  SSRecord *
  GetSsRecord (void);
  /**
   * Set SS record
   * \param ssRecord the SS record
   */
  void SetSsRecord (SSRecord* ssRecord);
  /**
   * Get scheduling type
   * \returns the scheduling type
   */
  enum ServiceFlow::SchedulingType GetSchedulingType (void);
  /**
   * Set scheduling type
   * \param schedulingType the scheduling type
   */
  void SetSchedulingType (ServiceFlow::SchedulingType schedulingType);
  /**
   * Get service flow
   * \returns the service flow
   */
  ServiceFlow *
  GetServiceFlow (void);
  /**
   * Set service flow
   * \param serviceFlow
   */
  void SetServiceFlow (ServiceFlow *serviceFlow);

  /**
   * Get type
   * \returns the request type
   */
  ReqType GetType (void);
  /**
   * Set type
   * \param type the type
   */
  void SetType (ReqType type);

  /**
   * Get release time
   * \returns the release time
   */
  Time GetReleaseTime (void);
  /**
   * Set release time
   * \param releaseTime the release time
   */
  void SetReleaseTime (Time releaseTime);

  /**
   * Get period
   * \returns the period time
   */
  Time GetPeriod (void);
  /**
   * Set period
   * \param period the period
   */
  void SetPeriod (Time period);

  /**
   * Get deadline
   * \returns the deadline time
   */
  Time GetDeadline (void);
  /**
   * Set deadline
   * \param deadline the dead line
   */
  void SetDeadline (Time deadline);

  /**
   * Get size
   * \returns the size
   */
  uint32_t GetSize (void);
  /**
   * Set size
   * \param size the size
   */
  void SetSize (uint32_t size);

private:
  /// equality operator
  friend bool operator == (const UlJob &a, const UlJob &b);

  Time m_releaseTime; ///< The time after which the job can be processed
  Time m_period; ///< For periodic jobs
  Time m_deadline; ///< Request should be satisfied by this time
  uint32_t m_size; ///< Number of minislots requested
  enum ServiceFlow::SchedulingType m_schedulingType; ///< Scheduling type of flow

  SSRecord *m_ssRecord; ///< Pointer to SSRecord

  ReqType m_type; ///< Type of request, DATA or Unicast req slots
  ServiceFlow *m_serviceFlow; ///< service flow

};

/**
 * PriorityUlJob class
 */
class PriorityUlJob : public Object
{

  /**
   * \brief this class implements an auxiliary struct to compute the priority of the rtPS and nrtPS in
   * the intermediate queue
   */
public:
  PriorityUlJob ();
  /**
   * Get priority
   * \returns the priority
   */
  int GetPriority (void);
  /**
   * Set priority
   * \param priority the priority
   */
  void SetPriority (int priority);

  /**
   * Get UL job functiion
   * \returns the UL job
   */
  Ptr<UlJob>
  GetUlJob (void);
  /**
   * Set UL job
   * \param job the UL job
   */
  void SetUlJob (Ptr<UlJob> job);

private:
  int m_priority; ///< the priority
  Ptr<UlJob> m_job; ///< the job
};

/// SortProcess structure 
struct SortProcess
{
  /**
   * \brief comparison operator
   * \param left left side input
   * \param right right side input
   * \returns true if left is logically less then right for given comparison
   */
   bool operator () (PriorityUlJob& left, PriorityUlJob& right) const
   { 
    if (left.GetPriority () < right.GetPriority ())
      {
        return true;
      }
    else if (left.GetPriority () == right.GetPriority ())
      {
        int32_t leftBacklogged = left.GetUlJob ()->GetServiceFlow ()->GetRecord ()->GetBacklogged ();
        int32_t rightBacklogged = left.GetUlJob ()->GetServiceFlow ()->GetRecord ()->GetBacklogged ();
        if (leftBacklogged <= rightBacklogged)
          {
            return true;
          }
        else
          {
            return false;
          }
      }
    else
      {
        return false;
      }
  }
};

/// SortProcessPtr structure
struct SortProcessPtr
{
  /**
   * \brief comparison operator
   * \param left left side input
   * \param right right side input
   * \returns true if left is logically less then right for given comparison
   */
  bool operator () (Ptr<PriorityUlJob>& left, Ptr<PriorityUlJob>& right) const
  { 
    if (left->GetPriority () < right->GetPriority ())
      {
        return true;
      }
    else if (left->GetPriority () == right->GetPriority ())
      {
        int32_t leftBacklogged = left->GetUlJob ()->GetServiceFlow ()->GetRecord ()->GetBacklogged ();
        int32_t rightBacklogged = left->GetUlJob ()->GetServiceFlow ()->GetRecord ()->GetBacklogged ();
        if (leftBacklogged <= rightBacklogged)
          {
            return true;
          }
        else
          {
            return false;
          }
      }
    else
      {
        return false;
      }
  }
};


} // namespace ns3

#endif /* UL_JOB_H */
