/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2005,2006 INRIA
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
 * Author: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 */

#include "ns3/test.h"
#include "ns3/simulator.h"
#include "ns3/channel-access-manager.h"
#include "ns3/frame-exchange-manager.h"
#include "ns3/qos-txop.h"

using namespace ns3;

template <typename TxopType>
class ChannelAccessManagerTest;

/**
 * \ingroup wifi-test
 * \ingroup tests
 *
 * \brief TxopTest Txop Test
 */
template <typename TxopType>
class TxopTest : public TxopType
{
public:
  /**
   * Constructor
   *
   * \param test the test channel access manager
   * \param i the index of the Txop
   */
  TxopTest (ChannelAccessManagerTest<TxopType> *test, uint32_t i);

  /**
   * Queue transmit function
   * \param txTime the transmit time
   * \param expectedGrantTime the expected grant time
   */
  void QueueTx (uint64_t txTime, uint64_t expectedGrantTime);

private:
  /// allow ChannelAccessManagerTest class access
  friend class ChannelAccessManagerTest<TxopType>;

  /// \copydoc ns3::Txop::DoDispose
  void DoDispose (void) override;
  /// \copydoc ns3::Txop::NotifyChannelAccessed
  void NotifyChannelAccessed (Time txopDuration = Seconds (0)) override;
  /// \copydoc ns3::Txop::HasFramesToTransmit
  bool HasFramesToTransmit (void) override;
  /// \copydoc ns3::Txop::NotifySleep
  void NotifySleep (void) override;
  /// \copydoc ns3::Txop::NotifyWakeUp
  void NotifyWakeUp (void) override;
  /// \copydoc ns3::Txop::GenerateBackoff
  void GenerateBackoff (void) override;

  typedef std::pair<uint64_t,uint64_t> ExpectedGrant; //!< the expected grant typedef
  typedef std::list<ExpectedGrant> ExpectedGrants; //!< the collection of expected grants typedef
  /// ExpectedBackoff structure
  struct ExpectedBackoff
  {
    uint64_t at; //!< at
    uint32_t nSlots; //!< number of slots
  };
  typedef std::list<struct ExpectedBackoff> ExpectedBackoffs; //!< expected backoffs typedef

  ExpectedBackoffs m_expectedInternalCollision; //!< expected backoff due to an internal collision
  ExpectedBackoffs m_expectedBackoff; //!< expected backoff (not due to an internal collision)
  ExpectedGrants m_expectedGrants; //!< expected grants

  /**
   * Check if the Txop has frames to transmit.
   * \return true if the Txop has frames to transmit.
   */

  ChannelAccessManagerTest<TxopType> *m_test; //!< the test DCF/EDCA manager
  uint32_t m_i; //!< the index of the Txop
};

/**
 * \ingroup wifi-test
 * \ingroup tests
 *
 * \brief ChannelAccessManager Stub
 */
class ChannelAccessManagerStub : public ChannelAccessManager
{
public:
  ChannelAccessManagerStub ()
  {
  }
  /**
   * Set the Short Interframe Space (SIFS).
   *
   * \param sifs the SIFS duration
   */
  void SetSifs (Time sifs)
  {
    m_sifs = sifs;
  }
  /**
   * Set the slot duration.
   *
   * \param slot the slot duration
   */
  void SetSlot (Time slot)
  {
    m_slot = slot;
  }
  /**
   * Set the duration of EIFS - DIFS
   *
   * \param eifsNoDifs the duration of EIFS - DIFS
   */
  void SetEifsNoDifs (Time eifsNoDifs)
  {
    m_eifsNoDifs = eifsNoDifs;
  }

private:
  Time GetSifs (void) const override
  {
    return m_sifs;
  }

  Time GetSlot (void) const override
  {
    return m_slot;
  }

  Time GetEifsNoDifs (void) const override
  {
    return m_eifsNoDifs;
  }

  Time m_slot;        //!< slot duration
  Time m_sifs;        //!< SIFS duration
  Time m_eifsNoDifs;  //!< EIFS duration minus a DIFS
};

/**
 * \ingroup wifi-test
 * \ingroup tests
 *
 * \brief Frame Exchange Manager Stub
 */
template <typename TxopType>
class FrameExchangeManagerStub : public FrameExchangeManager
{
public:
  /**
   * Constructor
   *
   * \param test the test channel access manager
   */
  FrameExchangeManagerStub (ChannelAccessManagerTest<TxopType> *test)
    : m_test (test)
  {
  }
  /**
   * Request the FrameExchangeManager to start a frame exchange sequence.
   *
   * \param dcf the channel access function that gained channel access. It is
   *            the DCF on non-QoS stations and an EDCA on QoS stations.
   * \return true if a frame exchange sequence was started, false otherwise
   */
  bool StartTransmission (Ptr<Txop> dcf) override
  {
    dcf->NotifyChannelAccessed ();
    return true;
  }
  /// \copydoc ns3::FrameExchangeManager::NotifyInternalCollision
  void NotifyInternalCollision (Ptr<Txop> txop) override
  {
    m_test->NotifyInternalCollision (DynamicCast<TxopTest<TxopType>> (txop));
  }
  /// \copydoc ns3::FrameExchangeManager::NotifySwitchingStartNow
  void NotifySwitchingStartNow (Time duration) override
  {
    m_test->NotifyChannelSwitching ();
  }

private:
  ChannelAccessManagerTest<TxopType> *m_test; //!< the test DCF/EDCA manager
};

/**
 * \ingroup wifi-test
 * \ingroup tests
 *
 * \brief Channel Access Manager Test
 */
template <typename TxopType>
class ChannelAccessManagerTest : public TestCase
{
public:
  ChannelAccessManagerTest ();
  void DoRun (void) override;

  /**
   * Notify access granted function
   * \param i the index of the Txop
   */
  void NotifyAccessGranted (uint32_t i);
  /**
   * Notify internal collision function
   * \param state the Txop
   */
  void NotifyInternalCollision (Ptr<TxopTest<TxopType>> state);
  /**
   * Generate backoff function
   * \param i the index of the Txop
   */
  void GenerateBackoff (uint32_t i);
  /**
   * Notify channel switching function
   */
  void NotifyChannelSwitching (void);


private:
  /**
   * Start test function
   * \param slotTime the slot time
   * \param sifs the SIFS
   * \param eifsNoDifsNoSifs the EIFS no DIFS no SIFS
   * \param ackTimeoutValue the Ack timeout value
   */
  void StartTest (uint64_t slotTime, uint64_t sifs, uint64_t eifsNoDifsNoSifs, uint32_t ackTimeoutValue = 20);
  /**
   * Add Txop function
   * \param aifsn the AIFSN
   */
  void AddTxop (uint32_t aifsn);
  /// End test function
  void EndTest (void);
  /**
   * Expect internal collision function
   * \param time the expected time
   * \param nSlots the number of slots
   * \param from the expected from
   */
  void ExpectInternalCollision (uint64_t time, uint32_t nSlots, uint32_t from);
  /**
   * Expect generate backoff function
   * \param time the expected time
   * \param nSlots the number of slots
   * \param from the expected from
   */
  void ExpectBackoff (uint64_t time, uint32_t nSlots, uint32_t from);
  /**
   * Schedule a check that the channel access manager is busy or idle
   * \param time the expected time
   * \param busy whether the manager is expected to be busy
   */
  void ExpectBusy (uint64_t time, bool busy);
  /**
   * Perform check that channel access manager is busy or idle
   * \param busy whether expected state is busy
   */
  void DoCheckBusy (bool busy);
  /**
   * Add receive OK event function
   * \param at the event time
   * \param duration the duration
   */
  void AddRxOkEvt (uint64_t at, uint64_t duration);
  /**
   * Add receive error event function for error at end of frame
   * \param at the event time
   * \param duration the duration
   */
  void AddRxErrorEvt (uint64_t at, uint64_t duration);
  /**
   * Add receive error event function for error during frame
   * \param at the event time
   * \param duration the duration
   * \param timeUntilError the time after event time to force the error
   */
  void AddRxErrorEvt (uint64_t at, uint64_t duration, uint64_t timeUntilError);
  /**
   * Add receive inside SIFS event function
   * \param at the event time
   * \param duration the duration
   */
  void AddRxInsideSifsEvt (uint64_t at, uint64_t duration);
  /**
   * Add transmit event function
   * \param at the event time
   * \param duration the duration
   */
  void AddTxEvt (uint64_t at, uint64_t duration);
  /**
   * Add NAV reset function
   * \param at the event time
   * \param duration the duration
   */
  void AddNavReset (uint64_t at, uint64_t duration);
  /**
   * Add NAV start function
   * \param at the event time
   * \param duration the duration
   */
  void AddNavStart (uint64_t at, uint64_t duration);
  /**
   * Add Ack timeout reset function
   * \param at the event time
   */
  void AddAckTimeoutReset (uint64_t at);
  /**
   * Add access function
   * \param at the event time
   * \param txTime the transmit time
   * \param expectedGrantTime the expected grant time
   * \param from the index of the requesting Txop
   */
  void AddAccessRequest (uint64_t at, uint64_t txTime,
                         uint64_t expectedGrantTime, uint32_t from);
  /**
   * Add access request with Ack timeout
   * \param at time to schedule DoAccessRequest event
   * \param txTime the transmit time
   * \param expectedGrantTime the expected grant time
   * \param from the index of the requesting Txop
   */
  void AddAccessRequestWithAckTimeout (uint64_t at, uint64_t txTime,
                                       uint64_t expectedGrantTime, uint32_t from);
  /**
   * Add access request with successful ack
   * \param at time to schedule DoAccessRequest event
   * \param txTime the transmit time
   * \param expectedGrantTime the expected grant time
   * \param ackDelay the delay of the Ack after txEnd
   * \param from the index of the requesting Txop
   */
  void AddAccessRequestWithSuccessfullAck (uint64_t at, uint64_t txTime,
                                           uint64_t expectedGrantTime, uint32_t ackDelay, uint32_t from);
  /**
   * Add access request with successful Ack
   * \param txTime the transmit time
   * \param expectedGrantTime the expected grant time
   * \param state TxopTest
   */
  void DoAccessRequest (uint64_t txTime, uint64_t expectedGrantTime, Ptr<TxopTest<TxopType>> state);
  /**
   * Add CCA busy event function
   * \param at the event time
   * \param duration the duration
   */
  void AddCcaBusyEvt (uint64_t at, uint64_t duration);
  /**
   * Add switching event function
   * \param at the event time
   * \param duration the duration
   */
  void AddSwitchingEvt (uint64_t at, uint64_t duration);
  /**
   * Add receive start event function
   * \param at the event time
   * \param duration the duration
   */
  void AddRxStartEvt (uint64_t at, uint64_t duration);

  typedef std::vector<Ptr<TxopTest<TxopType>>> TxopTests; //!< the TXOP tests typedef

  Ptr<FrameExchangeManagerStub<TxopType>> m_feManager; //!< the Frame Exchange Manager stubbed
  Ptr<ChannelAccessManagerStub> m_ChannelAccessManager; //!< the channel access manager
  TxopTests m_txop; //!< the vector of Txop test instances
  uint32_t m_ackTimeoutValue; //!< the Ack timeout value
};

template <typename TxopType>
void
TxopTest<TxopType>::QueueTx (uint64_t txTime, uint64_t expectedGrantTime)
{
  m_expectedGrants.push_back (std::make_pair (txTime, expectedGrantTime));
}

template <typename TxopType>
TxopTest<TxopType>::TxopTest (ChannelAccessManagerTest<TxopType> *test, uint32_t i)
  : m_test (test),
    m_i (i)
{
}

template <typename TxopType>
void
TxopTest<TxopType>::DoDispose (void)
{
  m_test = 0;
  TxopType::DoDispose ();
}

template <typename TxopType>
void
TxopTest<TxopType>::NotifyChannelAccessed (Time txopDuration)
{
  Txop::m_access = Txop::NOT_REQUESTED;
  m_test->NotifyAccessGranted (m_i);
}

template <typename TxopType>
void
TxopTest<TxopType>::GenerateBackoff (void)
{
  m_test->GenerateBackoff (m_i);
}

template <typename TxopType>
bool
TxopTest<TxopType>::HasFramesToTransmit (void)
{
  return !m_expectedGrants.empty ();
}

template <typename TxopType>
void
TxopTest<TxopType>::NotifySleep (void)
{
}

template <typename TxopType>
void
TxopTest<TxopType>::NotifyWakeUp (void)
{
}

template <typename TxopType>
ChannelAccessManagerTest<TxopType>::ChannelAccessManagerTest ()
  : TestCase ("ChannelAccessManager")
{
}

template <typename TxopType>
void
ChannelAccessManagerTest<TxopType>::NotifyAccessGranted (uint32_t i)
{
  Ptr<TxopTest<TxopType>> state = m_txop[i];
  NS_TEST_EXPECT_MSG_EQ (state->m_expectedGrants.empty (), false, "Have expected grants");
  if (!state->m_expectedGrants.empty ())
    {
      std::pair<uint64_t, uint64_t> expected = state->m_expectedGrants.front ();
      state->m_expectedGrants.pop_front ();
      NS_TEST_EXPECT_MSG_EQ (Simulator::Now (), MicroSeconds (expected.second), "Expected access grant is now");
      m_ChannelAccessManager->NotifyTxStartNow (MicroSeconds (expected.first));
      m_ChannelAccessManager->NotifyAckTimeoutStartNow (MicroSeconds (m_ackTimeoutValue + expected.first));
    }
}

template <typename TxopType>
void
ChannelAccessManagerTest<TxopType>::AddTxEvt (uint64_t at, uint64_t duration)
{
  Simulator::Schedule (MicroSeconds (at) - Now (),
                       &ChannelAccessManager::NotifyTxStartNow, m_ChannelAccessManager,
                       MicroSeconds (duration));
}

template <typename TxopType>
void
ChannelAccessManagerTest<TxopType>::NotifyInternalCollision (Ptr<TxopTest<TxopType>> state)
{
  NS_TEST_EXPECT_MSG_EQ (state->m_expectedInternalCollision.empty (), false, "Have expected internal collisions");
  if (!state->m_expectedInternalCollision.empty ())
    {
      struct TxopTest<TxopType>::ExpectedBackoff expected = state->m_expectedInternalCollision.front ();
      state->m_expectedInternalCollision.pop_front ();
      NS_TEST_EXPECT_MSG_EQ (Simulator::Now (), MicroSeconds (expected.at), "Expected internal collision time is now");
      state->StartBackoffNow (expected.nSlots);
    }
}

template <typename TxopType>
void
ChannelAccessManagerTest<TxopType>::GenerateBackoff (uint32_t i)
{
  Ptr<TxopTest<TxopType>> state = m_txop[i];
  NS_TEST_EXPECT_MSG_EQ (state->m_expectedBackoff.empty (), false, "Have expected backoffs");
  if (!state->m_expectedBackoff.empty ())
    {
      struct TxopTest<TxopType>::ExpectedBackoff expected = state->m_expectedBackoff.front ();
      state->m_expectedBackoff.pop_front ();
      NS_TEST_EXPECT_MSG_EQ (Simulator::Now (), MicroSeconds (expected.at), "Expected backoff is now");
      state->StartBackoffNow (expected.nSlots);
    }
}

template <typename TxopType>
void
ChannelAccessManagerTest<TxopType>::NotifyChannelSwitching (void)
{
  for (auto& state : m_txop)
    {
      if (!state->m_expectedGrants.empty ())
        {
          std::pair<uint64_t, uint64_t> expected = state->m_expectedGrants.front ();
          state->m_expectedGrants.pop_front ();
          NS_TEST_EXPECT_MSG_EQ (Simulator::Now (), MicroSeconds (expected.second), "Expected grant is now");
        }
      state->Txop::m_access = Txop::NOT_REQUESTED;
    }
}

template <typename TxopType>
void
ChannelAccessManagerTest<TxopType>::ExpectInternalCollision (uint64_t time, uint32_t nSlots, uint32_t from)
{
  Ptr<TxopTest<TxopType>> state = m_txop[from];
  struct TxopTest<TxopType>::ExpectedBackoff col;
  col.at = time;
  col.nSlots = nSlots;
  state->m_expectedInternalCollision.push_back (col);
}

template <typename TxopType>
void
ChannelAccessManagerTest<TxopType>::ExpectBackoff (uint64_t time, uint32_t nSlots, uint32_t from)
{
  Ptr<TxopTest<TxopType>> state = m_txop[from];
  struct TxopTest<TxopType>::ExpectedBackoff backoff;
  backoff.at = time;
  backoff.nSlots = nSlots;
  state->m_expectedBackoff.push_back (backoff);
}

template <typename TxopType>
void
ChannelAccessManagerTest<TxopType>::ExpectBusy (uint64_t time, bool busy)
{
  Simulator::Schedule (MicroSeconds (time) - Now (),
                       &ChannelAccessManagerTest::DoCheckBusy, this, busy);
}

template <typename TxopType>
void
ChannelAccessManagerTest<TxopType>::DoCheckBusy (bool busy)
{
  NS_TEST_EXPECT_MSG_EQ (m_ChannelAccessManager->IsBusy (), busy, "Incorrect busy/idle state");
}

template <typename TxopType>
void
ChannelAccessManagerTest<TxopType>::StartTest (uint64_t slotTime, uint64_t sifs, uint64_t eifsNoDifsNoSifs, uint32_t ackTimeoutValue)
{
  m_ChannelAccessManager = CreateObject<ChannelAccessManagerStub> ();
  m_feManager = CreateObject<FrameExchangeManagerStub<TxopType>> (this);
  m_ChannelAccessManager->SetupFrameExchangeManager (m_feManager);
  m_ChannelAccessManager->SetSlot (MicroSeconds (slotTime));
  m_ChannelAccessManager->SetSifs (MicroSeconds (sifs));
  m_ChannelAccessManager->SetEifsNoDifs (MicroSeconds (eifsNoDifsNoSifs + sifs));
  m_ackTimeoutValue = ackTimeoutValue;
}

template <typename TxopType>
void
ChannelAccessManagerTest<TxopType>::AddTxop (uint32_t aifsn)
{
  Ptr<TxopTest<TxopType>> txop = CreateObject<TxopTest<TxopType>> (this, m_txop.size ());
  txop->SetAifsn (aifsn);
  m_txop.push_back (txop);
  txop->SetChannelAccessManager (m_ChannelAccessManager);
}

template <typename TxopType>
void
ChannelAccessManagerTest<TxopType>::EndTest (void)
{
  Simulator::Run ();

  for (typename TxopTests::const_iterator i = m_txop.begin (); i != m_txop.end (); i++)
    {
      Ptr<TxopTest<TxopType>> state = *i;
      NS_TEST_EXPECT_MSG_EQ (state->m_expectedGrants.empty (), true, "Have no expected grants");
      NS_TEST_EXPECT_MSG_EQ (state->m_expectedInternalCollision.empty (), true, "Have no internal collisions");
      NS_TEST_EXPECT_MSG_EQ (state->m_expectedBackoff.empty (), true, "Have no expected backoffs");
      state->Dispose ();
      state = 0;
    }
  m_txop.clear ();

  m_ChannelAccessManager->Dispose ();
  m_ChannelAccessManager = 0;
  m_feManager = 0;
  Simulator::Destroy ();
}

template <typename TxopType>
void
ChannelAccessManagerTest<TxopType>::AddRxOkEvt (uint64_t at, uint64_t duration)
{
  Simulator::Schedule (MicroSeconds (at) - Now (),
                       &ChannelAccessManager::NotifyRxStartNow, m_ChannelAccessManager,
                       MicroSeconds (duration));
  Simulator::Schedule (MicroSeconds (at + duration) - Now (),
                       &ChannelAccessManager::NotifyRxEndOkNow, m_ChannelAccessManager);
}

template <typename TxopType>
void
ChannelAccessManagerTest<TxopType>::AddRxInsideSifsEvt (uint64_t at, uint64_t duration)
{
  Simulator::Schedule (MicroSeconds (at) - Now (),
                       &ChannelAccessManager::NotifyRxStartNow, m_ChannelAccessManager,
                       MicroSeconds (duration));
}

template <typename TxopType>
void
ChannelAccessManagerTest<TxopType>::AddRxErrorEvt (uint64_t at, uint64_t duration)
{
  Simulator::Schedule (MicroSeconds (at) - Now (),
                       &ChannelAccessManager::NotifyRxStartNow, m_ChannelAccessManager,
                       MicroSeconds (duration));
  Simulator::Schedule (MicroSeconds (at + duration) - Now (),
                       &ChannelAccessManager::NotifyRxEndErrorNow, m_ChannelAccessManager);
}

template <typename TxopType>
void
ChannelAccessManagerTest<TxopType>::AddRxErrorEvt (uint64_t at, uint64_t duration, uint64_t timeUntilError)
{
  Simulator::Schedule (MicroSeconds (at) - Now (),
                       &ChannelAccessManager::NotifyRxStartNow, m_ChannelAccessManager,
                       MicroSeconds (duration));
  Simulator::Schedule (MicroSeconds (at + timeUntilError) - Now (),
                       &ChannelAccessManager::NotifyRxEndErrorNow, m_ChannelAccessManager);
}


template <typename TxopType>
void
ChannelAccessManagerTest<TxopType>::AddNavReset (uint64_t at, uint64_t duration)
{
  Simulator::Schedule (MicroSeconds (at) - Now (),
                       &ChannelAccessManager::NotifyNavResetNow, m_ChannelAccessManager,
                       MicroSeconds (duration));
}

template <typename TxopType>
void
ChannelAccessManagerTest<TxopType>::AddNavStart (uint64_t at, uint64_t duration)
{
  Simulator::Schedule (MicroSeconds (at) - Now (),
                       &ChannelAccessManager::NotifyNavStartNow, m_ChannelAccessManager,
                       MicroSeconds (duration));
}

template <typename TxopType>
void
ChannelAccessManagerTest<TxopType>::AddAckTimeoutReset (uint64_t at)
{
  Simulator::Schedule (MicroSeconds (at) - Now (),
                       &ChannelAccessManager::NotifyAckTimeoutResetNow, m_ChannelAccessManager);
}

template <typename TxopType>
void
ChannelAccessManagerTest<TxopType>::AddAccessRequest (uint64_t at, uint64_t txTime,
                                            uint64_t expectedGrantTime, uint32_t from)
{
  AddAccessRequestWithSuccessfullAck (at, txTime, expectedGrantTime, 0, from);
}

template <typename TxopType>
void
ChannelAccessManagerTest<TxopType>::AddAccessRequestWithAckTimeout (uint64_t at, uint64_t txTime,
                                                          uint64_t expectedGrantTime, uint32_t from)
{
  Simulator::Schedule (MicroSeconds (at) - Now (),
                       &ChannelAccessManagerTest::DoAccessRequest, this,
                       txTime, expectedGrantTime, m_txop[from]);
}

template <typename TxopType>
void
ChannelAccessManagerTest<TxopType>::AddAccessRequestWithSuccessfullAck (uint64_t at, uint64_t txTime,
                                                              uint64_t expectedGrantTime, uint32_t ackDelay, uint32_t from)
{
  NS_ASSERT (ackDelay < m_ackTimeoutValue);
  Simulator::Schedule (MicroSeconds (at) - Now (),
                       &ChannelAccessManagerTest::DoAccessRequest, this,
                       txTime, expectedGrantTime, m_txop[from]);
  AddAckTimeoutReset (expectedGrantTime + txTime + ackDelay);
}

template <typename TxopType>
void
ChannelAccessManagerTest<TxopType>::DoAccessRequest (uint64_t txTime, uint64_t expectedGrantTime,
                                                     Ptr<TxopTest<TxopType>> state)
{
  if (m_ChannelAccessManager->NeedBackoffUponAccess (state))
    {
      state->GenerateBackoff ();
    }
  state->QueueTx (txTime, expectedGrantTime);
  m_ChannelAccessManager->RequestAccess (state);
}

template <typename TxopType>
void
ChannelAccessManagerTest<TxopType>::AddCcaBusyEvt (uint64_t at, uint64_t duration)
{
  Simulator::Schedule (MicroSeconds (at) - Now (),
                       &ChannelAccessManager::NotifyMaybeCcaBusyStartNow, m_ChannelAccessManager,
                       MicroSeconds (duration));
}

template <typename TxopType>
void
ChannelAccessManagerTest<TxopType>::AddSwitchingEvt (uint64_t at, uint64_t duration)
{
  Simulator::Schedule (MicroSeconds (at) - Now (),
                       &ChannelAccessManager::NotifySwitchingStartNow, m_ChannelAccessManager,
                       MicroSeconds (duration));
}

template <typename TxopType>
void
ChannelAccessManagerTest<TxopType>::AddRxStartEvt (uint64_t at, uint64_t duration)
{
  Simulator::Schedule (MicroSeconds (at) - Now (),
                       &ChannelAccessManager::NotifyRxStartNow, m_ChannelAccessManager,
                       MicroSeconds (duration));
}

/*
 * Specialization of DoRun () method for DCF
 */
template <>
void
ChannelAccessManagerTest<Txop>::DoRun (void)
{
  // DCF immediate access (no backoff)
  //  1      4       5    6      8     11      12
  //  | sifs | aifsn | tx | idle | sifs | aifsn | tx |
  //
  StartTest (1, 3, 10);
  AddTxop (1);
  AddAccessRequest (1, 1, 5, 0);
  AddAccessRequest (8, 2, 12, 0);
  EndTest ();
  // Check that receiving inside SIFS shall be cancelled properly:
  //  1      4       5    6      9    10     14     17      18
  //  | sifs | aifsn | tx | sifs | ack | idle | sifs | aifsn | tx |
  //                        |
  //                        7 start rx
  //

  StartTest (1, 3, 10);
  AddTxop (1);
  AddAccessRequest (1, 1, 5, 0);
  AddRxInsideSifsEvt (7, 10);
  AddTxEvt (9, 1);
  AddAccessRequest (14, 2, 18, 0);
  EndTest ();
  // The test below mainly intends to test the case where the medium
  // becomes busy in the middle of a backoff slot: the backoff counter
  // must not be decremented for this backoff slot. This is the case
  // below for the backoff slot starting at time 78us.
  //
  //  20          60     66      70        74        78  80    100     106      110      114      118   120
  //   |    rx     | sifs | aifsn | bslot0  | bslot1  |   | rx   | sifs  |  aifsn | bslot2 | bslot3 | tx  |
  //        |
  //       30 request access. backoff slots: 4

  StartTest (4, 6, 10);
  AddTxop (1);
  AddRxOkEvt (20, 40);
  AddRxOkEvt (80, 20);
  AddAccessRequest (30, 2, 118, 0);
  ExpectBackoff (30, 4, 0); //backoff: 4 slots
  EndTest ();
  // Test the case where the backoff slots is zero.
  //
  //  20          60     66      70   72
  //   |    rx     | sifs | aifsn | tx |
  //        |
  //       30 request access. backoff slots: 0

  StartTest (4, 6, 10);
  AddTxop (1);
  AddRxOkEvt (20, 40);
  AddAccessRequest (30, 2, 70, 0);
  ExpectBackoff (30, 0, 0); // backoff: 0 slots
  EndTest ();
  // Test shows when two frames are received without interval between
  // them:
  //  20          60         100   106     110  112
  //   |    rx     |    rx     |sifs | aifsn | tx |
  //        |
  //       30 request access. backoff slots: 0

  StartTest (4, 6, 10);
  AddTxop (1);
  AddRxOkEvt (20, 40);
  AddRxOkEvt (60, 40);
  AddAccessRequest (30, 2, 110, 0);
  ExpectBackoff (30, 0, 0); //backoff: 0 slots
  EndTest ();

  // Requesting access within SIFS interval (DCF immediate access)
  //
  //  20    60     62     68      72
  //   | rx  | idle | sifs | aifsn | tx |
  //
  StartTest (4, 6, 10);
  AddTxop (1);
  AddRxOkEvt (20, 40);
  AddAccessRequest (62, 2, 72, 0);
  EndTest ();

  // Requesting access after DIFS (DCF immediate access)
  //
  //   20   60     70     76      80
  //   | rx  | idle | sifs | aifsn | tx |
  //
  StartTest (4, 6, 10);
  AddTxop (1);
  AddRxOkEvt (20, 40);
  AddAccessRequest (70, 2, 80, 0);
  EndTest ();

  // Test an EIFS
  //
  //  20          60     66           76             86       90       94       98       102   106
  //   |    rx     | sifs | acktxttime | sifs + aifsn | bslot0 | bslot1 | bslot2 | bslot3 | tx |
  //        |      | <------eifs------>|
  //       30 request access. backoff slots: 4
  StartTest (4, 6, 10);
  AddTxop (1);
  AddRxErrorEvt (20, 40);
  AddAccessRequest (30, 2, 102, 0);
  ExpectBackoff (30, 4, 0); //backoff: 4 slots
  EndTest ();

  // Test DCF immediate access after an EIFS (EIFS is greater)
  //
  //  20          60     66           76             86
  //               | <----+-eifs------>|
  //   |    rx     | sifs | acktxttime | sifs + aifsn | tx |
  //                             | sifs + aifsn |
  //             request access 70             80
  StartTest (4, 6, 10);
  AddTxop (1);
  AddRxErrorEvt (20, 40);
  AddAccessRequest (70, 2, 86, 0);
  EndTest ();

  // Test that channel stays busy for first frame's duration after Rx error
  //
  //  20          60
  //   |    rx     |
  //        |
  //       40 force Rx error
  StartTest (4, 6, 10);
  AddTxop (1);
  AddRxErrorEvt (20, 40, 20); // At time 20, start reception for 40, but force error 20 into frame
  ExpectBusy (41, true); // channel should remain busy for remaining duration
  ExpectBusy (59, true);
  ExpectBusy (61, false);
  EndTest ();

  // Test an EIFS which is interrupted by a successful transmission.
  //
  //  20          60      66  69     75     81      85       89       93       97      101  103
  //   |    rx     | sifs  |   |  rx  | sifs | aifsn | bslot0 | bslot1 | bslot2 | bslot3 | tx |
  //        |      | <--eifs-->|
  //       30 request access. backoff slots: 4
  StartTest (4, 6, 10);
  AddTxop (1);
  AddRxErrorEvt (20, 40);
  AddAccessRequest (30, 2, 101, 0);
  ExpectBackoff (30, 4, 0); //backoff: 4 slots
  AddRxOkEvt (69, 6);
  EndTest ();

  // Test two DCFs which suffer an internal collision. the first DCF has a higher
  // priority than the second DCF.
  //
  //      20          60      66      70       74       78    88
  // DCF0  |    rx     | sifs  | aifsn | bslot0 | bslot1 | tx  |
  // DCF1  |    rx     | sifs  | aifsn | aifsn  | aifsn  |     | sifs | aifsn | aifsn | aifsn | bslot |  tx  |
  //                                                                 94      98     102     106     110    112
  StartTest (4, 6, 10);
  AddTxop (1); //high priority DCF
  AddTxop (3); //low priority DCF
  AddRxOkEvt (20, 40);
  AddAccessRequest (30, 10, 78, 0);
  ExpectBackoff (30, 2, 0); //backoff: 2 slot
  AddAccessRequest (40, 2, 110, 1);
  ExpectBackoff (40, 0, 1); //backoff: 0 slot
  ExpectInternalCollision (78, 1, 1); //backoff: 1 slot
  EndTest ();

  // Test of AckTimeout handling: First queue requests access and ack procedure fails,
  // inside the Ack timeout second queue with higher priority requests access.
  //
  //            20     26      34       54            74     80
  // DCF1 - low  | sifs | aifsn |   tx   | Ack timeout | sifs |       |
  // DCF0 - high |                              |      | sifs |  tx   |
  //                                            ^ request access
  StartTest (4, 6, 10);
  AddTxop (0); //high priority DCF
  AddTxop (2); //low priority DCF
  AddAccessRequestWithAckTimeout (20, 20, 34, 1);
  AddAccessRequest (64, 10, 80, 0);
  EndTest ();

  // Test of AckTimeout handling:
  //
  // First queue requests access and Ack is 2 us delayed (got Ack interval at the picture),
  // inside this interval second queue with higher priority requests access.
  //
  //            20     26      34           54        56     62
  // DCF1 - low  | sifs | aifsn |     tx     | got Ack | sifs |       |
  // DCF0 - high |                                |    | sifs |  tx   |
  //                                              ^ request access
  StartTest (4, 6, 10);
  AddTxop (0); //high priority DCF
  AddTxop (2); //low priority DCF
  AddAccessRequestWithSuccessfullAck (20, 20, 34, 2, 1);
  AddAccessRequest (55, 10, 62, 0);
  EndTest ();

  //Repeat the same but with one queue:
  //      20     26      34         54     60    62     68      76       80
  // DCF0  | sifs | aifsn |    tx    | sifs | Ack | sifs | aifsn | bslot0 | tx |
  //                                           ^ request access
  StartTest (4, 6, 10);
  AddTxop (2);
  AddAccessRequest (20, 20, 34, 0);
  AddRxOkEvt (60, 2); // Ack
  AddAccessRequest (61, 10, 80, 0);
  ExpectBackoff (61, 1, 0); // 1 slot
  EndTest ();

  // test simple NAV count. This scenario models a simple Data+Ack handshake
  // where the data rate used for the Ack is higher than expected by the Data source
  // so, the data exchange completes before the end of NAV.
  StartTest (4, 6, 10);
  AddTxop (1);
  AddRxOkEvt (20, 40);
  AddNavStart (60, 15);
  AddRxOkEvt (66, 5);
  AddNavStart (71, 0);
  AddAccessRequest (30, 10, 93, 0);
  ExpectBackoff (30, 2, 0); //backoff: 2 slots
  EndTest ();

  // test more complex NAV handling by a CF-poll. This scenario models a
  // simple Data+Ack handshake interrupted by a CF-poll which resets the
  // NAV counter.
  StartTest (4, 6, 10);
  AddTxop (1);
  AddRxOkEvt (20, 40);
  AddNavStart (60, 15);
  AddRxOkEvt (66, 5);
  AddNavReset (71, 2);
  AddAccessRequest (30, 10, 91, 0);
  ExpectBackoff (30, 2, 0); //backoff: 2 slots
  EndTest ();


  //  20         60         80     86      94
  //   |    rx    |   idle   | sifs | aifsn |    tx    |
  //                         ^ request access
  StartTest (4, 6, 10);
  AddTxop (2);
  AddRxOkEvt (20, 40);
  AddAccessRequest (80, 10, 94, 0);
  EndTest ();


  StartTest (4, 6, 10);
  AddTxop (2);
  AddRxOkEvt (20, 40);
  AddRxOkEvt (78, 8);
  AddAccessRequest (30, 50, 108, 0);
  ExpectBackoff (30, 3, 0); //backoff: 3 slots
  EndTest ();


  // Channel switching tests

  //  0          20     21     24      25   26
  //  | switching | idle | sifs | aifsn | tx |
  //                     ^ access request.
  StartTest (1, 3, 10);
  AddTxop (1);
  AddSwitchingEvt (0, 20);
  AddAccessRequest (21, 1, 25, 0);
  EndTest ();

  //  20          40       50     53      54       55        56   57
  //   | switching |  busy  | sifs | aifsn | bslot0 | bslot 1 | tx |
  //         |          |
  //        30 busy.   45 access request.
  //
  StartTest (1, 3, 10);
  AddTxop (1);
  AddSwitchingEvt (20,20);
  AddCcaBusyEvt (30,20);
  ExpectBackoff (45, 2, 0); //backoff: 2 slots
  AddAccessRequest (45, 1, 56, 0);
  EndTest ();

  //  20     30          50     51     54      55   56
  //   |  rx  | switching | idle | sifs | aifsn | tx |
  //                             ^ access request.
  //
  StartTest (1, 3, 10);
  AddTxop (1);
  AddRxStartEvt (20, 40);
  AddSwitchingEvt (30, 20);
  AddAccessRequest (51, 1, 55, 0);
  EndTest ();

  //  20     30          50     51     54      55   56
  //   | busy | switching | idle | sifs | aifsn | tx |
  //                             ^ access request.
  //
  StartTest (1, 3, 10);
  AddTxop (1);
  AddCcaBusyEvt (20, 40);
  AddSwitchingEvt (30, 20);
  AddAccessRequest (51, 1, 55, 0);
  EndTest ();

  //  20      30          50     51     54      55   56
  //   |  nav  | switching | idle | sifs | aifsn | tx |
  //                              ^ access request.
  //
  StartTest (1, 3, 10);
  AddTxop (1);
  AddNavStart (20,40);
  AddSwitchingEvt (30,20);
  AddAccessRequest (51, 1, 55, 0);
  EndTest ();

  //  20     23      24      44             54          59     60     63      64   65
  //   | sifs | aifsn |  tx   | Ack timeout  | switching | idle | sifs | aifsn | tx |
  //                                 |                          |
  //                                49 access request.          ^ access request.
  //
  StartTest (1, 3, 10);
  AddTxop (1);
  AddAccessRequestWithAckTimeout (20, 20, 24, 0);
  AddAccessRequest (49, 1, 54, 0);
  AddSwitchingEvt (54, 5);
  AddAccessRequest (60, 1, 64, 0);
  EndTest ();

  //  20         60     66      70       74       78  80         100    101    107     111  113
  //   |    rx    | sifs | aifsn | bslot0 | bslot1 |   | switching | idle | sifs | aifsn | tx |
  //        |                                                             |
  //       30 access request.                                             ^ access request.
  //
  StartTest (4, 6, 10);
  AddTxop (1);
  AddRxOkEvt (20,40);
  AddAccessRequest (30, 2, 80, 0);
  ExpectBackoff (30, 4, 0); //backoff: 4 slots
  AddSwitchingEvt (80,20);
  AddAccessRequest (101, 2, 111, 0);
  EndTest ();
}

/*
 * Specialization of DoRun () method for EDCA
 */
template <>
void
ChannelAccessManagerTest<QosTxop>::DoRun (void)
{
  // Check alignment at slot boundary after successful reception (backoff = 0):
  //    20     50     56      60     80
  //     |  rx  | sifs | aifsn |  tx  |
  //                |
  //               52 request access
  StartTest (4, 6, 10);
  AddTxop (1);
  AddRxOkEvt (20, 30);
  AddAccessRequest (52, 20, 60, 0);
  EndTest ();

  // Check alignment at slot boundary after successful reception (backoff = 0):
  //    20     50     56      60     80
  //     |  rx  | sifs | aifsn |  tx  |
  //                       |
  //                      58 request access
  StartTest (4, 6, 10);
  AddTxop (1);
  AddRxOkEvt (20, 30);
  AddAccessRequest (58, 20, 60, 0);
  EndTest ();

  // Check alignment at slot boundary after successful reception (backoff = 0):
  //    20     50     56      60     64     84
  //     |  rx  | sifs | aifsn | idle |  tx  |
  //                               |
  //                              62 request access
  StartTest (4, 6, 10);
  AddTxop (1);
  AddRxOkEvt (20, 30);
  AddAccessRequest (62, 20, 64, 0);
  EndTest ();

  // Check alignment at slot boundary after failed reception (backoff = 0):
  //  20         50     56           66             76     96
  //   |          | <------eifs------>|              |      |
  //   |    rx    | sifs | acktxttime | sifs + aifsn |  tx  |
  //                   |
  //                  55 request access
  StartTest (4, 6, 10);
  AddTxop (1);
  AddRxErrorEvt (20, 30);
  AddAccessRequest (55, 20, 76, 0);
  EndTest ();

  // Check alignment at slot boundary after failed reception (backoff = 0):
  //  20         50     56           66             76     96
  //   |          | <------eifs------>|              |      |
  //   |    rx    | sifs | acktxttime | sifs + aifsn |  tx  |
  //                                        |
  //                                       70 request access
  StartTest (4, 6, 10);
  AddTxop (1);
  AddRxErrorEvt (20, 30);
  AddAccessRequest (70, 20, 76, 0);
  EndTest ();

  // Check alignment at slot boundary after failed reception (backoff = 0):
  //  20         50     56           66             76     84
  //   |          | <------eifs------>|              |      |
  //   |    rx    | sifs | acktxttime | sifs + aifsn | idle |  tx  |
  //                                                     |
  //                                                    82 request access
  StartTest (4, 6, 10);
  AddTxop (1);
  AddRxErrorEvt (20, 30);
  AddAccessRequest (82, 20, 84, 0);
  EndTest ();

  // Check backoff decrement at slot boundaries. Medium idle during backoff
  //  20           50     56      60         64         68         72         76     96
  //   |     rx     | sifs | aifsn |   idle   |   idle   |   idle   |   idle   |  tx  |
  //      |                        |          |          |          |
  //     30 request access.    decrement  decrement  decrement  decrement
  //        backoff slots: 4    slots: 3   slots: 2   slots: 1   slots: 0
  StartTest (4, 6, 10);
  AddTxop (1);
  AddRxOkEvt (20, 30);
  AddAccessRequest (30, 20, 76, 0);
  ExpectBackoff (30, 4, 0);
  EndTest ();

  // Check backoff decrement at slot boundaries. Medium becomes busy during backoff
  //  20           50     56      60     61     71     77      81         85     87     97    103     107    127
  //   |     rx     | sifs | aifsn | idle |  rx  | sifs | aifsn |   idle   | idle |  rx  | sifs | aifsn |  tx  |
  //      |                        |                            |          |
  //     30 request access.    decrement                    decrement  decrement
  //        backoff slots: 3    slots: 2                     slots: 1   slots: 0
  StartTest (4, 6, 10);
  AddTxop (1);
  AddRxOkEvt (20, 30);
  AddRxOkEvt (61, 10);
  AddRxOkEvt (87, 10);
  AddAccessRequest (30, 20, 107, 0);
  ExpectBackoff (30, 3, 0);
  EndTest ();
}

/**
 * \ingroup wifi-test
 * \ingroup tests
 *
 * \brief Txop Test Suite
 */
class TxopTestSuite : public TestSuite
{
public:
  TxopTestSuite ();
};

TxopTestSuite::TxopTestSuite ()
  : TestSuite ("wifi-devices-dcf", UNIT)
{
  AddTestCase (new ChannelAccessManagerTest<Txop>, TestCase::QUICK);
}

static TxopTestSuite g_dcfTestSuite;

/**
 * \ingroup wifi-test
 * \ingroup tests
 *
 * \brief QosTxop Test Suite
 */
class QosTxopTestSuite : public TestSuite
{
public:
  QosTxopTestSuite ();
};

QosTxopTestSuite::QosTxopTestSuite ()
  : TestSuite ("wifi-devices-edca", UNIT)
{
  AddTestCase (new ChannelAccessManagerTest<QosTxop>, TestCase::QUICK);
}

static QosTxopTestSuite g_edcaTestSuite;
