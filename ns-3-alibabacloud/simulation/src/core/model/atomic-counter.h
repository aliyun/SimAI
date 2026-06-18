/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
#ifndef ATOMIC_COUNTER_H
#define ATOMIC_COUNTER_H

#include <atomic>

namespace ns3 {

class AtomicCounter
{
public:
  inline AtomicCounter ()
  {
    m_count.store (0, std::memory_order_release);
  }

  inline AtomicCounter (uint32_t count)
  {
    m_count.store (count, std::memory_order_release);
  }

  inline operator uint32_t () const
  {
    return m_count.load (std::memory_order_acquire);
  }

  inline uint32_t operator = (const uint32_t count)
  {
    m_count.store (count, std::memory_order_release);
    return count;
  }

  inline uint32_t operator++ (int)
  {
    return m_count.fetch_add (1, std::memory_order_relaxed);
  }

  inline uint32_t operator-- (int)
  {
    return m_count.fetch_sub (1, std::memory_order_release);
  }

private:
  std::atomic<uint32_t> m_count;
};

} // namespace ns3

#endif /* ATOMIC_COUNTER_H */
