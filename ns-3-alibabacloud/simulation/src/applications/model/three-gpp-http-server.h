/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2013 Magister Solutions
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
 * Author: Budiarto Herman <budiarto.herman@magister.fi>
 *
 */

#ifndef THREE_GPP_HTTP_SERVER_H
#define THREE_GPP_HTTP_SERVER_H

#include <ns3/ptr.h>
#include <ns3/simple-ref-count.h>
#include <ns3/nstime.h>
#include <ns3/event-id.h>
#include <ns3/three-gpp-http-header.h>
#include <ns3/application.h>
#include <ns3/address.h>
#include <ns3/traced-callback.h>
#include <map>
#include <ostream>


namespace ns3 {


class Socket;
class Packet;
class ThreeGppHttpVariables;
class ThreeGppHttpServerTxBuffer;


/**
 * \ingroup http
 * Model application which simulates the traffic of a web server. This
 * application works in conjunction with ThreeGppHttpClient applications.
 *
 * The application works by responding to requests. Each request is a small
 * packet of data which contains ThreeGppHttpHeader. The value of the *content
 * type* field of the header determines the type of object that the client is
 * requesting. The possible type is either a *main object* or an *embedded
 * object*.
 *
 * The application is responsible to generate the right type of object and send
 * it back to the client. The size of each object to be sent is randomly
 * determined (see ThreeGppHttpVariables). Each object may be sent as multiple packets
 * due to limited socket buffer space.
 *
 * To assist with the transmission, the application maintains several instances
 * of ThreeGppHttpServerTxBuffer. Each instance keeps track of the object type to be
 * served and the number of bytes left to be sent.
 *
 * The application accepts connection request from clients. Every connection is
 * kept open until the client disconnects.
 */
class ThreeGppHttpServer : public Application
{
public:
  /**
   * Creates a new instance of HTTP server application.
   *
   * After creation, the application must be further configured through
   * attributes. To avoid having to do this process manually, please use
   * ThreeGppHttpServerHelper.
   *
   * Upon creation, the application randomly determines the MTU size that it
   * will use (either 536 or 1460 bytes). The chosen size will be used while
   * creating the listener socket.
   */
  ThreeGppHttpServer ();

  /**
   * Returns the object TypeId.
   * \return The object TypeId.
   */
  static TypeId GetTypeId ();

  /**
   * Sets the maximum transmission unit (MTU) size used by the application.
   *
   * This overrides the MTU size which is randomly determined once the
   * application is created. Values other than the standard 536 and 1460 bytes
   * can be set using this method.
   *
   * \param mtuSize MTU size in bytes.
   */
  void SetMtuSize (uint32_t mtuSize);

  /**
   * Returns a pointer to the listening socket.
   * \return Pointer to the listening socket
   */
  Ptr<Socket> GetSocket () const;

  /// The possible states of the application.
  enum State_t
  {
    NOT_STARTED = 0,  ///< Before StartApplication() is invoked.
    STARTED,          ///< Passively listening and responding to requests.
    STOPPED           ///< After StopApplication() is invoked.
  };

  /**
   * Returns the current state of the application.
   * \return The current state of the application.
   */
  State_t GetState () const;

  /**
   * Returns the current state of the application in string format.
   * \return The current state of the application in string format.
   */
  std::string GetStateString () const;

  /**
   * Returns the given state in string format.
   * \param state An arbitrary state of an application.
   * \return The given state equivalently expressed in string format.
   */
  static std::string GetStateString (State_t state);

  /**
   * Common callback signature for `MainObject` and `EmbeddedObject` trace
   * sources.
   * \param size Size of the generated object in bytes.
   */
  typedef void (*ThreeGppHttpObjectCallback)(uint32_t size);

  /**
   * Callback signature for `ConnectionEstablished` trace source.
   * \param httpServer Pointer to this instance of ThreeGppHttpServer, which is where
   *                   the trace originated.
   * \param socket Pointer to the socket where the connection is established.
   */
  typedef void (*ConnectionEstablishedCallback)(Ptr<const ThreeGppHttpServer> httpServer,
                                                Ptr<Socket>                   socket);

protected:
  // Inherited from Object base class
  virtual void DoDispose ();

  // Inherited from Application base class
  virtual void StartApplication ();
  virtual void StopApplication ();

private:
  // SOCKET CALLBACK METHODS

  /**
   * Invoked when #m_initialSocket receives a connection request.
   * \param socket Pointer to the socket where the event originates from.
   * \param address The address of the remote client where the connection
   *                request comes from.
   * \return Always true, to indicate to the other end that the connection
   *         request is accepted.
   */
  bool ConnectionRequestCallback (Ptr<Socket>   socket,
                                  const Address &address);
  /**
   * Invoked when a new connection has been established.
   * \param socket Pointer to the socket that maintains the connection to the
   *               remote client. This socket will be saved to the Tx buffer.
   * \param address The address the connection is incoming from.
   */
  void NewConnectionCreatedCallback (Ptr<Socket>    socket,
                                     const Address  &address);
  /**
   * Invoked when a connection with a web client is terminated. The
   * corresponding socket will be removed from Tx buffer.
   * \param socket Pointer to the socket where the event originates from.
   */
  void NormalCloseCallback (Ptr<Socket> socket);
  /**
   * Invoked when a connection with a web client is terminated. The
   * corresponding socket will be removed from Tx buffer.
   * \param socket Pointer to the socket where the event originates from.
   */
  void ErrorCloseCallback (Ptr<Socket> socket);
  /**
   * Invoked when #m_initialSocket receives some packet data. It will check the
   * packet for ThreeGppHttpHeader. It also fires the `Rx` trace source.
   *
   * Depending on the type of object requested, the method will trigger
   * ServeMainObject() or ServeEmbeddedObject() after some delays.
   *
   * \param socket Pointer to the socket where the event originates from.
   */
  void ReceivedDataCallback (Ptr<Socket> socket);
  /**
   * Invoked when more buffer space for transmission is added to a socket. The
   * method will invoke ServeFromTxBuffer() to start some transmission using
   * the socket.
   * \param socket Pointer to the socket where the event originates from.
   * \param availableBufferSize The number of bytes available in the socket's
   *                            transmission buffer.
   */
  void SendCallback (Ptr<Socket> socket, uint32_t availableBufferSize);

  // TX-RELATED METHODS

  /**
   * Generates a new main object and push it into the Tx buffer.
   *
   * The size of the object is randomly determined by ThreeGppHttpVariables.
   * Fires the `MainObject` trace source. It then immediately triggers
   * ServeFromTxBuffer() to send the object.
   *
   * \param socket Pointer to the socket which is associated with the
   *               destination client.
   */
  void ServeNewMainObject (Ptr<Socket> socket);
  /**
   * Generates a new embedded object and push it into the Tx buffer.
   *
   * The size of the object is randomly determined by ThreeGppHttpVariables.
   * Fires the `EmbeddedObject` trace source. It then immediately triggers
   * ServeFromTxBuffer() to send the object.
   *
   * \param socket Pointer to the socket which is associated with the
   *               destination client.
   */
  void ServeNewEmbeddedObject (Ptr<Socket> socket);
  /**
   * Creates a packet out of a pending object in the Tx buffer send it over the
   * given socket. If the socket capacity is smaller than the object size, then
   * the method only convert a part of the object into a packet.
   *
   * ThreeGppHttpHeader will be attached in the beginning of each application
   * layer packet - if a packet is split, then then the following parts will
   * not have the header. The method fires the `Tx` trace source after sending
   * the packet to the socket.
   *
   * This method is invoked when a new object is generated by
   * ServeNewMainObject() or ServeNewEmbeddedObject(). It's also invoked when
   * the socket informs (through SendCallback()) that more buffer space for
   * transmission has become available.
   *
   * \param socket Pointer to the socket which is associated with the
   *               destination client.
   * \return Size of the packet sent (in bytes).
   */
  uint32_t ServeFromTxBuffer (Ptr<Socket> socket);

  /**
   * Change the state of the server. Fires the `StateTransition` trace source.
   * \param state The new state.
   */
  void SwitchToState (State_t state);

  /// The current state of the client application. Begins with NOT_STARTED.
  State_t                  m_state;
  /// The listening socket, for receiving connection requests from clients.
  Ptr<Socket>              m_initialSocket;
  /// Pointer to the transmission buffer.
  Ptr<ThreeGppHttpServerTxBuffer>  m_txBuffer;

  // ATTRIBUTES

  /// The `Variables` attribute.
  Ptr<ThreeGppHttpVariables>  m_httpVariables;
  /// The `LocalAddress` attribute.
  Address                     m_localAddress;
  /// The `LocalPort` attribute.
  uint16_t                    m_localPort;
  /// The `Mtu` attribute.
  uint32_t                    m_mtuSize;

  // TRACE SOURCES

  /// The `ConnectionEstablished` trace source.
  TracedCallback<Ptr<const ThreeGppHttpServer>, Ptr<Socket> > m_connectionEstablishedTrace;
  /// The `MainObject` trace source.
  TracedCallback<uint32_t> m_mainObjectTrace;
  /// The `EmbeddedObject` trace source.
  TracedCallback<uint32_t> m_embeddedObjectTrace;
  /// The `Tx` trace source.
  TracedCallback<Ptr<const Packet> > m_txTrace;
  /// The `Rx` trace source.
  TracedCallback<Ptr<const Packet>, const Address &> m_rxTrace;
  /// The `RxDelay` trace source.
  TracedCallback<const Time &, const Address &> m_rxDelayTrace;
  /// The `StateTransition` trace source.
  TracedCallback<const std::string &, const std::string &> m_stateTransitionTrace;

}; // end of `class ThreeGppHttpServer`


/**
 * \internal
 * \ingroup http
 * Transmission buffer used by an HTTP server instance.
 *
 * The class handles the sockets which face the connected HTTP clients. An
 * individual buffer is allocated for each socket. The buffer indicates the
 * length (in bytes) and the type of the data within, i.e., it does *not*
 * contain the actual packet data.
 *
 * Types of data are expressed using the ThreeGppHttpHeader::ContentType_t type. Only one
 * type of data can be active for one client at a time, i.e., the current
 * content of a buffer has to be removed before a different type of data can
 * be added.
 */
class ThreeGppHttpServerTxBuffer : public SimpleRefCount<ThreeGppHttpServerTxBuffer>
{
public:
  /// Create an empty instance of transmission buffer.
  ThreeGppHttpServerTxBuffer ();

  // SOCKET MANAGEMENT

  /**
   * This method is typically used before calling other methods. For example,
   * AddSocket() requires that the given socket does not exist among the stored
   * buffers. On the other hand, all the other methods that accept a pointer to
   * a socket as an argument require the existence of a buffer allocated to the
   * given socket.
   * \param socket Pointer to the socket to be found.
   * \return True if the given socket is found within the buffer.
   */
  bool IsSocketAvailable (Ptr<Socket> socket) const;

  /**
   * Add a new socket and create an empty transmission buffer for it. After the
   * method is completed, IsSocketAvailable() for the same pointer of socket
   * shall return true.
   * \param socket Pointer to the new socket to be added (must not exist in the
   *               buffer).
   * \warning Must be called only when IsSocketAvailable() for the given socket
   *          is false.
   */
  void AddSocket (Ptr<Socket> socket);

  /**
   * Remove a socket and its associated transmission buffer, and then unset the
   * socket's callbacks to prevent further interaction with the socket. If the
   * socket has a pending transmission event, it will be canceled.
   *
   * This method is useful for discarding a socket which is already closed,
   * e.g., by the HTTP client. This is due to the fact that double closing of a
   * socket may introduce undefined behaviour.
   *
   * After the method is completed, IsSocketAvailable() for the same pointer of
   * socket shall return false.
   *
   * \param socket Pointer to the socket to be removed.
   * \warning Must be called only when IsSocketAvailable() for the given socket
   *          is true.
   */
  void RemoveSocket (Ptr<Socket> socket);

  /**
   * Close and remove a socket and its associated transmission buffer, and then
   * unset the socket's callback to prevent further interaction with the
   * socket.
   *
   * This method is similar with RemoveSocket(), except that the latter does
   * not close the socket.
   *
   * After the method is completed, IsSocketAvailable() for the same pointer of
   * socket shall return false.
   *
   * \param socket Pointer to the socket to be closed and removed.
   * \warning Must be called only when IsSocketAvailable() for the given socket
   *          is true.
   */
  void CloseSocket (Ptr<Socket> socket);

  /**
   * Close and remove all stored sockets, hence clearing the buffer.
   */
  void CloseAllSockets ();

  // BUFFER MANAGEMENT

  /**
   * \param socket Pointer to the socket which is associated with the
   *               transmission buffer of interest.
   * \return True if the current length of the transmission buffer is zero,
   *         i.e., no pending packet.
   * \warning Must be called only when IsSocketAvailable() for the given socket
   *          is true.
   */
  bool IsBufferEmpty (Ptr<Socket> socket) const;

  /**
   * \param socket Pointer to the socket which is associated with the
   *               transmission buffer of interest
   * \return The client time stamp that comes from the last request packet
   *         received by the given socket. It indicates the time the request
   *         packet was transmitted by the client.
   */
  Time GetClientTs (Ptr<Socket> socket) const;

  /**
   * Returns ThreeGppHttpHeader::NOT_SET when the buffer is new and never been filled
   * with any data before. Otherwise, returns either ThreeGppHttpHeader::MAIN_OBJECT
   * or ThreeGppHttpHeader::EMBEDDED_OBJECT.
   * \param socket Pointer to the socket which is associated with the
   *               transmission buffer of interest
   * \return The content type of the current data inside the transmission
   *         buffer.
   * \warning Must be called only when IsSocketAvailable() for the given socket
   *          is true.
   */
  ThreeGppHttpHeader::ContentType_t GetBufferContentType (Ptr<Socket> socket) const;

  /**
   * \param socket Pointer to the socket which is associated with the
   *               transmission buffer of interest
   * \return The length (in bytes) of the current data inside the transmission
   *         buffer.
   * \warning Must be called only when IsSocketAvailable() for the given socket
   *          is true.
   */
  uint32_t GetBufferSize (Ptr<Socket> socket) const;

  /**
   * \param socket pointer to the socket which is associated with the
   *               transmission buffer of interest
   * \return true if the buffer content has been read since it is written
   *
   * \warning Must be called only when IsSocketAvailable() for the given socket
   *          is true.
   *
   * This method returns true after WriteNewObject() method is called. It
   * becomes false after DepleteBufferSize() method is called.
   */
  bool HasTxedPartOfObject (Ptr<Socket> socket) const;

  /**
   * Writes a data representing a new main object or embedded object to the
   * transmission buffer.
   *
   * The stored data can be later consumed wholly of partially by
   * DepleteBufferSize() method.
   *
   * \param socket Pointer to the socket which is associated with the
   *               transmission buffer of interest.
   * \param contentType The content-type of the data to be written (must not
   *                    equal to ThreeGppHttpHeader:NOT_SET).
   * \param objectSize The length (in bytes) of the new object to be created
   *                   (must be greater than zero).
   * \warning Must be called only when both IsSocketAvailable() and
   *          IsBufferEmpty() for the given socket are true.
   */
  void WriteNewObject (Ptr<Socket>                    socket,
                       ThreeGppHttpHeader::ContentType_t contentType,
                       uint32_t                       objectSize);

  /**
   * Informs about a pending transmission event associated with the socket, so
   * that it would be automatically canceled in case the socket is closed.
   *
   * The method also indicates the time stamp given by the client. The time
   * stamp will be included in every packet sent.
   *
   * \param socket pointer to the socket which is associated with the
   *               transmission buffer of interest
   * \param eventId the event to be recorded, e.g., the return value of
   *                Simulator::Schedule function
   * \param clientTs client time stamp
   *
   * \warning Must be called only when IsSocketAvailable() for the given socket
   *          is true.
   */
  void RecordNextServe (Ptr<Socket>   socket,
                        const EventId &eventId,
                        const Time    &clientTs);

  /**
   * Decrements a buffer size by a given amount.
   *
   * The content type of the object to be consumed can be inquired beforehand
   * by the GetBufferContentType() method.
   *
   * If the method has consumed all the remaining bytes within the buffer,
   * IsBufferEmpty() for the buffer shall return true.
   *
   * \param socket Pointer to the socket which is associated with the
   *               transmission buffer of interest.
   * \param amount The length (in bytes) to be consumed (must be greater than
   *               zero).
   *
   * \warning Must be called only when IsSocketAvailable() for the given socket
   *          is true. In addition, the requested amount must be larger than
   *          the current buffer size, which can be checked by calling the
   *          GetBufferSize() method.
   */
  void DepleteBufferSize (Ptr<Socket> socket, uint32_t amount);

  /**
   * Tell the buffer to close the associated socket once the buffer becomes
   * empty.
   * \param socket Pointer to the socket which is associated with the
   *               transmission buffer of interest.
   * \warning Must be called only when IsSocketAvailable() for the given socket
   *          is true.
   */
  void PrepareClose (Ptr<Socket> socket);

private:
  /**
   * Set of fields representing a single transmission buffer, which will be
   * associated with a socket.
   */
  struct TxBuffer_t
  {
    /**
     * Pending transmission event which will be automatically canceled when the
     * associated socket is closed.
     */
    EventId nextServe;
    /**
     * The client time stamp that comes from the request packet. This value
     * will be set in ThreeGppHttpHeader of every corresponding response packet sent, to
     * be used by the client to compute round trip delay time (RTT).
     */
    Time clientTs;
    /**
     * The content type of the current data inside the transmission buffer.
     * Accessible using the GetBufferContentType() method.
     */
    ThreeGppHttpHeader::ContentType_t txBufferContentType;
    /**
     * The length (in bytes) of the current data inside the transmission
     * buffer. Accessible using the GetBufferSize() method.
     */
    uint32_t txBufferSize;
    /**
     * True if the remote end has issued a request to close, which means that
     * this socket will immediately closes itself once the buffer becomes
     * empty.
     */
    bool isClosing;
    /**
     * \brief True if the buffer content has been read since it is written.
     *        Accessible using the HasTxedPartOfObject() method.
     */
    bool hasTxedPartOfObject;
  };

  /// Collection of accepted sockets and its individual transmission buffer.
  std::map<Ptr<Socket>, TxBuffer_t> m_txBuffer;

}; // end of `class ThreeGppHttpServerTxBuffer`


}  // end of `namespace ns3`


#endif /* THREE_GPP_HTTP_SERVER_H */
