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

#ifndef THREE_GPP_HTTP_CLIENT_H
#define THREE_GPP_HTTP_CLIENT_H

#include <ns3/application.h>
#include <ns3/address.h>
#include <ns3/traced-callback.h>
#include <ns3/three-gpp-http-header.h>


namespace ns3 {


class Socket;
class Packet;
class ThreeGppHttpVariables;


/**
 * \ingroup applications
 * \defgroup http ThreeGppHttpClientServer
 *
 * This traffic generator simulates web browsing traffic using the Hypertext
 * Transfer Protocol (HTTP). It consists of one or more ThreeGppHttpClient
 * applications which connect to an ThreeGppHttpServer application. The client
 * models a web browser which requests web pages to the server. The server
 * is then responsible to serve the web pages as requested. Please refer to
 * ThreeGppHttpClientHelper and ThreeGppHttpServerHelper for usage instructions.
 *
 * Technically speaking, the client transmits *request objects* to demand a
 * service from the server. Depending on the type of request received, the
 * server transmits either:
 *   - a *main object*, i.e., the HTML file of the web page; or
 *   - an *embedded object*, e.g., an image referenced by the HTML file.
 *
 * A major portion of the traffic pattern is *reading time*, which does not
 * generate any traffic. Because of this, one may need to simulate a good
 * number of clients and/or sufficiently long simulation duration in order to
 * generate any significant traffic in the system.
 *
 * Many aspects of the traffic are randomly determined by ThreeGppHttpVariables.
 * These characteristics are based on a legacy 3GPP specification. The description
 * can be found in the following references:
 *   - 3GPP TR 25.892, "Feasibility Study for Orthogonal Frequency Division
 *         Multiplexing (OFDM) for UTRAN enhancement"
 *   - IEEE 802.16m, "Evaluation Methodology Document (EMD)",
 *     IEEE 802.16m-08/004r5, July 2008.
 *   - NGMN Alliance, "NGMN Radio Access Performance Evaluation Methodology",
 *     v1.0, January 2008.
 *   - 3GPP2-TSGC5, "HTTP, FTP and TCP models for 1xEV-DV simulations", 2001.
 */

/**
 * \ingroup http
 * Model application which simulates the traffic of a web browser. This
 * application works in conjunction with an ThreeGppHttpServer application.
 *
 * In summary, the application works as follows.
 * 1. Upon start, it opens a connection to the destination web server
 *    (ThreeGppHttpServer).
 * 2. After the connection is established, the application immediately requests
 *    a *main object* from the server by sending a request packet.
 * 3. After receiving a main object (which can take some time if it consists of
 *    several packets), the application "parses" the main object.
 * 4. The parsing takes a short time (randomly determined) to determine the
 *    number of *embedded objects* (also randomly determined) in the web page.
 *    - If at least one embedded object is determined, the application requests
 *      the first embedded object from the server. The request for the next
 *      embedded object follows after the previous embedded object has been
 *      completely received.
 *    - If there is no more embedded object to request, the application enters
 *      the *reading time*.
 * 5. Reading time is a long delay (again, randomly determined) where the
 *    application does not induce any network traffic, thus simulating the user
 *    reading the downloaded web page.
 * 6. After the reading time is finished, the process repeats to step #2.
 *
 * The client models HTTP *persistent connection*, i.e., HTTP 1.1, where the
 * connection to the server is maintained and used for transmitting and receiving
 * all objects.
 *
 * Each request by default has a constant size of 350 bytes. A ThreeGppHttpHeader
 * is attached to each request packet. The header contains information
 * such as the content type requested (either main object or embedded object)
 * and the timestamp when the packet is transmitted (which will be used to
 * compute the delay and RTT of the packet).
 */
class ThreeGppHttpClient : public Application
{
public:
  /**
   * Creates a new instance of HTTP client application.
   *
   * After creation, the application must be further configured through
   * attributes. To avoid having to do this process manually, please use
   * ThreeGppHttpClientHelper.
   */
  ThreeGppHttpClient ();

  /**
   * Returns the object TypeId.
   * \return The object TypeId.
   */
  static TypeId GetTypeId ();

  /**
   * Returns a pointer to the associated socket.
   * \return Pointer to the associated socket.
   */
  Ptr<Socket> GetSocket () const;

  /// The possible states of the application.
  enum State_t
  {
    /// Before StartApplication() is invoked.
    NOT_STARTED = 0,
    /// Sent the server a connection request and waiting for the server to be accept it.
    CONNECTING,
    /// Sent the server a request for a main object and waiting to receive the packets.
    EXPECTING_MAIN_OBJECT,
    /// Parsing a main object that has just been received.
    PARSING_MAIN_OBJECT,
    /// Sent the server a request for an embedded object and waiting to receive the packets.
    EXPECTING_EMBEDDED_OBJECT,
    /// User reading a web page that has just been received.
    READING,
    /// After StopApplication() is invoked.
    STOPPED
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
   * Common callback signature for `ConnectionEstablished`, `RxMainObject`, and
   * `RxEmbeddedObject` trace sources.
   * \param httpClient Pointer to this instance of ThreeGppHttpClient,
   *                               which is where the trace originated.
   */
  typedef void (*TracedCallback)(Ptr<const ThreeGppHttpClient> httpClient);

protected:
  // Inherited from Object base class.
  virtual void DoDispose ();

  // Inherited from Application base class.
  virtual void StartApplication ();
  virtual void StopApplication ();

private:
  // SOCKET CALLBACK METHODS

  /**
   * Invoked when a connection is established successfully on #m_socket. This
   * triggers a request for a main object.
   * \param socket Pointer to the socket where the event originates from.
   */
  void ConnectionSucceededCallback (Ptr<Socket> socket);
  /**
   * Invoked when #m_socket cannot establish a connection with the web server.
   * Simulation will stop and error will be raised.
   * \param socket Pointer to the socket where the event originates from.
   */
  void ConnectionFailedCallback (Ptr<Socket> socket);
  /**
   * Invoked when connection between #m_socket and the web sever is terminated.
   * Error will be logged, but simulation continues.
   * \param socket Pointer to the socket where the event originates from.
   */
  void NormalCloseCallback (Ptr<Socket> socket);
  /**
   * Invoked when connection between #m_socket and the web sever is terminated.
   * Error will be logged, but simulation continues.
   * \param socket Pointer to the socket where the event originates from.
   */
  void ErrorCloseCallback (Ptr<Socket> socket);
  /**
   * Invoked when #m_socket receives some packet data. Fires the `Rx` trace
   * source and triggers ReceiveMainObject() or ReceiveEmbeddedObject().
   * \param socket Pointer to the socket where the event originates from.
   */
  void ReceivedDataCallback (Ptr<Socket> socket);

  // CONNECTION-RELATED METHOD

  /**
   * Initialize #m_socket to connect to the destination web server at
   * #m_remoteServerAddress and #m_remoteServerPort and set up callbacks to
   * listen to its event. Invoked upon the start of the application.
   */
  void OpenConnection ();

  // TX-RELATED METHODS

  /**
   * Send a request object for a main object to the destination web server.
   * The size of the request packet is randomly determined by HttpVariables and
   * is assumed to be smaller than 536 bytes. Fires the `TxMainObjectRequest`
   * trace source.
   *
   * The method is invoked after a connection is established or after a
   * reading time has elapsed.
   */
  void RequestMainObject ();
  /**
   * Send a request object for an embedded object to the destination web
   * server. The size of the request packet is randomly determined by
   * ThreeGppHttpVariables and is assumed to be smaller than 536 bytes. Fires the
   * `TxEmbeddedObjectRequest` trace source.
   */
  void RequestEmbeddedObject ();

  // RX-RELATED METHODS

  /**
   * Receive a packet of main object from the destination web server. Fires the
   * `RxMainObjectPacket` trace source.
   *
   * A main object may come from more than one packets. This is determined by
   * comparing the content length specified in the ThreeGppHttpHeader of the packet and
   * the actual packet size. #m_objectBytesToBeReceived keeps track of the
   * number of bytes that has been received.
   *
   * If the received packet is not the last packet of the object, then the
   * method simply quits, expecting it to be invoked again when the next packet
   * comes.
   *
   * If the received packet is the last packet of the object, then the method
   * fires the `RxMainObject`, `RxDelay`, and `RxRtt` trace sources. The client
   * then triggers EnterParsingTime().
   *
   * \param packet The received packet.
   * \param from Address of the sender.
   */
  void ReceiveMainObject (Ptr<Packet> packet, const Address &from);
  /**
   * Receive a packet of embedded object from the destination web server. Fires
   * the `RxEmbeddedObjectPacket` trace source.
   *
   * An embedded object may come from more than one packets. This is determined
   * by comparing the content length specified in the TheeGppHttpHeader of the packet and
   * the actual packet size. #m_objectBytesToBeReceived keeps track of the
   * number of bytes that has been received.
   *
   * If the received packet is not the last packet of the object, then the
   * method simply quits, expecting it to be invoked again when the next packet
   * comes.
   *
   * If the received packet is the last packet of the object, then the method
   * fires the `RxEmbeddedObject`, `RxDelay`, and `RxRtt` trace sources.
   * Depending on the number of embedded objects remaining
   * (#m_embeddedObjectsToBeRequested) the client can either trigger
   * RequestEmbeddedObject() or EnterReadingTime().
   *
   * \param packet The received packet.
   * \param from Address of the sender.
   */
  void ReceiveEmbeddedObject (Ptr<Packet> packet, const Address &from);
  /**
   * Simulate a consumption of the received packet by subtracting the packet
   * size from the internal counter at #m_objectBytesToBeReceived. Also updates
   * #m_objectClientTs and #m_objectServerTs according to the ThreeGppHttpHeader
   * found in the packet.
   *
   * This method is invoked as a sub-procedure of ReceiveMainObject() and
   * ReceiveEmbeddedObject().
   *
   * \param packet The received packet. If it is the first packet of the object,
   *               then it must have a ThreeGppHttpHeader attached to it.
   */
  void Receive (Ptr<Packet> packet);

  // OFF-TIME-RELATED METHODS

  /**
   * Becomes idle for a randomly determined amount of time, and then triggers
   * ParseMainObject(). The length of idle time is determined by
   * TheeGppHttpVariables.
   *
   * The method is invoked after a complete main object has been received.
   */
  void EnterParsingTime ();
  /**
   * Randomly determines the number of embedded objects in the main object.
   * ThreeGppHttpVariables is utilized for this purpose. Then proceeds with
   * RequestEmbeddedObject(). If the number of embedded objects has been
   * determined as zero, then EnterReadingTime() is triggered.
   *
   * The method is invoked after parsing time has elapsed.
   */
  void ParseMainObject ();
  /**
   * Becomes idle for a randomly determined amount of time, and then triggers
   * RequestMainObject(). The length of idle time is determined by
   * ThreeGppHttpVariables.
   *
   * The method is invoked after a complete web page has been received.
   */
  void EnterReadingTime ();
  /**
   * Cancels #m_eventRequestMainObject, #m_eventRequestEmbeddedObject, and
   * #m_eventParseMainObject. Invoked by StopApplication() and when connection
   * has been terminated.
   */
  void CancelAllPendingEvents ();

  /**
   * Change the state of the client. Fires the `StateTransition` trace source.
   * \param state The new state.
   */
  void SwitchToState (State_t state);

  /// The current state of the client application. Begins with NOT_STARTED.
  State_t      m_state;
  /// The socket for sending and receiving packets to/from the web server.
  Ptr<Socket>  m_socket;
  /// According to the content length specified by the ThreeGppHttpHeader.
  uint32_t     m_objectBytesToBeReceived;
  /// The packet constructed of one or more parts with ThreeGppHttpHeader.
  Ptr<Packet>  m_constructedPacket;
  /// The client time stamp of the ThreeGppHttpHeader from the last received packet.
  Time         m_objectClientTs;
  /// The server time stamp of the ThreeGppHttpHeader from the last received packet.
  Time         m_objectServerTs;
  /// Determined after parsing the main object.
  uint32_t     m_embeddedObjectsToBeRequested;

  // ATTRIBUTES

  /// The `Variables` attribute.
  Ptr<ThreeGppHttpVariables>    m_httpVariables;
  /// The `RemoteServerAddress` attribute. The address of the web server.
  Address                               m_remoteServerAddress;
  /// The `RemoteServerPort` attribute.
  uint16_t                              m_remoteServerPort;

  // TRACE SOURCES

  /// The `ConnectionEstablished` trace source.
  ns3::TracedCallback<Ptr<const ThreeGppHttpClient> > m_connectionEstablishedTrace;
  /// The `ConnectionClosed` trace source.
  ns3::TracedCallback<Ptr<const ThreeGppHttpClient> > m_connectionClosedTrace;
  /// The `Tx` trace source.
  ns3::TracedCallback<Ptr<const Packet> >   m_txTrace;
  /// The `TxMainObjectRequest` trace source.
  ns3::TracedCallback<Ptr<const Packet> >     m_txMainObjectRequestTrace;
  /// The `TxEmbeddedObjectRequest` trace source.
  ns3::TracedCallback<Ptr<const Packet> >     m_txEmbeddedObjectRequestTrace;
  /// The `TxMainObjectPacket` trace source.
  ns3::TracedCallback<Ptr<const Packet> >     m_rxMainObjectPacketTrace;
  /// The `TxMainObject` trace source.
  ns3::TracedCallback<Ptr<const ThreeGppHttpClient>, Ptr<const Packet>  > m_rxMainObjectTrace;
  /// The `TxEmbeddedObjectPacket` trace source.
  ns3::TracedCallback<Ptr<const Packet> >     m_rxEmbeddedObjectPacketTrace;
  /// The `TxEmbeddedObject` trace source.
  ns3::TracedCallback<Ptr<const ThreeGppHttpClient>, Ptr<const Packet> > m_rxEmbeddedObjectTrace;
  /// The `Rx` trace source.
  ns3::TracedCallback<Ptr<const Packet>, const Address &> m_rxTrace;
  /// The `RxDelay` trace source.
  ns3::TracedCallback<const Time &, const Address &>  m_rxDelayTrace;
  /// The `RxRtt` trace source.
  ns3::TracedCallback<const Time &, const Address &>  m_rxRttTrace;
  /// The `StateTransition` trace source.
  ns3::TracedCallback<const std::string &, const std::string &> m_stateTransitionTrace;

  // EVENTS

  /**
   * An event of either RequestMainObject() or OpenConnection(), scheduled to
   * trigger after a connection has been established or reading time has
   * elapsed.
   */
  EventId m_eventRequestMainObject;
  /**
   * An event of either RequestEmbeddedObject() or OpenConnection().
   */
  EventId m_eventRequestEmbeddedObject;
  /**
   * An event of ParseMainObject(), scheduled to trigger after parsing time has
   * elapsed.
   */
  EventId m_eventParseMainObject;

}; // end of `class ThreeGppHttpClient`


}  // end of `namespace ns3`


#endif /* THREE_GPP_HTTP_CLIENT_H */
