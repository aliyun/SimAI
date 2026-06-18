from gi.repository import Gtk
import ns.core
import ns.network
from visualizer.base import InformationWindow

NODE_STATISTICS_MEMORY = 10


## StatisticsCollector class
class StatisticsCollector(object):
    """
    Collects interface statistics for all nodes.
    """
    ## @var node_statistics
    #  node statistics
    ## @var visualizer
    #  visualizer

    ## NetDevStats class
    class NetDevStats(object):
        ## class members
        __slots__ = ['rxPackets', 'rxBytes', 'txPackets', 'txBytes',
                     'rxPacketRate', 'rxBitRate', 'txPacketRate', 'txBitRate']

    def __init__(self, visualizer):
        """!
        Collects interface statistics for all nodes.
        @param self this object
        @param visualizer visualizer object
        """
        self.node_statistics = {} # nodeid -> list(raw statistics)
        self.visualizer = visualizer

    def simulation_periodic_update(self, viz):
        """!
        Simulation Periodic Update function.
        @param self this object
        @param viz visualizer object
        @return none
        """
        nodes_statistics = viz.simulation.sim_helper.GetNodesStatistics()
        for stats in nodes_statistics:
            try:
                raw_stats_list = self.node_statistics[stats.nodeId]
            except KeyError:
                raw_stats_list = []
                self.node_statistics[stats.nodeId] = raw_stats_list
            raw_stats_list.append(stats.statistics)
            while len(raw_stats_list) > NODE_STATISTICS_MEMORY:
                raw_stats_list.pop(0)

    def get_interface_statistics(self, nodeId):
        """!
        Get interface statistics function.
        @param self this object
        @param nodeId node ID
        @return the statistics
        """
        try:
            raw_stats_list = self.node_statistics[nodeId]
        except KeyError:
            return []

        if len(raw_stats_list) < NODE_STATISTICS_MEMORY:
            return []
        assert len(raw_stats_list) == NODE_STATISTICS_MEMORY
        tx_packets1 = [] # transmitted packets, one value per interface
        rx_packets1 = []
        tx_bytes1 = []
        rx_bytes1 = []
        for iface, stats in enumerate(raw_stats_list[0]):
            tx_packets1.append(stats.transmittedPackets)
            tx_bytes1.append(stats.transmittedBytes)
            rx_packets1.append(stats.receivedPackets)
            rx_bytes1.append(stats.receivedBytes)

        retval = []

        k = self.visualizer.sample_period*(NODE_STATISTICS_MEMORY-1)
        for iface, stats in enumerate(raw_stats_list[-1]):
            outStat = self.NetDevStats()
            outStat.txPackets = stats.transmittedPackets
            outStat.txBytes = stats.transmittedBytes
            outStat.rxPackets = stats.receivedPackets
            outStat.rxBytes = stats.receivedBytes

            outStat.txPacketRate = (stats.transmittedPackets - tx_packets1[iface])/k
            outStat.rxPacketRate = (stats.receivedPackets - rx_packets1[iface])/k
            outStat.txBitRate = (stats.transmittedBytes - tx_bytes1[iface])*8/k
            outStat.rxBitRate = (stats.receivedBytes - rx_bytes1[iface])*8/k
            retval.append(outStat)
        return retval


## ShowInterfaceStatistics class
class ShowInterfaceStatistics(InformationWindow):
    ## @var win
    #  window
    ## @var visualizer
    #  visualizer
    ## @var statistics_collector
    #  statistics collector
    ## @var node_index
    #  node index
    ## @var viz_node
    #  visualizer node
    ## @var table_model
    #  table model
    (
        COLUMN_INTERFACE,

        COLUMN_TX_PACKETS,
        COLUMN_TX_BYTES,
        COLUMN_TX_PACKET_RATE,
        COLUMN_TX_BIT_RATE,

        COLUMN_RX_PACKETS,
        COLUMN_RX_BYTES,
        COLUMN_RX_PACKET_RATE,
        COLUMN_RX_BIT_RATE,

        ) = range(9)

    def __init__(self, visualizer, node_index, statistics_collector):
        """!
        Initializer.
        @param self this object
        @param visualizer the visualizer object
        @param node_index the node index
        @param statistics_collector statistics collector class
        """
        InformationWindow.__init__(self)
        self.win = Gtk.Dialog(parent=visualizer.window,
                              flags=Gtk.DialogFlags.DESTROY_WITH_PARENT,
                              buttons=(Gtk.STOCK_CLOSE, Gtk.ResponseType.CLOSE))
        self.win.connect("response", self._response_cb)
        self.win.set_title("Statistics for node %i" % node_index)
        self.visualizer = visualizer
        self.statistics_collector = statistics_collector
        self.node_index = node_index
        self.viz_node = visualizer.get_node(node_index)

        self.table_model = Gtk.ListStore(*([str]*13))

        treeview = Gtk.TreeView(self.table_model)
        treeview.show()
        self.win.vbox.add(treeview)

        def add_column(descr, colid):
            column = Gtk.TreeViewColumn(descr, Gtk.CellRendererText(), text=colid)
            treeview.append_column(column)

        add_column("Interface", self.COLUMN_INTERFACE)

        add_column("Tx Packets", self.COLUMN_TX_PACKETS)
        add_column("Tx Bytes", self.COLUMN_TX_BYTES)
        add_column("Tx pkt/1s", self.COLUMN_TX_PACKET_RATE)
        add_column("Tx bit/1s", self.COLUMN_TX_BIT_RATE)

        add_column("Rx Packets", self.COLUMN_RX_PACKETS)
        add_column("Rx Bytes", self.COLUMN_RX_BYTES)
        add_column("Rx pkt/1s", self.COLUMN_RX_PACKET_RATE)
        add_column("Rx bit/1s", self.COLUMN_RX_BIT_RATE)

        self.visualizer.add_information_window(self)
        self.win.show()

    def _response_cb(self, win, response):
        """!
        Response callback function.
        @param self this object
        @param win the window
        @param response the response
        @return none
        """
        self.win.destroy()
        self.visualizer.remove_information_window(self)

    def update(self):
        """!
        Update function.
        @param self this object
        @return none
        """
        node = ns.network.NodeList.GetNode(self.node_index)
        stats_list = self.statistics_collector.get_interface_statistics(self.node_index)
        self.table_model.clear()
        for iface, stats in enumerate(stats_list):
            tree_iter = self.table_model.append()
            netdevice = node.GetDevice(iface)
            interface_name = ns.core.Names.FindName(netdevice)
            if not interface_name:
                interface_name = "(interface %i)" % iface
            self.table_model.set(tree_iter,
                                 self.COLUMN_INTERFACE, interface_name,

                                 self.COLUMN_TX_PACKETS, str(stats.txPackets),
                                 self.COLUMN_TX_BYTES, str(stats.txBytes),
                                 self.COLUMN_TX_PACKET_RATE, str(stats.txPacketRate),
                                 self.COLUMN_TX_BIT_RATE, str(stats.txBitRate),

                                 self.COLUMN_RX_PACKETS, str(stats.rxPackets),
                                 self.COLUMN_RX_BYTES, str(stats.rxBytes),
                                 self.COLUMN_RX_PACKET_RATE, str(stats.rxPacketRate),
                                 self.COLUMN_RX_BIT_RATE, str(stats.rxBitRate)
                                 )


def populate_node_menu(viz, node, menu, statistics_collector):

    menu_item = Gtk.MenuItem("Show Interface Statistics")
    menu_item.show()

    def _show_it(dummy_menu_item):
        ShowInterfaceStatistics(viz, node.node_index, statistics_collector)

    menu_item.connect("activate", _show_it)
    menu.add(menu_item)


def register(viz):
    statistics_collector = StatisticsCollector(viz)
    viz.connect("populate-node-menu", populate_node_menu, statistics_collector)
    viz.connect("simulation-periodic-update", statistics_collector.simulation_periodic_update)
