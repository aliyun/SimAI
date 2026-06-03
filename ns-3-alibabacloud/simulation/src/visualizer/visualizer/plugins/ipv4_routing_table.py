from gi.repository import Gtk

import ns.core
import ns.network
import ns.internet

from visualizer.base import InformationWindow

## ShowIpv4RoutingTable class
class ShowIpv4RoutingTable(InformationWindow):
    ## @var win
    #  window
    ## @var visualizer
    #  visualizer
    ## @var node_index
    #  node index
    ## @var table_model
    #  table model
    (
        COLUMN_DESTINATION,
        COLUMN_NEXT_HOP,
        COLUMN_INTERFACE,
        COLUMN_TYPE,
        COLUMN_PRIO
        ) = range(5)

    def __init__(self, visualizer, node_index):
        """!
        Initializer
        @param self this object
        @param visualizer visualizer object
        @param node_index the node index
        """
        InformationWindow.__init__(self)
        self.win = Gtk.Dialog(parent=visualizer.window,
                              flags=Gtk.DialogFlags.DESTROY_WITH_PARENT,
                              buttons=(Gtk.STOCK_CLOSE, Gtk.ResponseType.CLOSE))
        self.win.connect("response", self._response_cb)
        self.win.set_title("IPv4 routing table for node %i" % node_index)
        self.visualizer = visualizer
        self.node_index = node_index

        self.table_model = Gtk.ListStore(str, str, str, str, int)

        treeview = Gtk.TreeView(self.table_model)
        treeview.show()
        sw = Gtk.ScrolledWindow()
        sw.set_properties(hscrollbar_policy=Gtk.PolicyType.AUTOMATIC,
                          vscrollbar_policy=Gtk.PolicyType.AUTOMATIC)
        sw.show()
        sw.add(treeview)
        self.win.vbox.add(sw)
        self.win.set_default_size(600, 300)

        # Dest.
        column = Gtk.TreeViewColumn('Destination', Gtk.CellRendererText(),
                                    text=self.COLUMN_DESTINATION)
        treeview.append_column(column)

        # Next hop
        column = Gtk.TreeViewColumn('Next hop', Gtk.CellRendererText(),
                                    text=self.COLUMN_NEXT_HOP)
        treeview.append_column(column)

        # Interface
        column = Gtk.TreeViewColumn('Interface', Gtk.CellRendererText(),
                                    text=self.COLUMN_INTERFACE)
        treeview.append_column(column)

        # Type
        column = Gtk.TreeViewColumn('Type', Gtk.CellRendererText(),
                                    text=self.COLUMN_TYPE)
        treeview.append_column(column)

        # Prio
        column = Gtk.TreeViewColumn('Prio', Gtk.CellRendererText(),
                                    text=self.COLUMN_PRIO)
        treeview.append_column(column)

        self.visualizer.add_information_window(self)
        self.win.show()

    def _response_cb(self, win, response):
        """!
        Response callback function
        @param self this object
        @param win the window
        @param response the response
        @return none
        """
        self.win.destroy()
        self.visualizer.remove_information_window(self)

    def update(self):
        """!
        Update function
        @param self this object
        @return none
        """
        node = ns.network.NodeList.GetNode(self.node_index)
        ipv4 = node.GetObject(ns.internet.Ipv4.GetTypeId())
        routing = ipv4.GetRoutingProtocol()
        if routing is None:
            return

        routing_protocols = [] # list of (protocol, type_string, priority)

        if isinstance(routing, ns.internet.Ipv4StaticRouting):
            ipv4_routing = routing_protocols.append((routing, "static", 0))
        elif isinstance(routing, ns.internet.Ipv4ListRouting):
            list_routing = routing
            for rI in range(list_routing.GetNRoutingProtocols()):
                routing, prio = list_routing.GetRoutingProtocol(rI)
                if isinstance(routing, ns.internet.Ipv4StaticRouting):
                    routing_protocols.append((routing, "static", prio))
                elif isinstance(routing, ns.internet.Ipv4GlobalRouting):
                    routing_protocols.append((routing, "global", prio))
        if not routing_protocols:
            return

        self.table_model.clear()
        for route_proto, type_string, prio in routing_protocols:
            for routeI in range(route_proto.GetNRoutes()):
                route = route_proto.GetRoute(routeI)
                tree_iter = self.table_model.append()
                netdevice = ipv4.GetNetDevice(route.GetInterface())
                if netdevice is None:
                    interface_name = 'lo'
                else:
                    interface_name = ns.core.Names.FindName(netdevice)
                    if not interface_name:
                        interface_name = "(interface %i)" % route.GetInterface()
                self.table_model.set(tree_iter,
                                     self.COLUMN_DESTINATION, str(route.GetDest()),
                                     self.COLUMN_NEXT_HOP, str(route.GetGateway()),
                                     self.COLUMN_INTERFACE, interface_name,
                                     self.COLUMN_TYPE, type_string,
                                     self.COLUMN_PRIO, prio)


def populate_node_menu(viz, node, menu):
    menu_item = Gtk.MenuItem("Show IPv4 Routing Table")
    menu_item.show()

    def _show_ipv4_routing_table(dummy_menu_item):
        ShowIpv4RoutingTable(viz, node.node_index)

    menu_item.connect("activate", _show_ipv4_routing_table)
    menu.add(menu_item)

def register(viz):
    viz.connect("populate-node-menu", populate_node_menu)
