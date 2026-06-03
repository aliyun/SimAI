import math
import ns.wifi
import ns.network
from gi.repository import GooCanvas
from visualizer.base import Link, transform_distance_canvas_to_simulation

## WifiLink class
class WifiLink(Link):
    ## @var node1
    #  sta
    ## @var dev
    #  dev
    ## @var node2
    #  ap
    ## @var canvas_item
    #  parent_canvas_item
    ## @var invisible_line
    #  invisible line
    ## @var visible_line
    #  visible line
    def __init__(self, parent_canvas_item, sta, dev):
        """! Initialize function.
        @param self The object pointer.
        @param parent_canvas_item: parent canvas
        @param sta The STA node
        @param dev The dev
        """
        super(WifiLink, self).__init__()
        self.node1 = sta
        self.dev = dev
        self.node2 = None # ap
        self.canvas_item = GooCanvas.CanvasGroup(parent=parent_canvas_item)
        self.invisible_line = GooCanvas.CanvasPolyline(parent=self.canvas_item,
                                                 line_width=25.0,
                                                 visibility=GooCanvas.CanvasItemVisibility.HIDDEN)
        self.visible_line = GooCanvas.CanvasPolyline(parent=self.canvas_item,
                                              line_width=1.0,
                                              stroke_color_rgba=0xC00000FF,
                                              line_dash=GooCanvas.CanvasLineDash.newv([2.0, 2.0 ]))
        # self.invisible_line.set_property("pointer-events", (GooCanvas.CanvasPointerEvents.STROKE_MASK
        #                                             |GooCanvas.CanvasPointerEvents.FILL_MASK
        #                                             |GooCanvas.CanvasPointerEvents.PAINTED_MASK))
        self.canvas_item.pyviz_object = self
        self.canvas_item.lower(None)
        self.set_ap(None)

    def set_ap(self, ap):
        """! Set AP.
        @param self The object pointer.
        @param ap The AP node
        @return none
        """
        if ap is self.node2:
            return
        if self.node2 is not None:
            self.node2.remove_link(self)
        self.node2 = ap
        if self.node2 is None:
            self.canvas_item.set_property("visibility", GooCanvas.CanvasItemVisibility.HIDDEN)
        else:
            self.node2.add_link(self)
            self.canvas_item.set_property("visibility", GooCanvas.CanvasItemVisibility.VISIBLE)
        self.update_points()

    def update_points(self):
        """! Update points function.
        @param self The object pointer.
        @return none
        """
        if self.node2 is None:
            return
        pos1_x, pos1_y = self.node1.get_position()
        pos2_x, pos2_y = self.node2.get_position()
        points = GooCanvas.CanvasPoints.new(2)
        points.set_point(0, pos1_x, pos1_y)
        points.set_point(1, pos2_x, pos2_y)
        self.visible_line.set_property("points", points)
        self.invisible_line.set_property("points", points)

    def destroy(self):
        """! Destroy function.
        @param self The object pointer.
        @return none
        """
        self.canvas_item.destroy()
        self.node1 = None
        self.node2 = None

    def tooltip_query(self, tooltip):
        """! Destroy function.
        @param self The object pointer.
        @param tooltip The tooltip.
        @return tooltip
        """
        pos1_x, pos1_y = self.node1.get_position()
        pos2_x, pos2_y = self.node2.get_position()
        dx = pos2_x - pos1_x
        dy = pos2_y - pos1_y
        d = transform_distance_canvas_to_simulation(math.sqrt(dx*dx + dy*dy))
        mac = self.dev.GetMac()
        tooltip.set_text(("WiFi link between STA Node %i and AP Node %i; distance=%.2f m.\n"
                          "SSID: %s\n"
                          "BSSID: %s")
                         % (self.node1.node_index, self.node2.node_index, d,
                            mac.GetSsid(), mac.GetBssid()))

## WifiLinkMonitor class
class WifiLinkMonitor(object):
    ## @var access_points
    #  bssid -> node
    ## @var stations
    #  list of (sta_netdevice, viz_node, wifi_link)
    def __init__(self, dummy_viz):
        """! Initialize function.
        @param self The object pointer.
        @param dummy_viz A dummy visualizer
        """
        self.access_points = {} # bssid -> node
        self.stations = [] # list of (sta_netdevice, viz_node, wifi_link)

    def scan_nodes(self, viz):
        """! Scan nodes function.
        @param self The object pointer.
        @param viz The visualizer object
        @return none
        """
        for (sta_netdevice, viz_node, wifi_link) in self.stations:
            wifi_link.destroy()

        self.access_points = {}
        self.stations = []

        for node in viz.nodes.values():
            ns3_node = ns.network.NodeList.GetNode(node.node_index)
            for devI in range(ns3_node.GetNDevices()):
                dev = ns3_node.GetDevice(devI)
                if not isinstance(dev, ns.wifi.WifiNetDevice):
                    continue
                wifi_mac = dev.GetMac()
                if isinstance(wifi_mac, ns.wifi.StaWifiMac):
                    wifi_link = WifiLink(viz.links_group, node, dev)
                    self.stations.append((dev, node, wifi_link))
                elif isinstance(wifi_mac, ns.wifi.ApWifiMac):
                    bssid = ns.network.Mac48Address.ConvertFrom(dev.GetAddress())
                    self.access_points[str(bssid)] = node
        #print "APs: ", self.access_points
        #print "STAs: ", self.stations

    def simulation_periodic_update(self, viz):
        """! Simulation Periodic Update function.
        @param self The object pointer.
        @param viz The visualizer object
        @return none
        """
        for (sta_netdevice, viz_node, wifi_link) in self.stations:
            if not sta_netdevice.IsLinkUp():
                wifi_link.set_ap(None)
                continue
            bssid = str(sta_netdevice.GetMac().GetBssid())
            if bssid == '00:00:00:00:00:00':
                wifi_link.set_ap(None)
                continue
            ap = self.access_points[bssid]
            wifi_link.set_ap(ap)

    def update_view(self, viz):
        """! Update View function.
        @param self The object pointer.
        @param viz The visualizer object
        @return none
        """
        for (dummy_sta_netdevice, dummy_viz_node, wifi_link) in self.stations:
            if wifi_link is not None:
                wifi_link.update_points()


def register(viz):
    link_monitor = WifiLinkMonitor(viz)
    viz.connect("simulation-periodic-update", link_monitor.simulation_periodic_update)
    viz.connect("update-view", link_monitor.update_view)
    viz.connect("topology-scanned", link_monitor.scan_nodes)
