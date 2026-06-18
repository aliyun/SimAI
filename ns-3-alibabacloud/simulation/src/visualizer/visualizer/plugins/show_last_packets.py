from gi.repository import GObject
from gi.repository import Gtk

import ns.core
import ns.network
import ns.visualizer

from visualizer.base import InformationWindow
from kiwi.ui.objectlist import ObjectList, Column

## ShowLastPackets class
class ShowLastPackets(InformationWindow):
    ## @var win
    #  window
    ## @var visualizer
    #  visualizer
    ## @var viz_node
    #  visualizer node
    ## @var node
    #  the node
    ## @var tx_list
    #  packet transmit list
    ## @var rx_list
    #  packet receive list
    ## @var drop_list
    #  packet drop list
    ## @var packet_capture_options
    #  packet capture options
    ## @var packet_filter_widget
    #  packet filter widget
    ## @var packet_filter_list
    #  list of TypeIdConfig instances
    ## @var op_AND_button
    #  AND button
    ## @var op_OR_button
    #  OR button
    class PacketList(Gtk.ScrolledWindow):
        """
        PacketList class
        """
        ## @var table_model
        #  table model
        (
            COLUMN_TIME,
            COLUMN_INTERFACE,
            COLUMN_SIZE,
            COLUMN_CONTENTS,
            ) = range(4)

        def __init__(self):
            """
            Initializer
            @param self this object
            """
            super(ShowLastPackets.PacketList, self).__init__()
            self.set_properties(hscrollbar_policy=Gtk.PolicyType.AUTOMATIC,
                                vscrollbar_policy=Gtk.PolicyType.AUTOMATIC)
            self.table_model = Gtk.ListStore(*([str]*4))
            treeview = Gtk.TreeView(self.table_model)
            treeview.show()
            self.add(treeview)

            def add_column(descr, colid):
                column = Gtk.TreeViewColumn(descr, Gtk.CellRendererText(), text=colid)
                treeview.append_column(column)

            add_column("Time", self.COLUMN_TIME)
            add_column("Interface", self.COLUMN_INTERFACE)
            add_column("Size", self.COLUMN_SIZE)
            add_column("Contents", self.COLUMN_CONTENTS)

        def update(self, node, packet_list):
            """!
            Update function
            @param self this object
            @param node the node
            @param packet_list packet list
            @return none
            """
            self.table_model.clear()
            for sample in packet_list:
                tree_iter = self.table_model.append()
                if sample.device is None:
                    interface_name = "(unknown)"
                else:
                    interface_name = ns.core.Names.FindName(sample.device)
                    if not interface_name:
                        interface_name = "(interface %i)" % sample.device.GetIfIndex()
                self.table_model.set(tree_iter,
                                     self.COLUMN_TIME, str(sample.time.GetSeconds()),
                                     self.COLUMN_INTERFACE, interface_name,
                                     self.COLUMN_SIZE, str(sample.packet.GetSize ()),
                                     self.COLUMN_CONTENTS, str(sample.packet)
                                     )


    def __init__(self, visualizer, node_index):
        """!
        Initializer
        @param self this object
        @param visualizer the visualizer object
        @param node_index the node index
        """
        InformationWindow.__init__(self)
        self.win = Gtk.Dialog(parent=visualizer.window,
                              flags=Gtk.DialogFlags.DESTROY_WITH_PARENT|Gtk.DialogFlags.NO_SEPARATOR,
                              buttons=(Gtk.STOCK_CLOSE, Gtk.ResponseType.CLOSE))
        self.win.connect("response", self._response_cb)
        self.win.set_title("Last packets for node %i" % node_index)
        self.visualizer = visualizer
        self.viz_node = visualizer.get_node(node_index)
        self.node = ns.network.NodeList.GetNode(node_index)

        def smart_expand(expander, vbox):
            if expander.get_expanded():
                vbox.set_child_packing(expander, expand=True, fill=True, padding=0, pack_type=Gtk.PACK_START)
            else:
                vbox.set_child_packing(expander, expand=False, fill=False, padding=0, pack_type=Gtk.PACK_START)

        main_hbox = Gtk.HBox(False, 4)
        main_hbox.show()
        main_vbox = Gtk.VBox(False, 4)
        main_vbox.show()
        self.win.vbox.add(main_hbox)
        main_hbox.add(main_vbox)

        self.tx_list = self.PacketList()
        self.tx_list.show()
        group = Gtk.Expander("Last transmitted packets")
        group.show()
        group.add(self.tx_list)
        main_vbox.pack_start(group, expand=False, fill=False)
        group.connect_after("activate", smart_expand, main_vbox)

        self.rx_list = self.PacketList()
        self.rx_list.show()
        group = Gtk.Expander("Last received packets")
        group.show()
        group.add(self.rx_list)
        main_vbox.pack_start(group, expand=False, fill=False)
        group.connect_after("activate", smart_expand, main_vbox)

        self.drop_list = self.PacketList()
        self.drop_list.show()
        group = Gtk.Expander("Last dropped packets")
        group.show()
        group.add(self.drop_list)
        main_vbox.pack_start(group, expand=False, fill=False)
        group.connect_after("activate", smart_expand, main_vbox)


        # Packet Filter

        # - options
        self.packet_capture_options = ns.visualizer.PyViz.PacketCaptureOptions()
        self.packet_capture_options.numLastPackets = 100

        packet_filter_vbox = Gtk.VBox(False, 4)
        packet_filter_vbox.show()
        main_hbox.add(packet_filter_vbox)

        sel_buttons_box = Gtk.HButtonBox()
        sel_buttons_box.show()
        packet_filter_vbox.pack_start(sel_buttons_box, False, False, 4)
        select_all_button = GObject.new(Gtk.Button, label="Sel. All", visible=True)
        select_none_button = GObject.new(Gtk.Button, label="Sel. None", visible=True)
        sel_buttons_box.add(select_all_button)
        sel_buttons_box.add(select_none_button)

        self.packet_filter_widget = ObjectList([
                Column('selected', title="Sel.", data_type=bool, editable=True),
                Column('name', title="Header"),
                ], sortable=True)
        self.packet_filter_widget.show()
        packet_filter_vbox.pack_start(self.packet_filter_widget, True, True, 4)

        class TypeIdConfig(object):
            __slots__ = ['name', 'selected', 'typeid']

        self.packet_filter_list = [] # list of TypeIdConfig instances

        Header = ns.core.TypeId.LookupByName("ns3::Header")
        Trailer = ns.core.TypeId.LookupByName("ns3::Trailer")
        for typeid_i in range(ns.core.TypeId.GetRegisteredN()):
            typeid = ns.core.TypeId.GetRegistered(typeid_i)
            # check if this is a header or trailer subtype
            typeid_tmp = typeid
            type_is_good = False
            while 1:
                if typeid_tmp == Header or typeid_tmp == Trailer:
                    type_is_good = True
                    break
                if typeid_tmp.HasParent():
                    typeid_tmp = typeid_tmp.GetParent()
                else:
                    break
            if not type_is_good:
                continue
            if typeid in [Header, Trailer]:
                continue
            c = TypeIdConfig()
            c.selected = True
            c.name = typeid.GetName()
            c.typeid = typeid
            self.packet_filter_list.append(c)
        self.packet_filter_widget.add_list(self.packet_filter_list)

        def update_capture_options():
            if self.op_AND_button.props.active:
                self.packet_capture_options.mode = ns.visualizer.PyViz.PACKET_CAPTURE_FILTER_HEADERS_AND
            else:
                self.packet_capture_options.mode = ns.visualizer.PyViz.PACKET_CAPTURE_FILTER_HEADERS_OR
            self.packet_capture_options.numLastPackets = 100
            self.packet_capture_options.headers = [c.typeid for c in self.packet_filter_list if c.selected]
            self.visualizer.simulation.lock.acquire()
            try:
                self.visualizer.simulation.sim_helper.SetPacketCaptureOptions(
                    self.node.GetId(), self.packet_capture_options)
            finally:
                self.visualizer.simulation.lock.release()

        def sel_all_cb(bt):
            for c in self.packet_filter_list:
                c.selected = True
            self.packet_filter_widget.refresh()
            update_capture_options()

        def sel_none_cb(bt):
            for c in self.packet_filter_list:
                c.selected = False
            self.packet_filter_widget.refresh()
            update_capture_options()

        select_all_button.connect("clicked", sel_all_cb)
        select_none_button.connect("clicked", sel_none_cb)

        op_buttons_box = Gtk.HButtonBox()
        op_buttons_box.show()
        packet_filter_vbox.pack_start(op_buttons_box, False, False, 4)
        self.op_AND_button = GObject.new(Gtk.RadioButton, label="AND", visible=True)
        self.op_OR_button = GObject.new(Gtk.RadioButton, label="OR", visible=True, group=self.op_AND_button)
        op_buttons_box.add(self.op_AND_button)
        op_buttons_box.add(self.op_OR_button)
        self.op_OR_button.props.active = True

        self.op_AND_button.connect("toggled", lambda b: update_capture_options())

        def cell_edited(l, obj, attribute):
            update_capture_options()
        self.packet_filter_widget.connect("cell-edited", cell_edited)

        update_capture_options()

        self.visualizer.add_information_window(self)
        self.win.set_default_size(600, 300)
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
        last_packets = self.visualizer.simulation.sim_helper.GetLastPackets(self.node.GetId())

        self.tx_list.update(self.node, last_packets.lastTransmittedPackets)
        self.rx_list.update(self.node, last_packets.lastReceivedPackets)
        self.drop_list.update(self.node, last_packets.lastDroppedPackets)


def populate_node_menu(viz, node, menu):
    menu_item = Gtk.MenuItem("Show Last Packets")
    menu_item.show()

    def _show_it(dummy_menu_item):
        ShowLastPackets(viz, node.node_index)

    menu_item.connect("activate", _show_it)
    menu.add(menu_item)

def register(viz):
    viz.connect("populate-node-menu", populate_node_menu)
