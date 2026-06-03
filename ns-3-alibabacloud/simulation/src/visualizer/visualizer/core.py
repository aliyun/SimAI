# -*- Mode: python; coding: utf-8 -*-
from __future__ import division, print_function
#from __future__ import with_statement

LAYOUT_ALGORITHM = 'neato' # ['neato'|'dot'|'twopi'|'circo'|'fdp'|'nop']
REPRESENT_CHANNELS_AS_NODES = 1
DEFAULT_NODE_SIZE = 1.0 # default node size in meters
DEFAULT_TRANSMISSIONS_MEMORY = 5 # default number of of past intervals whose transmissions are remembered
BITRATE_FONT_SIZE = 10

# internal constants, normally not meant to be changed
SAMPLE_PERIOD = 0.1
PRIORITY_UPDATE_MODEL = -100
PRIORITY_UPDATE_VIEW = 200

import warnings
import platform
if platform.system() == "Windows":
    SHELL_FONT = "Lucida Console 9"
else:
    SHELL_FONT = "Luxi Mono 10"


import ns.core
import ns.network
import ns.visualizer
import ns.internet
import ns.mobility

import math
import os
import sys

if sys.version_info > (3,):
    long = int

try:
    import gi
    gi.require_version('GooCanvas', '2.0')
    gi.require_version('Gtk', '3.0')
    gi.require_version('Gdk', '3.0')
    from gi.repository import GObject
    from gi.repository import GLib
    import cairo
    gi.require_foreign("cairo")
    import pygraphviz
    from gi.repository import Gtk
    from gi.repository import Gdk
    from gi.repository import Pango
    from gi.repository import GooCanvas
    import threading
    from . import hud
    #import time
    try:
        import svgitem
    except ImportError:
        svgitem = None
except ImportError as e:
    _import_error = e
    import dummy_threading as threading
else:
    _import_error = None

try:
    import ipython_viewxxxxxxxxxx
except ImportError:
    ipython_view = None

from .base import InformationWindow, PyVizObject, Link, lookup_netdevice_traits, PIXELS_PER_METER
from .base import transform_distance_simulation_to_canvas, transform_point_simulation_to_canvas
from .base import transform_distance_canvas_to_simulation, transform_point_canvas_to_simulation
from .base import load_plugins, register_plugin, plugins

PI_OVER_2 = math.pi/2
PI_TIMES_2 = math.pi*2

## Node class
class Node(PyVizObject):
    ## @var visualizer
    #  visualier object
    ## @var node_index
    #  node index
    ## @var canvas_item
    #  canvas item
    ## @var links
    #  links
    ## @var _has_mobility
    #  has mobility model
    ## @var _selected
    #  is selected
    ## @var _highlighted
    #  is highlighted
    ## @var _color
    #  color
    ## @var _size
    #  size
    ## @var menu
    #  menu
    ## @var svg_item
    #  svg item
    ## @var svg_align_x
    #  svg align X
    ## @var svg_align_y
    #  svg align Y
    ## @var _label
    #  label
    ## @var _label_canvas_item
    #  label canvas
    ## @var selected
    #  selected property
    ## @var highlighted
    #  highlighted property

    ## signal emitted whenever a tooltip is about to be shown for the node
    ## the first signal parameter is a python list of strings, to which 
    ## information can be appended
    __gsignals__ = {
        'query-extra-tooltip-info': (GObject.SignalFlags.RUN_LAST, None, (object,)),
        }

    def __init__(self, visualizer, node_index):
        """! Initialize function.
        @param self The object pointer.
        @param visualizer visualizer object
        @param node_index node index
        """
        super(Node, self).__init__()

        self.visualizer = visualizer
        self.node_index = node_index
        self.canvas_item = GooCanvas.CanvasEllipse()
        self.canvas_item.pyviz_object = self
        self.links = []
        self._has_mobility = None
        self._selected = False
        self._highlighted = False
        self._color = 0x808080ff
        self._size = DEFAULT_NODE_SIZE
        self.canvas_item.connect("enter-notify-event", self.on_enter_notify_event)
        self.canvas_item.connect("leave-notify-event", self.on_leave_notify_event)
        self.menu = None
        self.svg_item = None
        self.svg_align_x = None
        self.svg_align_y = None
        self._label = None
        self._label_canvas_item = None

        self._update_appearance() # call this last

    def set_svg_icon(self, file_base_name, width=None, height=None, align_x=0.5, align_y=0.5):
        """!
        Set a background SVG icon for the node.

        @param file_base_name: base file name, including .svg
        extension, of the svg file.  Place the file in the folder
        src/contrib/visualizer/resource.

        @param width: scale to the specified width, in meters
        @param height: scale to the specified height, in meters

        @param align_x: horizontal alignment of the icon relative to
        the node position, from 0 (icon fully to the left of the node)
        to 1.0 (icon fully to the right of the node)

        @param align_y: vertical alignment of the icon relative to the
        node position, from 0 (icon fully to the top of the node) to
        1.0 (icon fully to the bottom of the node)

        @return a ValueError exception if invalid dimensions.

        """
        if width is None and height is None:
            raise ValueError("either width or height must be given")
        rsvg_handle = svgitem.rsvg_handle_factory(file_base_name)
        x = self.canvas_item.props.center_x
        y = self.canvas_item.props.center_y
        self.svg_item = svgitem.SvgItem(x, y, rsvg_handle)
        self.svg_item.props.parent = self.visualizer.canvas.get_root_item()
        self.svg_item.props.pointer_events = GooCanvas.CanvasPointerEvents.NONE
        self.svg_item.lower(None)
        self.svg_item.props.visibility = GooCanvas.CanvasItemVisibility.VISIBLE_ABOVE_THRESHOLD
        if width is not None:
            self.svg_item.props.width = transform_distance_simulation_to_canvas(width)
        if height is not None:
            self.svg_item.props.height = transform_distance_simulation_to_canvas(height)

        #threshold1 = 10.0/self.svg_item.props.height
        #threshold2 = 10.0/self.svg_item.props.width
        #self.svg_item.props.visibility_threshold = min(threshold1, threshold2)

        self.svg_align_x = align_x
        self.svg_align_y = align_y
        self._update_svg_position(x, y)
        self._update_appearance()

    def set_label(self, label):
        """!
        Set a label for the node.

        @param self: class object.
        @param label: label to set

        @return: an exception if invalid parameter.
        """
        assert isinstance(label, basestring)
        self._label = label
        self._update_appearance()

    def _update_svg_position(self, x, y):
        """!
        Update svg position.

        @param self: class object.
        @param x: x position
        @param y: y position
        @return none
        """
        w = self.svg_item.width
        h = self.svg_item.height
        self.svg_item.set_properties(x=(x - (1-self.svg_align_x)*w),
                                     y=(y - (1-self.svg_align_y)*h))


    def tooltip_query(self, tooltip):
        """!
        Query tooltip.

        @param self: class object.
        @param tooltip: tooltip
        @return none
        """
        self.visualizer.simulation.lock.acquire()
        try:
            ns3_node = ns.network.NodeList.GetNode(self.node_index)
            ipv4 = ns3_node.GetObject(ns.internet.Ipv4.GetTypeId())
            ipv6 = ns3_node.GetObject(ns.internet.Ipv6.GetTypeId())

            name = '<b><u>Node %i</u></b>' % self.node_index
            node_name = ns.core.Names.FindName (ns3_node)
            if len(node_name)!=0:
                name += ' <b>(' + node_name + ')</b>'

            lines = [name]
            lines.append('')

            self.emit("query-extra-tooltip-info", lines)

            mob = ns3_node.GetObject(ns.mobility.MobilityModel.GetTypeId())
            if mob is not None:
                lines.append('  <b>Mobility Model</b>: %s' % mob.GetInstanceTypeId().GetName())

            for devI in range(ns3_node.GetNDevices()):
                lines.append('')
                lines.append('  <u>NetDevice %i:</u>' % devI)
                dev = ns3_node.GetDevice(devI)
                name = ns.core.Names.FindName(dev)
                if name:
                    lines.append('    <b>Name:</b> %s' % name)
                devname = dev.GetInstanceTypeId().GetName()
                lines.append('    <b>Type:</b> %s' % devname)

                if ipv4 is not None:
                    ipv4_idx = ipv4.GetInterfaceForDevice(dev)
                    if ipv4_idx != -1:
                        addresses = [
                            '%s/%s' % (ipv4.GetAddress(ipv4_idx, i).GetLocal(),
                                       ipv4.GetAddress(ipv4_idx, i).GetMask())
                            for i in range(ipv4.GetNAddresses(ipv4_idx))]
                        lines.append('    <b>IPv4 Addresses:</b> %s' % '; '.join(addresses))

                if ipv6 is not None:
                    ipv6_idx = ipv6.GetInterfaceForDevice(dev)
                    if ipv6_idx != -1:
                        addresses = [
                            '%s/%s' % (ipv6.GetAddress(ipv6_idx, i).GetAddress(),
                                       ipv6.GetAddress(ipv6_idx, i).GetPrefix())
                            for i in range(ipv6.GetNAddresses(ipv6_idx))]
                        lines.append('    <b>IPv6 Addresses:</b> %s' % '; '.join(addresses))

                lines.append('    <b>MAC Address:</b> %s' % (dev.GetAddress(),))

            tooltip.set_markup('\n'.join(lines))
        finally:
            self.visualizer.simulation.lock.release()

    def on_enter_notify_event(self, view, target, event):
        """!
        On Enter event handle.

        @param self: class object.
        @param view: view
        @param target: target
        @param event: event
        @return none
        """
        self.highlighted = True
    def on_leave_notify_event(self, view, target, event):
        """!
        On Leave event handle.

        @param self: class object.
        @param view: view
        @param target: target
        @param event: event
        @return none
        """
        self.highlighted = False

    def _set_selected(self, value):
        """!
        Set selected function.

        @param self: class object.
        @param value: selected value
        @return none
        """
        self._selected = value
        self._update_appearance()
    def _get_selected(self):
        """!
        Get selected function.

        @param self: class object.
        @return selected status
        """
        return self._selected

    selected = property(_get_selected, _set_selected)

    def _set_highlighted(self, value):
        """!
        Set highlighted function.

        @param self: class object.
        @param value: selected value
        @return none
        """
        self._highlighted = value
        self._update_appearance()
    def _get_highlighted(self):
        """!
        Get highlighted function.

        @param self: class object.
        @return highlighted status
        """
        return self._highlighted

    highlighted = property(_get_highlighted, _set_highlighted)

    def set_size(self, size):
        """!
        Set size function.

        @param self: class object.
        @param size: selected size
        @return none
        """
        self._size = size
        self._update_appearance()

    def _update_appearance(self):
        """!
        Update the node aspect to reflect the selected/highlighted state

        @param self: class object.
        @return none
        """

        size = transform_distance_simulation_to_canvas(self._size)
        if self.svg_item is not None:
            alpha = 0x80
        else:
            alpha = 0xff
        fill_color_rgba = (self._color & 0xffffff00) | alpha
        self.canvas_item.set_properties(radius_x=size, radius_y=size,
                                        fill_color_rgba=fill_color_rgba)
        if self._selected:
            line_width = size*.3
        else:
            line_width = size*.15
        if self.highlighted:
            stroke_color = 'yellow'
        else:
            stroke_color = 'black'
        self.canvas_item.set_properties(line_width=line_width, stroke_color=stroke_color)

        if self._label is not None:
            if self._label_canvas_item is None:
                self._label_canvas_item = GooCanvas.CanvasText(visibility_threshold=0.5,
                                                         font="Sans Serif 10",
                                                         fill_color_rgba=0x808080ff,
                                                         alignment=Pango.Alignment.CENTER,
                                                         anchor=GooCanvas.CanvasAnchorType.N,
                                                         parent=self.visualizer.canvas.get_root_item(),
                                                         pointer_events=GooCanvas.CanvasPointerEvents.NONE)
                self._label_canvas_item.lower(None)

            self._label_canvas_item.set_properties(visibility=GooCanvas.CanvasItemVisibility.VISIBLE_ABOVE_THRESHOLD,
                                                   text=self._label)
            self._update_position()

    def set_position(self, x, y):
        """!
        Set position function.

        @param self: class object.
        @param x: x position
        @param y: y position
        @return none
        """
        self.canvas_item.set_property("center_x", x)
        self.canvas_item.set_property("center_y", y)
        if self.svg_item is not None:
            self._update_svg_position(x, y)

        for link in self.links:
            link.update_points()

        if self._label_canvas_item is not None:
            self._label_canvas_item.set_properties(x=x, y=(y+self._size*3))

        # If the location of the point is now beyond the bounds of the
        # canvas then those bounds now need to be increased
        try:
            bounds = self.visualizer.canvas.get_bounds()

            (min_x, min_y, max_x, max_y) = bounds

            min_x = min(x, min_x)
            min_y = min(y, min_y)
            max_x = max(x, max_x)
            max_y = max(y, max_y)

            new_bounds = (min_x, min_y, max_x, max_y)

            if new_bounds != bounds:
                self.visualizer.canvas.set_bounds(*new_bounds)
        except TypeError:
            # bug 2969:  GooCanvas.Canvas.get_bounds() inconsistency
            pass

    def get_position(self):
        """!
        Get position function.

        @param self: class object.
        @return x and y position
        """
        return (self.canvas_item.get_property("center_x"), self.canvas_item.get_property("center_y"))

    def _update_position(self):
        """!
        Update position function.

        @param self: class object.
        @return none
        """
        x, y = self.get_position()
        self.set_position(x, y)

    def set_color(self, color):
        """!
        Set color function.

        @param self: class object.
        @param color: color to set.
        @return none
        """
        if isinstance(color, str):
            color = Gdk.color_parse(color)
            color = ((color.red>>8) << 24) | ((color.green>>8) << 16) | ((color.blue>>8) << 8) | 0xff
        self._color = color
        self._update_appearance()

    def add_link(self, link):
        """!
        Add link function.

        @param self: class object.
        @param link: link to add.
        @return none
        """
        assert isinstance(link, Link)
        self.links.append(link)

    def remove_link(self, link):
        """!
        Remove link function.

        @param self: class object.
        @param link: link to add.
        @return none
        """
        assert isinstance(link, Link)
        self.links.remove(link)

    @property
    def has_mobility(self):
        """!
        Has mobility function.

        @param self: class object.
        @return modility option
        """
        if self._has_mobility is None:
            node = ns.network.NodeList.GetNode(self.node_index)
            mobility = node.GetObject(ns.mobility.MobilityModel.GetTypeId())
            self._has_mobility = (mobility is not None)
        return self._has_mobility


## Channel
class Channel(PyVizObject):
    ## @var channel
    #  channel
    ## @var canvas_item
    #  canvas
    ## @var links
    #  list of links
    #
    def __init__(self, channel):
        """!
        Initializer function.

        @param self: class object.
        @param channel: channel.
        """
        self.channel = channel
        self.canvas_item = GooCanvas.CanvasEllipse(radius_x=30, radius_y=30,
                                             fill_color="white",
                                             stroke_color="grey", line_width=2.0,
                                             line_dash=GooCanvas.CanvasLineDash.newv([10.0, 10.0 ]),
                                             visibility=GooCanvas.CanvasItemVisibility.VISIBLE)
        self.canvas_item.pyviz_object = self
        self.links = []

    def set_position(self, x, y):
        """!
        Initializer function.

        @param self: class object.
        @param x: x position.
        @param y: y position.
        @return
        """
        self.canvas_item.set_property("center_x", x)
        self.canvas_item.set_property("center_y", y)

        for link in self.links:
            link.update_points()

    def get_position(self):
        """!
        Initializer function.

        @param self: class object.
        @return x / y position.
        """
        return (self.canvas_item.get_property("center_x"), self.canvas_item.get_property("center_y"))


## WiredLink
class WiredLink(Link):
    ## @var node1
    #  first node
    ## @var node2
    #  second node
    ## @var canvas_item
    #  canvas
    #
    def __init__(self, node1, node2):
        """!
        Initializer function.

        @param self: class object.
        @param node1: class object.
        @param node2: class object.
        """
        assert isinstance(node1, Node)
        assert isinstance(node2, (Node, Channel))
        self.node1 = node1
        self.node2 = node2
        self.canvas_item = GooCanvas.CanvasPath(line_width=1.0, stroke_color="black")
        self.canvas_item.pyviz_object = self
        self.node1.links.append(self)
        self.node2.links.append(self)

    def update_points(self):
        """!
        Update points function.

        @param self: class object.
        @return none
        """
        pos1_x, pos1_y = self.node1.get_position()
        pos2_x, pos2_y = self.node2.get_position()
        self.canvas_item.set_property("data", "M %r %r L %r %r" % (pos1_x, pos1_y, pos2_x, pos2_y))


## SimulationThread
class SimulationThread(threading.Thread):
    ## @var viz
    #  Visualizer object
    ## @var lock
    #  thread lock
    ## @var go
    #  thread event
    ## @var target_time
    #  in seconds
    ## @var quit
    #  quit indicator
    ## @var sim_helper
    #  helper function
    ## @var pause_messages
    #  pause messages
    def __init__(self, viz):
        """!
        Initializer function.

        @param self: class object.
        @param viz: class object.
        """
        super(SimulationThread, self).__init__()
        assert isinstance(viz, Visualizer)
        self.viz = viz # Visualizer object
        self.lock = threading.Lock()
        self.go = threading.Event()
        self.go.clear()
        self.target_time = 0 # in seconds
        self.quit = False
        self.sim_helper = ns.visualizer.PyViz()
        self.pause_messages = []

    def set_nodes_of_interest(self, nodes):
        """!
        Set nodes of interest function.

        @param self: class object.
        @param nodes: class object.
        @return
        """
        self.lock.acquire()
        try:
            self.sim_helper.SetNodesOfInterest(nodes)
        finally:
            self.lock.release()

    def run(self):
        """!
        Initializer function.

        @param self: class object.
        @return none
        """
        while not self.quit:
            #print "sim: Wait for go"
            self.go.wait() # wait until the main (view) thread gives us the go signal
            self.go.clear()
            if self.quit:
                break
            #self.go.clear()
            #print "sim: Acquire lock"
            self.lock.acquire()
            try:
                if 0:
                    if ns3.core.Simulator.IsFinished():
                        self.viz.play_button.set_sensitive(False)
                        break
                #print "sim: Current time is %f; Run until: %f" % (ns3.Simulator.Now ().GetSeconds (), self.target_time)
                #if ns3.Simulator.Now ().GetSeconds () > self.target_time:
                #    print "skipping, model is ahead of view!"
                self.sim_helper.SimulatorRunUntil(ns.core.Seconds(self.target_time))
                #print "sim: Run until ended at current time: ", ns3.Simulator.Now ().GetSeconds ()
                self.pause_messages.extend(self.sim_helper.GetPauseMessages())
                GLib.idle_add(self.viz.update_model, priority=PRIORITY_UPDATE_MODEL)
                #print "sim: Run until: ", self.target_time, ": finished."
            finally:
                self.lock.release()
            #print "sim: Release lock, loop."

## ShowTransmissionsMode
class ShowTransmissionsMode(object):
    ## @var ALL
    #  all
    ## @var NONE
    #  none
    ## @var SELECTED
    #  selected

    ## enumeration
    __slots__ = []
ShowTransmissionsMode.ALL = ShowTransmissionsMode()
ShowTransmissionsMode.NONE = ShowTransmissionsMode()
ShowTransmissionsMode.SELECTED = ShowTransmissionsMode()

## Visualizer
class Visualizer(GObject.GObject):
    ## @var INSTANCE
    #  all
    INSTANCE = None

    if _import_error is None:
        __gsignals__ = {

            # signal emitted whenever a right-click-on-node popup menu is being constructed
            'populate-node-menu': (GObject.SignalFlags.RUN_LAST, None, (object, Gtk.Menu,)),

            # signal emitted after every simulation period (SAMPLE_PERIOD seconds of simulated time)
            # the simulation lock is acquired while the signal is emitted
            'simulation-periodic-update': (GObject.SignalFlags.RUN_LAST, None, ()),

            # signal emitted right after the topology is scanned
            'topology-scanned': (GObject.SignalFlags.RUN_LAST, None, ()),

            # signal emitted when it's time to update the view objects
            'update-view': (GObject.SignalFlags.RUN_LAST, None, ()),

            }

    def __init__(self):
        """!
        Initializer function.

        @param self: class object.
        @return none
        """
        assert Visualizer.INSTANCE is None
        Visualizer.INSTANCE = self
        super(Visualizer, self).__init__()
        self.nodes =  {} # node index -> Node
        self.channels = {} # id(ns3.Channel) -> Channel
        self.window = None # toplevel window
        self.canvas = None # GooCanvas.Canvas
        self.time_label = None # Gtk.Label
        self.play_button = None # Gtk.ToggleButton
        self.zoom = None # Gtk.Adjustment
        self._scrolled_window = None # Gtk.ScrolledWindow

        self.links_group = GooCanvas.CanvasGroup()
        self.channels_group = GooCanvas.CanvasGroup()
        self.nodes_group = GooCanvas.CanvasGroup()

        self._update_timeout_id = None
        self.simulation = SimulationThread(self)
        self.selected_node = None # node currently selected
        self.speed = 1.0
        self.information_windows = []
        self._transmission_arrows = []
        self._last_transmissions = []
        self._drop_arrows = []
        self._last_drops = []
        self._show_transmissions_mode = None
        self.set_show_transmissions_mode(ShowTransmissionsMode.ALL)
        self._panning_state = None
        self.node_size_adjustment = None
        self.transmissions_smoothing_adjustment = None
        self.sample_period = SAMPLE_PERIOD
        self.node_drag_state = None
        self.follow_node = None
        self.shell_window = None

        self.create_gui()

        for plugin in plugins:
            plugin(self)

    def set_show_transmissions_mode(self, mode):
        """!
        Set show transmission mode.

        @param self: class object.
        @param mode: mode to set.
        @return none
        """
        assert isinstance(mode, ShowTransmissionsMode)
        self._show_transmissions_mode = mode
        if self._show_transmissions_mode == ShowTransmissionsMode.ALL:
            self.simulation.set_nodes_of_interest(list(range(ns.network.NodeList.GetNNodes())))
        elif self._show_transmissions_mode == ShowTransmissionsMode.NONE:
            self.simulation.set_nodes_of_interest([])
        elif self._show_transmissions_mode == ShowTransmissionsMode.SELECTED:
            if self.selected_node is None:
                self.simulation.set_nodes_of_interest([])
            else:
                self.simulation.set_nodes_of_interest([self.selected_node.node_index])

    def _create_advanced_controls(self):
        """!
        Create advanced controls.

        @param self: class object.
        @return expander
        """
        expander = Gtk.Expander.new("Advanced")
        expander.show()

        main_vbox = GObject.new(Gtk.VBox, border_width=8, visible=True)
        expander.add(main_vbox)

        main_hbox1 = GObject.new(Gtk.HBox, border_width=8, visible=True)
        main_vbox.pack_start(main_hbox1, True, True, 0)

        show_transmissions_group = GObject.new(Gtk.HeaderBar,
                                               title="Show transmissions",
                                               visible=True)
        main_hbox1.pack_start(show_transmissions_group, False, False, 8)

        vbox = Gtk.VBox(homogeneous=True, spacing=4)
        vbox.show()
        show_transmissions_group.add(vbox)

        all_nodes = Gtk.RadioButton.new(None)
        all_nodes.set_label("All nodes")
        all_nodes.set_active(True)
        all_nodes.show()
        vbox.add(all_nodes)

        selected_node = Gtk.RadioButton.new_from_widget(all_nodes)
        selected_node.show()
        selected_node.set_label("Selected node")
        selected_node.set_active(False)
        vbox.add(selected_node)

        no_node = Gtk.RadioButton.new_from_widget(all_nodes)
        no_node.show()
        no_node.set_label("Disabled")
        no_node.set_active(False)
        vbox.add(no_node)

        def toggled(radio):
            if radio.get_active():
                self.set_show_transmissions_mode(ShowTransmissionsMode.ALL)
        all_nodes.connect("toggled", toggled)

        def toggled(radio):
            if radio.get_active():
                self.set_show_transmissions_mode(ShowTransmissionsMode.NONE)
        no_node.connect("toggled", toggled)

        def toggled(radio):
            if radio.get_active():
                self.set_show_transmissions_mode(ShowTransmissionsMode.SELECTED)
        selected_node.connect("toggled", toggled)

        # -- misc settings
        misc_settings_group = GObject.new(Gtk.HeaderBar, title="Misc Settings", visible=True)
        main_hbox1.pack_start(misc_settings_group, False, False, 8)
        settings_hbox = GObject.new(Gtk.HBox, border_width=8, visible=True)
        misc_settings_group.add(settings_hbox)

        # --> node size
        vbox = GObject.new(Gtk.VBox, border_width=0, visible=True)
        scale = GObject.new(Gtk.HScale, visible=True, digits=2)
        vbox.pack_start(scale, True, True, 0)
        vbox.pack_start(GObject.new(Gtk.Label, label="Node Size", visible=True), True, True, 0)
        settings_hbox.pack_start(vbox, False, False, 6)
        self.node_size_adjustment = scale.get_adjustment()
        def node_size_changed(adj):
            for node in self.nodes.values():
                node.set_size(adj.get_value())
        self.node_size_adjustment.connect("value-changed", node_size_changed)
        self.node_size_adjustment.set_lower(0.01)
        self.node_size_adjustment.set_upper(20)
        self.node_size_adjustment.set_step_increment(0.1)
        self.node_size_adjustment.set_value(DEFAULT_NODE_SIZE)

        # --> transmissions smooth factor
        vbox = GObject.new(Gtk.VBox, border_width=0, visible=True)
        scale = GObject.new(Gtk.HScale, visible=True, digits=1)
        vbox.pack_start(scale, True, True, 0)
        vbox.pack_start(GObject.new(Gtk.Label, label="Tx. Smooth Factor (s)", visible=True), True, True, 0)
        settings_hbox.pack_start(vbox, False, False, 6)
        self.transmissions_smoothing_adjustment = scale.get_adjustment()
        adj = self.transmissions_smoothing_adjustment
        adj.set_lower(0.1)
        adj.set_upper(10)
        adj.set_step_increment(0.1)
        adj.set_value(DEFAULT_TRANSMISSIONS_MEMORY*0.1)

        return expander

    ## PanningState class
    class _PanningState(object):
        ## @var __slots__
        #  internal variables
        __slots__ = ['initial_mouse_pos', 'initial_canvas_pos', 'motion_signal']

    def _begin_panning(self, widget, event):
        """!
        Set show trnamission mode.

        @param self: class object.
        @param mode: mode to set.
        @return none
        """
        display = self.canvas.get_window().get_display()
        cursor = Gdk.Cursor.new_for_display(display, Gdk.CursorType.FLEUR)
        self.canvas.get_window().set_cursor(cursor)
        self._panning_state = self._PanningState()
        pos = widget.get_window().get_device_position(event.device)
        self._panning_state.initial_mouse_pos = (pos.x, pos.y)
        x = self._scrolled_window.get_hadjustment().get_value()
        y = self._scrolled_window.get_vadjustment().get_value()
        self._panning_state.initial_canvas_pos = (x, y)
        self._panning_state.motion_signal = self.canvas.connect("motion-notify-event", self._panning_motion)

    def _end_panning(self, event):
        """!
        End panning function.

        @param self: class object.
        @param event: active event.
        @return none
        """
        if self._panning_state is None:
            return
        self.canvas.get_window().set_cursor(None)
        self.canvas.disconnect(self._panning_state.motion_signal)
        self._panning_state = None

    def _panning_motion(self, widget, event):
        """!
        Panning motion function.

        @param self: class object.
        @param widget: widget.
        @param event: event.
        @return true if successful
        """
        assert self._panning_state is not None
        if event.is_hint:
            pos = widget.get_window().get_device_position(event.device)
            x, y = pos.x, pos.y
        else:
            x, y = event.x, event.y

        hadj = self._scrolled_window.get_hadjustment()
        vadj = self._scrolled_window.get_vadjustment()
        mx0, my0 = self._panning_state.initial_mouse_pos
        cx0, cy0 = self._panning_state.initial_canvas_pos

        dx = x - mx0
        dy = y - my0
        hadj.set_value(cx0 - dx)
        vadj.set_value(cy0 - dy)
        return True

    def _canvas_button_press(self, widget, event):
        if event.button == 2:
            self._begin_panning(widget, event)
            return True
        return False

    def _canvas_button_release(self, dummy_widget, event):
        if event.button == 2:
            self._end_panning(event)
            return True
        return False

    def _canvas_scroll_event(self, dummy_widget, event):
        if event.direction == Gdk.ScrollDirection.UP:
            self.zoom.set_value(self.zoom.get_value() * 1.25)
            return True
        elif event.direction == Gdk.ScrollDirection.DOWN:
            self.zoom.set_value(self.zoom.get_value() / 1.25)
            return True
        return False

    def get_hadjustment(self):
        return self._scrolled_window.get_hadjustment()
    def get_vadjustment(self):
        return self._scrolled_window.get_vadjustment()

    def create_gui(self):
        self.window = Gtk.Window()
        vbox = Gtk.VBox()
        vbox.show()
        self.window.add(vbox)

        # canvas
        self.canvas = GooCanvas.Canvas()
        self.canvas.connect_after("button-press-event", self._canvas_button_press)
        self.canvas.connect_after("button-release-event", self._canvas_button_release)
        self.canvas.connect("scroll-event", self._canvas_scroll_event)
        self.canvas.props.has_tooltip = True
        self.canvas.connect("query-tooltip", self._canvas_tooltip_cb)
        self.canvas.show()
        sw = Gtk.ScrolledWindow(); sw.show()
        self._scrolled_window = sw
        sw.add(self.canvas)
        vbox.pack_start(sw, True, True, 4)
        self.canvas.set_size_request(600, 450)
        self.canvas.set_bounds(-10000, -10000, 10000, 10000)
        self.canvas.scroll_to(0, 0)


        self.canvas.get_root_item().add_child(self.links_group, -1)
        self.links_group.set_property("visibility", GooCanvas.CanvasItemVisibility.VISIBLE)

        self.canvas.get_root_item().add_child(self.channels_group, -1)
        self.channels_group.set_property("visibility", GooCanvas.CanvasItemVisibility.VISIBLE)
        self.channels_group.raise_(self.links_group)

        self.canvas.get_root_item().add_child(self.nodes_group, -1)
        self.nodes_group.set_property("visibility", GooCanvas.CanvasItemVisibility.VISIBLE)
        self.nodes_group.raise_(self.channels_group)

        self.hud = hud.Axes(self)

        hbox = Gtk.HBox(); hbox.show()
        vbox.pack_start(hbox, False, False, 4)

        # zoom
        zoom_adj = Gtk.Adjustment(value=1.0, lower=0.01, upper=10.0,
                                  step_increment=0.02,
                                  page_increment=1.0,
                                  page_size=1.0)
        self.zoom = zoom_adj
        def _zoom_changed(adj):
            self.canvas.set_scale(adj.get_value())
        zoom_adj.connect("value-changed", _zoom_changed)
        zoom = Gtk.SpinButton.new(zoom_adj, 0.1, 1)
        zoom.set_digits(3)
        zoom.show()
        hbox.pack_start(GObject.new(Gtk.Label, label=" Zoom:", visible=True), False, False, 4)
        hbox.pack_start(zoom, False, False, 4)
        _zoom_changed(zoom_adj)

        # speed
        speed_adj = Gtk.Adjustment(value=1.0, lower=0.01, upper=10.0,
                                   step_increment=0.02,
                                   page_increment=1.0, page_size=0)
        def _speed_changed(adj):
            self.speed = adj.get_value()
            self.sample_period = SAMPLE_PERIOD*adj.get_value()
            self._start_update_timer()
        speed_adj.connect("value-changed", _speed_changed)
        speed = Gtk.SpinButton.new(speed_adj, 1, 0)
        speed.set_digits(3)
        speed.show()
        hbox.pack_start(GObject.new(Gtk.Label, label="  Speed:", visible=True), False, False, 4)
        hbox.pack_start(speed, False, False, 4)
        _speed_changed(speed_adj)

        # Current time
        self.time_label = GObject.new(Gtk.Label, label="  Speed:", visible=True)
        self.time_label.set_width_chars(20)
        hbox.pack_start(self.time_label, False, False, 4)

        # Screenshot button
        screenshot_button = GObject.new(Gtk.Button,
                                       label="Snapshot",
                                       relief=Gtk.ReliefStyle.NONE, focus_on_click=False,
                                       visible=True)
        hbox.pack_start(screenshot_button, False, False, 4)

        def load_button_icon(button, icon_name):
            try:
                import gnomedesktop
            except ImportError:
                sys.stderr.write("Could not load icon %s due to missing gnomedesktop Python module\n" % icon_name)
            else:
                icon = gnomedesktop.find_icon(Gtk.IconTheme.get_default(), icon_name, 16, 0)
                if icon is not None:
                    button.props.image = GObject.new(Gtk.Image, file=icon, visible=True)

        load_button_icon(screenshot_button, "applets-screenshooter")
        screenshot_button.connect("clicked", self._take_screenshot)

        # Shell button
        if ipython_view is not None:
            shell_button = GObject.new(Gtk.Button,
                                           label="Shell",
                                           relief=Gtk.ReliefStyle.NONE, focus_on_click=False,
                                           visible=True)
            hbox.pack_start(shell_button, False, False, 4)
            load_button_icon(shell_button, "gnome-terminal")
            shell_button.connect("clicked", self._start_shell)

        # Play button
        self.play_button = GObject.new(Gtk.ToggleButton,
                                       image=GObject.new(Gtk.Image, stock=Gtk.STOCK_MEDIA_PLAY, visible=True),
                                       label="Simulate (F3)",
                                       relief=Gtk.ReliefStyle.NONE, focus_on_click=False,
                                       use_stock=True, visible=True)
        accel_group = Gtk.AccelGroup()
        self.window.add_accel_group(accel_group)
        self.play_button.add_accelerator("clicked", accel_group,
                                         Gdk.KEY_F3, 0, Gtk.AccelFlags.VISIBLE)
        self.play_button.connect("toggled", self._on_play_button_toggled)
        hbox.pack_start(self.play_button, False, False, 4)

        self.canvas.get_root_item().connect("button-press-event", self.on_root_button_press_event)

        vbox.pack_start(self._create_advanced_controls(), False, False, 4)

        display = Gdk.Display.get_default()
        try:
            monitor = display.get_primary_monitor()
            geometry = monitor.get_geometry()
            scale_factor = monitor.get_scale_factor()
        except AttributeError:
            screen = display.get_default_screen()
            monitor_id = screen.get_primary_monitor()
            geometry = screen.get_monitor_geometry(monitor_id)
            scale_factor = screen.get_monitor_scale_factor(monitor_id)
        width = scale_factor * geometry.width
        height = scale_factor * geometry.height
        self.window.set_default_size(width * 2 / 3, height * 2 / 3)
        self.window.show()

    def scan_topology(self):
        print("scanning topology: %i nodes..." % (ns.network.NodeList.GetNNodes(),))
        graph = pygraphviz.AGraph()
        seen_nodes = 0
        for nodeI in range(ns.network.NodeList.GetNNodes()):
            seen_nodes += 1
            if seen_nodes == 100:
                print("scan topology... %i nodes visited (%.1f%%)" % (nodeI, 100*nodeI/ns.network.NodeList.GetNNodes()))
                seen_nodes = 0
            node = ns.network.NodeList.GetNode(nodeI)
            node_name = "Node %i" % nodeI
            node_view = self.get_node(nodeI)

            mobility = node.GetObject(ns.mobility.MobilityModel.GetTypeId())
            if mobility is not None:
                node_view.set_color("red")
                pos = mobility.GetPosition()
                node_view.set_position(*transform_point_simulation_to_canvas(pos.x, pos.y))
                #print "node has mobility position -> ", "%f,%f" % (pos.x, pos.y)
            else:
                graph.add_node(node_name)

            for devI in range(node.GetNDevices()):
                device = node.GetDevice(devI)
                device_traits = lookup_netdevice_traits(type(device))
                if device_traits.is_wireless:
                    continue
                if device_traits.is_virtual:
                    continue
                channel = device.GetChannel()
                if channel.GetNDevices() > 2:
                    if REPRESENT_CHANNELS_AS_NODES:
                        # represent channels as white nodes
                        if mobility is None:
                            channel_name = "Channel %s" % id(channel)
                            graph.add_edge(node_name, channel_name)
                        self.get_channel(channel)
                        self.create_link(self.get_node(nodeI), self.get_channel(channel))
                    else:
                        # don't represent channels, just add links between nodes in the same channel
                        for otherDevI in range(channel.GetNDevices()):
                            otherDev = channel.GetDevice(otherDevI)
                            otherNode = otherDev.GetNode()
                            otherNodeView = self.get_node(otherNode.GetId())
                            if otherNode is not node:
                                if mobility is None and not otherNodeView.has_mobility:
                                    other_node_name = "Node %i" % otherNode.GetId()
                                    graph.add_edge(node_name, other_node_name)
                                self.create_link(self.get_node(nodeI), otherNodeView)
                else:
                    for otherDevI in range(channel.GetNDevices()):
                        otherDev = channel.GetDevice(otherDevI)
                        otherNode = otherDev.GetNode()
                        otherNodeView = self.get_node(otherNode.GetId())
                        if otherNode is not node:
                            if mobility is None and not otherNodeView.has_mobility:
                                other_node_name = "Node %i" % otherNode.GetId()
                                graph.add_edge(node_name, other_node_name)
                            self.create_link(self.get_node(nodeI), otherNodeView)

        print("scanning topology: calling graphviz layout")
        graph.layout(LAYOUT_ALGORITHM)
        for node in graph.iternodes():
            #print node, "=>", node.attr['pos']
            node_type, node_id = node.split(' ')
            pos_x, pos_y = [float(s) for s in node.attr['pos'].split(',')]
            if node_type == 'Node':
                obj = self.nodes[int(node_id)]
            elif node_type == 'Channel':
                obj = self.channels[int(node_id)]
            obj.set_position(pos_x, pos_y)

        print("scanning topology: all done.")
        self.emit("topology-scanned")

    def get_node(self, index):
        try:
            return self.nodes[index]
        except KeyError:
            node = Node(self, index)
            self.nodes[index] = node
            self.nodes_group.add_child(node.canvas_item, -1)
            node.canvas_item.connect("button-press-event", self.on_node_button_press_event, node)
            node.canvas_item.connect("button-release-event", self.on_node_button_release_event, node)
            return node

    def get_channel(self, ns3_channel):
        try:
            return self.channels[id(ns3_channel)]
        except KeyError:
            channel = Channel(ns3_channel)
            self.channels[id(ns3_channel)] = channel
            self.channels_group.add_child(channel.canvas_item, -1)
            return channel

    def create_link(self, node, node_or_channel):
        link = WiredLink(node, node_or_channel)
        self.links_group.add_child(link.canvas_item, -1)
        link.canvas_item.lower(None)

    def update_view(self):
        #print "update_view"

        self.time_label.set_text("Time: %f s" % ns.core.Simulator.Now().GetSeconds())

        self._update_node_positions()

        # Update information
        for info_win in self.information_windows:
            info_win.update()

        self._update_transmissions_view()
        self._update_drops_view()

        self.emit("update-view")

    def _update_node_positions(self):
        for node in self.nodes.values():
            if node.has_mobility:
                ns3_node = ns.network.NodeList.GetNode(node.node_index)
                mobility = ns3_node.GetObject(ns.mobility.MobilityModel.GetTypeId())
                if mobility is not None:
                    pos = mobility.GetPosition()
                    x, y = transform_point_simulation_to_canvas(pos.x, pos.y)
                    node.set_position(x, y)
                    if node is self.follow_node:
                        hadj = self._scrolled_window.get_hadjustment()
                        vadj = self._scrolled_window.get_vadjustment()
                        px, py = self.canvas.convert_to_pixels(x, y)
                        hadj.set_value(px - hadj.get_page_size() / 2)
                        vadj.set_value(py - vadj.get_page_size() / 2)

    def center_on_node(self, node):
        if isinstance(node, ns.network.Node):
            node = self.nodes[node.GetId()]
        elif isinstance(node, (int, long)):
            node = self.nodes[node]
        elif isinstance(node, Node):
            pass
        else:
            raise TypeError("expected int, viz.Node or ns.network.Node, not %r" % node)

        x, y = node.get_position()
        hadj = self._scrolled_window.get_hadjustment()
        vadj = self._scrolled_window.get_vadjustment()
        px, py = self.canvas.convert_to_pixels(x, y)
        hadj.set_value(px - hadj.get_page_size() / 2)
        vadj.set_value(py - vadj.get_page_size() / 2)

    def update_model(self):
        self.simulation.lock.acquire()
        try:
            self.emit("simulation-periodic-update")
        finally:
            self.simulation.lock.release()

    def do_simulation_periodic_update(self):
        smooth_factor = int(self.transmissions_smoothing_adjustment.get_value()*10)

        transmissions = self.simulation.sim_helper.GetTransmissionSamples()
        self._last_transmissions.append(transmissions)
        while len(self._last_transmissions) > smooth_factor:
            self._last_transmissions.pop(0)

        drops = self.simulation.sim_helper.GetPacketDropSamples()
        self._last_drops.append(drops)
        while len(self._last_drops) > smooth_factor:
            self._last_drops.pop(0)

    def _get_label_over_line_position(self, pos1_x, pos1_y, pos2_x, pos2_y):
        hadj = self._scrolled_window.get_hadjustment()
        vadj = self._scrolled_window.get_vadjustment()
        bounds_x1, bounds_y1 = self.canvas.convert_from_pixels(hadj.get_value(), vadj.get_value())
        bounds_x2, bounds_y2 = self.canvas.convert_from_pixels(hadj.get_value() + hadj.get_page_size(),
                                                               vadj.get_value() + vadj.get_page_size())
        pos1_x, pos1_y, pos2_x, pos2_y = ns.visualizer.PyViz.LineClipping(bounds_x1, bounds_y1,
                                                                bounds_x2, bounds_y2,
                                                                pos1_x, pos1_y,
                                                                pos2_x, pos2_y)
        return (pos1_x + pos2_x)/2, (pos1_y + pos2_y)/2

    def _update_transmissions_view(self):
        transmissions_average = {}
        for transmission_set in self._last_transmissions:
            for transmission in transmission_set:
                key = (transmission.transmitter.GetId(), transmission.receiver.GetId())
                rx_bytes, count = transmissions_average.get(key, (0, 0))
                rx_bytes += transmission.bytes
                count += 1
                transmissions_average[key] = rx_bytes, count

        old_arrows = self._transmission_arrows
        for arrow, label in old_arrows:
            arrow.set_property("visibility", GooCanvas.CanvasItemVisibility.HIDDEN)
            label.set_property("visibility", GooCanvas.CanvasItemVisibility.HIDDEN)
        new_arrows = []

        k = self.node_size_adjustment.get_value()/5

        for (transmitter_id, receiver_id), (rx_bytes, rx_count) in transmissions_average.items():
            transmitter = self.get_node(transmitter_id)
            receiver = self.get_node(receiver_id)
            try:
                arrow, label = old_arrows.pop()
            except IndexError:
                arrow = GooCanvas.CanvasPolyline(line_width=2.0, stroke_color_rgba=0x00C000C0, close_path=False, end_arrow=True, pointer_events=GooCanvas.CanvasPointerEvents.NONE)
                arrow.set_property("parent", self.canvas.get_root_item())
                arrow.raise_(None)

                label = GooCanvas.CanvasText(parent=self.canvas.get_root_item(), pointer_events=GooCanvas.CanvasPointerEvents.NONE)
                label.raise_(None)

            arrow.set_property("visibility", GooCanvas.CanvasItemVisibility.VISIBLE)
            line_width = max(0.1, math.log(float(rx_bytes)/rx_count/self.sample_period)*k)
            arrow.set_property("line-width", line_width)

            pos1_x, pos1_y = transmitter.get_position()
            pos2_x, pos2_y = receiver.get_position()
            points = GooCanvas.CanvasPoints.new(2)
            points.set_point(0, pos1_x, pos1_y)
            points.set_point(1, pos2_x, pos2_y)
            arrow.set_property("points", points)

            kbps = float(rx_bytes*8)/1e3/rx_count/self.sample_period
            label.set_properties(visibility=GooCanvas.CanvasItemVisibility.VISIBLE_ABOVE_THRESHOLD,
                                 visibility_threshold=0.5,
                                 font=("Sans Serif %f" % int(1+BITRATE_FONT_SIZE*k)))
            angle = math.atan2((pos2_y - pos1_y), (pos2_x - pos1_x))
            if -PI_OVER_2 <= angle <= PI_OVER_2:
                label.set_properties(text=("%.2f kbit/s →" % (kbps,)),
                                     alignment=Pango.Alignment.CENTER,
                                     anchor=GooCanvas.CanvasAnchorType.S,
                                     x=0, y=-line_width/2)
            else:
                label.set_properties(text=("← %.2f kbit/s" % (kbps,)),
                                     alignment=Pango.Alignment.CENTER,
                                     anchor=GooCanvas.CanvasAnchorType.N,
                                     x=0, y=line_width/2)
            M = cairo.Matrix()
            lx, ly = self._get_label_over_line_position(pos1_x, pos1_y,
                                                        pos2_x, pos2_y)
            M.translate(lx, ly)
            M.rotate(angle)
            try:
                label.set_transform(M)
            except KeyError:
                # https://gitlab.gnome.org/GNOME/pygobject/issues/16
                warnings.warn("PyGobject bug causing label position error; "
                              "should be fixed in PyGObject >= 3.29.1")
                label.set_properties(x=(lx + label.props.x),
                                     y=(ly + label.props.y))

            new_arrows.append((arrow, label))

        self._transmission_arrows = new_arrows + old_arrows


    def _update_drops_view(self):
        drops_average = {}
        for drop_set in self._last_drops:
            for drop in drop_set:
                key = drop.transmitter.GetId()
                drop_bytes, count = drops_average.get(key, (0, 0))
                drop_bytes += drop.bytes
                count += 1
                drops_average[key] = drop_bytes, count

        old_arrows = self._drop_arrows
        for arrow, label in old_arrows:
            arrow.set_property("visibility", GooCanvas.CanvasItemVisibility.HIDDEN)
            label.set_property("visibility", GooCanvas.CanvasItemVisibility.HIDDEN)
        new_arrows = []

        # get the coordinates for the edge of screen
        vadjustment = self._scrolled_window.get_vadjustment()
        bottom_y = vadjustment.get_value() + vadjustment.get_page_size()
        dummy, edge_y = self.canvas.convert_from_pixels(0, bottom_y)

        k = self.node_size_adjustment.get_value()/5

        for transmitter_id, (drop_bytes, drop_count) in drops_average.items():
            transmitter = self.get_node(transmitter_id)
            try:
                arrow, label = old_arrows.pop()
            except IndexError:
                arrow = GooCanvas.CanvasPolyline(line_width=2.0, stroke_color_rgba=0xC00000C0, close_path=False, end_arrow=True, pointer_events=GooCanvas.CanvasPointerEvents.NONE)
                arrow.set_property("parent", self.canvas.get_root_item())
                arrow.raise_(None)

                label = GooCanvas.CanvasText(pointer_events=GooCanvas.CanvasPointerEvents.NONE)#, fill_color_rgba=0x00C000C0)
                label.set_property("parent", self.canvas.get_root_item())
                label.raise_(None)

            arrow.set_property("visibility", GooCanvas.CanvasItemVisibility.VISIBLE)
            arrow.set_property("line-width", max(0.1, math.log(float(drop_bytes)/drop_count/self.sample_period)*k))
            pos1_x, pos1_y = transmitter.get_position()
            pos2_x, pos2_y = pos1_x, edge_y
            points = GooCanvas.CanvasPoints.new(2)
            points.set_point(0, pos1_x, pos1_y)
            points.set_point(1, pos2_x, pos2_y)
            arrow.set_property("points", points)

            label.set_properties(visibility=GooCanvas.CanvasItemVisibility.VISIBLE_ABOVE_THRESHOLD,
                                 visibility_threshold=0.5,
                                 font=("Sans Serif %i" % int(1+BITRATE_FONT_SIZE*k)),
                                 text=("%.2f kbit/s" % (float(drop_bytes*8)/1e3/drop_count/self.sample_period,)),
                                 alignment=Pango.Alignment.CENTER,
                                 x=(pos1_x + pos2_x)/2,
                                 y=(pos1_y + pos2_y)/2)

            new_arrows.append((arrow, label))

        self._drop_arrows = new_arrows + old_arrows


    def update_view_timeout(self):
        #print "view: update_view_timeout called at real time ", time.time()

        # while the simulator is busy, run the gtk event loop
        while not self.simulation.lock.acquire(False):
            while Gtk.events_pending():
                Gtk.main_iteration()
        pause_messages = self.simulation.pause_messages
        self.simulation.pause_messages = []
        try:
            self.update_view()
            self.simulation.target_time = ns.core.Simulator.Now ().GetSeconds () + self.sample_period
            #print "view: target time set to %f" % self.simulation.target_time
        finally:
            self.simulation.lock.release()

        if pause_messages:
            #print pause_messages
            dialog = Gtk.MessageDialog(parent=self.window, flags=0, type=Gtk.MessageType.WARNING, buttons=Gtk.ButtonsType.OK,
                                       message_format='\n'.join(pause_messages))
            dialog.connect("response", lambda d, r: d.destroy())
            dialog.show()
            self.play_button.set_active(False)

        # if we're paused, stop the update timer
        if not self.play_button.get_active():
            self._update_timeout_id = None
            return False

        #print "view: self.simulation.go.set()"
        self.simulation.go.set()
        #print "view: done."
        return True

    def _start_update_timer(self):
        if self._update_timeout_id is not None:
            GLib.source_remove(self._update_timeout_id)
        #print "start_update_timer"
        self._update_timeout_id = GLib.timeout_add(int(SAMPLE_PERIOD/min(self.speed, 1)*1e3),
                                                   self.update_view_timeout,
                                                   priority=PRIORITY_UPDATE_VIEW)

    def _on_play_button_toggled(self, button):
        if button.get_active():
            self._start_update_timer()
        else:
            if self._update_timeout_id is not None:
                GLib.source_remove(self._update_timeout_id)

    def _quit(self, *dummy_args):
        if self._update_timeout_id is not None:
            GLib.source_remove(self._update_timeout_id)
            self._update_timeout_id = None
        self.simulation.quit = True
        self.simulation.go.set()
        self.simulation.join()
        Gtk.main_quit()

    def _monkey_patch_ipython(self):
        # The user may want to access the NS 3 simulation state, but
        # NS 3 is not thread safe, so it could cause serious problems.
        # To work around this, monkey-patch IPython to automatically
        # acquire and release the simulation lock around each code
        # that is executed.

        original_runcode = self.ipython.runcode
        def runcode(ip, *args):
            #print "lock"
            self.simulation.lock.acquire()
            try:
                return original_runcode(*args)
            finally:
                #print "unlock"
                self.simulation.lock.release()
        import types
        self.ipython.runcode = types.MethodType(runcode, self.ipython)

    def autoscale_view(self):
        if not self.nodes:
            return
        self._update_node_positions()
        positions = [node.get_position() for node in self.nodes.values()]
        min_x, min_y = min(x for (x,y) in positions), min(y for (x,y) in positions)
        max_x, max_y = max(x for (x,y) in positions), max(y for (x,y) in positions)
        min_x_px, min_y_px = self.canvas.convert_to_pixels(min_x, min_y)
        max_x_px, max_y_px = self.canvas.convert_to_pixels(max_x, max_y)
        dx = max_x - min_x
        dy = max_y - min_y
        dx_px = max_x_px - min_x_px
        dy_px = max_y_px - min_y_px
        hadj = self._scrolled_window.get_hadjustment()
        vadj = self._scrolled_window.get_vadjustment()
        new_dx, new_dy = 1.5*dx_px, 1.5*dy_px

        if new_dx == 0 or new_dy == 0:
            return

        self.zoom.set_value(min(hadj.get_page_size()/new_dx, vadj.get_page_size()/new_dy))

        x1, y1 = self.canvas.convert_from_pixels(hadj.get_value(), vadj.get_value())
        x2, y2 = self.canvas.convert_from_pixels((hadj.get_value() +
                                                  hadj.get_page_size()),
                                                 (vadj.get_value() +
                                                  vadj.get_page_size()))
        width = x2 - x1
        height = y2 - y1
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2

        self.canvas.scroll_to(center_x - width/2, center_y - height/2)

        return False

    def start(self):
        self.scan_topology()
        self.window.connect("delete-event", self._quit)
        #self._start_update_timer()
        GLib.timeout_add(200, self.autoscale_view)
        self.simulation.start()

        try:
            __IPYTHON__
        except NameError:
            pass
        else:
            self._monkey_patch_ipython()

        Gtk.main()


    def on_root_button_press_event(self, view, target, event):
        if event.button == 1:
            self.select_node(None)
            return True

    def on_node_button_press_event(self, view, target, event, node):
        button = event.button
        if button == 1:
            self.select_node(node)
            return True
        elif button == 3:
            self.popup_node_menu(node, event)
            return True
        elif button == 2:
            self.begin_node_drag(node, event)
            return True
        return False

    def on_node_button_release_event(self, view, target, event, node):
        if event.button == 2:
            self.end_node_drag(node)
            return True
        return False

    class NodeDragState(object):
        def __init__(self, canvas_x0, canvas_y0, sim_x0, sim_y0):
            self.canvas_x0 = canvas_x0
            self.canvas_y0 = canvas_y0
            self.sim_x0 = sim_x0
            self.sim_y0 = sim_y0
            self.motion_signal = None

    def begin_node_drag(self, node, event):
        self.simulation.lock.acquire()
        try:
            ns3_node = ns.network.NodeList.GetNode(node.node_index)
            mob = ns3_node.GetObject(ns.mobility.MobilityModel.GetTypeId())
            if mob is None:
                return
            if self.node_drag_state is not None:
                return
            pos = mob.GetPosition()
        finally:
            self.simulation.lock.release()
        devpos = self.canvas.get_window().get_device_position(event.device)
        x0, y0 = self.canvas.convert_from_pixels(devpos.x, devpos.y)
        self.node_drag_state = self.NodeDragState(x0, y0, pos.x, pos.y)
        self.node_drag_state.motion_signal = node.canvas_item.connect("motion-notify-event", self.node_drag_motion, node)

    def node_drag_motion(self, item, targe_item, event, node):
        self.simulation.lock.acquire()
        try:
            ns3_node = ns.network.NodeList.GetNode(node.node_index)
            mob = ns3_node.GetObject(ns.mobility.MobilityModel.GetTypeId())
            if mob is None:
                return False
            if self.node_drag_state is None:
                return False
            devpos = self.canvas.get_window().get_device_position(event.device)
            canvas_x, canvas_y = self.canvas.convert_from_pixels(devpos.x, devpos.y)
            dx = (canvas_x - self.node_drag_state.canvas_x0)
            dy = (canvas_y - self.node_drag_state.canvas_y0)
            pos = mob.GetPosition()
            pos.x = self.node_drag_state.sim_x0 + transform_distance_canvas_to_simulation(dx)
            pos.y = self.node_drag_state.sim_y0 + transform_distance_canvas_to_simulation(dy)
            #print "SetPosition(%G, %G)" % (pos.x, pos.y)
            mob.SetPosition(pos)
            node.set_position(*transform_point_simulation_to_canvas(pos.x, pos.y))
        finally:
            self.simulation.lock.release()
        return True

    def end_node_drag(self, node):
        if self.node_drag_state is None:
            return
        node.canvas_item.disconnect(self.node_drag_state.motion_signal)
        self.node_drag_state = None

    def popup_node_menu(self, node, event):
        menu = Gtk.Menu()
        self.emit("populate-node-menu", node, menu)
        menu.popup(None, None, None, None, event.button, event.time)

    def _update_ipython_selected_node(self):
        # If we are running under ipython -gthread, make this new
        # selected node available as a global 'selected_node'
        # variable.
        try:
            __IPYTHON__
        except NameError:
            pass
        else:
            if self.selected_node is None:
                ns3_node = None
            else:
                self.simulation.lock.acquire()
                try:
                    ns3_node = ns.network.NodeList.GetNode(self.selected_node.node_index)
                finally:
                    self.simulation.lock.release()
            self.ipython.updateNamespace({'selected_node': ns3_node})


    def select_node(self, node):
        if isinstance(node, ns.network.Node):
            node = self.nodes[node.GetId()]
        elif isinstance(node, (int, long)):
            node = self.nodes[node]
        elif isinstance(node, Node):
            pass
        elif node is None:
            pass
        else:
            raise TypeError("expected None, int, viz.Node or ns.network.Node, not %r" % node)

        if node is self.selected_node:
            return

        if self.selected_node is not None:
            self.selected_node.selected = False
        self.selected_node = node
        if self.selected_node is not None:
            self.selected_node.selected = True

        if self._show_transmissions_mode == ShowTransmissionsMode.SELECTED:
            if self.selected_node is None:
                self.simulation.set_nodes_of_interest([])
            else:
                self.simulation.set_nodes_of_interest([self.selected_node.node_index])

        self._update_ipython_selected_node()


    def add_information_window(self, info_win):
        self.information_windows.append(info_win)
        self.simulation.lock.acquire()
        try:
            info_win.update()
        finally:
            self.simulation.lock.release()

    def remove_information_window(self, info_win):
        self.information_windows.remove(info_win)

    def _canvas_tooltip_cb(self, canvas, x, y, keyboard_mode, tooltip):
        #print "tooltip query: ", x, y
        hadj = self._scrolled_window.get_hadjustment()
        vadj = self._scrolled_window.get_vadjustment()
        x, y = self.canvas.convert_from_pixels(hadj.get_value() + x, vadj.get_value() + y)
        item = self.canvas.get_item_at(x, y, True)
        #print "items at (%f, %f): %r | keyboard_mode=%r" % (x, y, item, keyboard_mode)
        if not item:
            return False
        while item is not None:
            obj = getattr(item, "pyviz_object", None)
            if obj is not None:
                obj.tooltip_query(tooltip)
                return True
            item = item.props.parent
        return False

    def _get_export_file_name(self):
        sel = Gtk.FileChooserDialog("Save...", self.canvas.get_toplevel(),
                                    Gtk.FileChooserAction.SAVE,
                                    (Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
                                     Gtk.STOCK_SAVE, Gtk.ResponseType.OK))
        sel.set_default_response(Gtk.ResponseType.OK)
        sel.set_local_only(True)
        sel.set_do_overwrite_confirmation(True)
        sel.set_current_name("Unnamed.pdf")

        filter = Gtk.FileFilter()
        filter.set_name("Embedded PostScript")
        filter.add_mime_type("image/x-eps")
        sel.add_filter(filter)

        filter = Gtk.FileFilter()
        filter.set_name("Portable Document Graphics")
        filter.add_mime_type("application/pdf")
        sel.add_filter(filter)

        filter = Gtk.FileFilter()
        filter.set_name("Scalable Vector Graphics")
        filter.add_mime_type("image/svg+xml")
        sel.add_filter(filter)

        resp = sel.run()
        if resp != Gtk.ResponseType.OK:
            sel.destroy()
            return None

        file_name = sel.get_filename()
        sel.destroy()
        return file_name

    def _take_screenshot(self, dummy_button):
        #print "Cheese!"
        file_name = self._get_export_file_name()
        if file_name is None:
            return

        # figure out the correct bounding box for what is visible on screen
        x1 = self._scrolled_window.get_hadjustment().get_value()
        y1 = self._scrolled_window.get_vadjustment().get_value()
        x2 = x1 + self._scrolled_window.get_hadjustment().get_page_size()
        y2 = y1 + self._scrolled_window.get_vadjustment().get_page_size()
        bounds = GooCanvas.CanvasBounds()
        bounds.x1, bounds.y1 = self.canvas.convert_from_pixels(x1, y1)
        bounds.x2, bounds.y2 = self.canvas.convert_from_pixels(x2, y2)
        dest_width = bounds.x2 - bounds.x1
        dest_height = bounds.y2 - bounds.y1
        #print bounds.x1, bounds.y1, " -> ", bounds.x2, bounds.y2

        dummy, extension = os.path.splitext(file_name)
        extension = extension.lower()
        if extension == '.eps':
            surface = cairo.PSSurface(file_name, dest_width, dest_height)
        elif extension == '.pdf':
            surface = cairo.PDFSurface(file_name, dest_width, dest_height)
        elif extension == '.svg':
            surface = cairo.SVGSurface(file_name, dest_width, dest_height)
        else:
            dialog = Gtk.MessageDialog(parent  = self.canvas.get_toplevel(),
                		       flags   = Gtk.DialogFlags.DESTROY_WITH_PARENT,
                		       type    = Gtk.MessageType.ERROR,
                		       buttons = Gtk.ButtonsType.OK,
                		       message_format = "Unknown extension '%s' (valid extensions are '.eps', '.svg', and '.pdf')"
                                                          % (extension,))
            dialog.run()
            dialog.destroy()
            return

        # draw the canvas to a printing context
        cr = cairo.Context(surface)
        cr.translate(-bounds.x1, -bounds.y1)
        self.canvas.render(cr, bounds, self.zoom.get_value())
        cr.show_page()
        surface.finish()

    def set_follow_node(self, node):
        if isinstance(node, ns.network.Node):
            node = self.nodes[node.GetId()]
        self.follow_node = node

    def _start_shell(self, dummy_button):
        if self.shell_window is not None:
            self.shell_window.present()
            return

        self.shell_window = Gtk.Window()
        self.shell_window.set_size_request(750,550)
        self.shell_window.set_resizable(True)
        scrolled_window = Gtk.ScrolledWindow()
        scrolled_window.set_policy(Gtk.PolicyType.AUTOMATIC,
                                   Gtk.PolicyType.AUTOMATIC)
        self.ipython = ipython_view.IPythonView()
        self.ipython.modify_font(Pango.FontDescription(SHELL_FONT))
        self.ipython.set_wrap_mode(Gtk.WrapMode.CHAR)
        self.ipython.show()
        scrolled_window.add(self.ipython)
        scrolled_window.show()
        self.shell_window.add(scrolled_window)
        self.shell_window.show()
        self.shell_window.connect('destroy', self._on_shell_window_destroy)

        self._update_ipython_selected_node()
        self.ipython.updateNamespace({'viz': self})


    def _on_shell_window_destroy(self, window):
        self.shell_window = None


initialization_hooks = []

def add_initialization_hook(hook, *args):
    """
    Adds a callback to be called after
    the visualizer is initialized, like this::
       initialization_hook(visualizer, *args)
    """
    global initialization_hooks
    initialization_hooks.append((hook, args))


def set_bounds(x1, y1, x2, y2):
    assert x2>x1
    assert y2>y1
    def hook(viz):
        cx1, cy1 = transform_point_simulation_to_canvas(x1, y1)
        cx2, cy2 = transform_point_simulation_to_canvas(x2, y2)
        viz.canvas.set_bounds(cx1, cy1, cx2, cy2)
    add_initialization_hook(hook)


def start():
    assert Visualizer.INSTANCE is None
    if _import_error is not None:
        import sys
        print("No visualization support (%s)." % (str(_import_error),),
              file=sys.stderr)
        ns.core.Simulator.Run()
        return
    load_plugins()
    viz = Visualizer()
    for hook, args in initialization_hooks:
        GLib.idle_add(hook, viz, *args)
    ns.network.Packet.EnablePrinting()
    viz.start()
