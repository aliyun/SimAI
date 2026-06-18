import math
from .base import PIXELS_PER_METER
from gi.repository import Pango
from gi.repository import Gtk
from gi.repository import GooCanvas


## Axes class
class Axes(object):
    ## @var viz
    #  visualizer
    ## @var color
    #  color
    ## @var hlines
    #  horizontal lines
    ## @var vlines
    #  vertical lines
    ## @var labels
    #  list of labels
    ## @var visible
    #  visible
    def __init__(self, viz):
        """!
        Initializer function

        @param self: this object
        @param viz: visualization object
        """
        self.viz = viz
        self.color = 0x8080C0FF
        self.hlines = GooCanvas.CanvasPath(parent=viz.canvas.get_root_item(), stroke_color_rgba=self.color)
        self.hlines.lower(None)
        self.vlines = GooCanvas.CanvasPath(parent=viz.canvas.get_root_item(), stroke_color_rgba=self.color)
        self.vlines.lower(None)
        self.labels = []
        hadj = self.viz.get_hadjustment()
        vadj = self.viz.get_vadjustment()
        def update(adj):
            if self.visible:
                self.update_view()
        hadj.connect("value-changed", update)
        vadj.connect("value-changed", update)
        hadj.connect("changed", update)
        vadj.connect("changed", update)
        self.visible = True
        self.update_view()

    def set_visible(self, visible):
        """!
        Set visible function

        @param self: this object
        @param visible: visible indicator
        @return none
        """
        self.visible = visible
        if self.visible:
            self.hlines.props.visibility = GooCanvas.CanvasItemVisibility.VISIBLE
            self.vlines.props.visibility = GooCanvas.CanvasItemVisibility.VISIBLE
        else:
            self.hlines.props.visibility = GooCanvas.CanvasItemVisibility.HIDDEN
            self.vlines.props.visibility = GooCanvas.CanvasItemVisibility.HIDDEN
            for label in self.labels:
                label.props.visibility = GooCanvas.CanvasItemVisibility.HIDDEN

    def _compute_divisions(self, xi, xf):
        """!
        Compute divisions function

        @param self: this object
        @param xi: xi
        @param xf: xf
        @return x0 and div
        """
        assert xf > xi
        dx = xf - xi
        size = dx
        ndiv = 5
        text_width = dx/ndiv/2

        def rint(x):
            """!
            Compute divisions function

            @param x: x
            @return x rounded up
            """
            return math.floor(x+0.5)

        dx_over_ndiv = dx / ndiv
        for n in range(5): # iterate 5 times to find optimum division size
            #/* div: length of each division */
            tbe = math.log10(dx_over_ndiv)#;   /* looking for approx. 'ndiv' divisions in a length 'dx' */
            div = pow(10, rint(tbe))#;	/* div: power of 10 closest to dx/ndiv */
            if math.fabs(div/2 - dx_over_ndiv) < math.fabs(div - dx_over_ndiv): #/* test if div/2 is closer to dx/ndiv */
                div /= 2
            elif math.fabs(div*2 - dx_over_ndiv) < math.fabs(div - dx_over_ndiv):
                div *= 2 #			/* test if div*2 is closer to dx/ndiv */
            x0 = div*math.ceil(xi / div) - div
            if n > 1:
                ndiv = rint(size / text_width)
        return x0, div


    def update_view(self):
        """!
        Update view function

        @param self: this object
        @return none
        """
        if self.viz.zoom is None:
            return

        unused_labels = self.labels
        self.labels = []
        for label in unused_labels:
            label.set_property("visibility", GooCanvas.CanvasItemVisibility.HIDDEN)
        def get_label():
            """!
            Get label function

            @param self: this object
            @return label
            """
            try:
                label = unused_labels.pop(0)
            except IndexError:
                label = GooCanvas.CanvasText(parent=self.viz.canvas.get_root_item(), stroke_color_rgba=self.color)
            else:
                label.set_property("visibility", GooCanvas.CanvasItemVisibility.VISIBLE)
                label.lower(None)
            self.labels.append(label)
            return label

        hadj = self.viz.get_hadjustment()
        vadj = self.viz.get_vadjustment()
        zoom = self.viz.zoom.get_value()
        offset = 10/zoom

        x1, y1 = self.viz.canvas.convert_from_pixels(hadj.get_value(), vadj.get_value())
        x2, y2 = self.viz.canvas.convert_from_pixels(hadj.get_value() + hadj.get_page_size(), vadj.get_value() + vadj.get_page_size())
        line_width = 5.0/self.viz.zoom.get_value()

        # draw the horizontal axis
        self.hlines.set_property("line-width", line_width)
        yc = y2 - line_width/2

        sim_x1 = x1/PIXELS_PER_METER
        sim_x2 = x2/PIXELS_PER_METER
        x0, xdiv = self._compute_divisions(sim_x1, sim_x2)
        path = ["M %r %r L %r %r" % (x1, yc, x2, yc)]
        x = x0
        while x < sim_x2:
            path.append("M %r %r L %r %r" % (PIXELS_PER_METER*x, yc - offset, PIXELS_PER_METER*x, yc))
            label = get_label()
            label.set_properties(font=("Sans Serif %f" % int(12/zoom)),
                                 text=("%G" % x),
                                 fill_color_rgba=self.color,
                                 alignment=Pango.Alignment.CENTER,
                                 # anchor=Gtk.Widget.ANCHOR_S,
                                 x=PIXELS_PER_METER*x,
                                 y=(yc - offset))
            x += xdiv
        del x

        self.hlines.set_property("data", " ".join(path))

        # draw the vertical axis
        self.vlines.set_property("line-width", line_width)
        xc = x1 + line_width/2

        sim_y1 = y1/PIXELS_PER_METER
        sim_y2 = y2/PIXELS_PER_METER


        y0, ydiv = self._compute_divisions(sim_y1, sim_y2)
        path = ["M %r %r L %r %r" % (xc, y1, xc, y2)]
        y = y0
        while y < sim_y2:
            path.append("M %r %r L %r %r" % (xc, PIXELS_PER_METER*y, xc + offset, PIXELS_PER_METER*y))
            label = get_label()
            label.set_properties(font=("Sans Serif %f" % int(12/zoom)),
                                 text=("%G" % y),
                                 fill_color_rgba=self.color,
                                 alignment=Pango.Alignment.LEFT,
                                 # anchor=Gtk.ANCHOR_W,
                                 x=xc + offset,
                                 y=PIXELS_PER_METER*y)
            y += ydiv

        self.vlines.set_property("data", " ".join(path))



        self.labels.extend(unused_labels)
