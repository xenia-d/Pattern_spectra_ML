"""
gui.py - Python/PyQt equivalent of gui.c (interactive GUI)
Requires: PyQt5 or PySide6, numpy
Run: python gui.py
"""

import sys
import os
import math
import numpy as np
from typing import List, Optional

# Prefer PyQt5, fall back to PySide6
try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QLabel, QWidget, QGridLayout, QScrollArea,
        QVBoxLayout, QPushButton, QHBoxLayout, QGroupBox, QFormLayout, QSpinBox,
        QDoubleSpinBox, QComboBox
    )
    from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor
    from PyQt5.QtCore import Qt, QSize, QPoint
    QT_BACKEND = "PyQt5"
except Exception:
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QLabel, QWidget, QGridLayout, QScrollArea,
        QVBoxLayout, QPushButton, QHBoxLayout, QGroupBox, QFormLayout, QSpinBox,
        QDoubleSpinBox, QComboBox
    )
    from PySide6.QtGui import QPixmap, QImage, QPainter, QColor
    from PySide6.QtCore import Qt, QSize, QPoint
    QT_BACKEND = "PySide6"

print(f"Using Qt backend: {QT_BACKEND}")

# ---------------------------
# Minimal placeholder MaxTree and ProcSet classes
# ---------------------------

class Node:
    def __init__(self, parent:int, attributes:List[float], area:int=1, level:float=0.0):
        self.parent = parent
        self.attributes = attributes
        self.NodeStatus = 0
        self.Pos = [0]*len(attributes)
        self.Area = area
        self.Level = level


class MaxTree:
    """
    Minimal flat MaxTree representation. In the original C code, the
    tree is stored in a flat array where nodes are grouped by levels.
    This placeholder stores:
      - nodes: list[Node]
      - nodes_per_level: list[int]
    """
    def __init__(self, nodes:List[Node]=None, nodes_per_level:List[int]=None):
        self.nodes = nodes or []
        self.nodes_per_level = nodes_per_level or [len(self.nodes)]

    def get_number_of_nodes(self, level:int) -> int:
        return self.nodes_per_level[level]

    def get_num_pixels_below_level(self, level:int) -> int:
        return sum(self.nodes_per_level[:level])

    def __len__(self):
        return len(self.nodes)


class Proc:
    """
    Minimal Proc wrapper. The real code uses procs[j].Attribute(..) and procs[j].Mapper(...)
    For now, attribute_func should accept the stored attribute and return scalar; mapper_func
    maps that scalar into a bin index 0..dims[j].
    """
    def __init__(self, attribute_func, mapper_func):
        self.Attribute = attribute_func
        self.Mapper = mapper_func


# ---------------------------
# GUIInfo class (translation of GUIInfoCreate and fields)
# ---------------------------

class GUIInfo:
    def __init__(self,
                 diatom_pgm: Optional[np.ndarray],
                 shape_pgm: Optional[np.ndarray],
                 width: int,
                 height: int,
                 tree: MaxTree,
                 filter_mode: int,
                 k: int,
                 mingreylevel: int,
                 numattrs: int,
                 procs: List[Proc],
                 dims: List[int],
                 dls: List[float],
                 dhs: List[float],
                 sharedcm: bool=False):
        # core fields from C struct
        self.Tree = tree
        self.Filter = filter_mode
        self.k = k
        self.mingreylevel = mingreylevel
        self.NumAttrs = numattrs
        self.Procs = procs
        self.Dims = dims
        self.DLs = dls
        self.DHs = dhs

        # allocate result array
        self.Result = gui_alloc_result_array(numattrs, dims)

        # images (numpy uint8)
        self.DiatomPGM = diatom_pgm if diatom_pgm is not None else np.zeros((height, width), dtype=np.uint8)
        self.DiatomWidth = width
        self.DiatomHeight = height
        self.ShapePGM = shape_pgm if shape_pgm is not None else np.ones((height, width), dtype=np.uint8)

        # compute granulometry (placeholder)
        GUIComputeGranulometry(self)

        # create granulometry PGM (scale up by 8)
        self.GranPGM = None
        GUICreateGranulometryImage(self)

        # GUI handles (populated by window open functions)
        self.Display = None  # no direct equivalent; kept for API parity
        self.ControlWindow = None
        self.DiatomWindow = None
        self.DiatomImage = None
        self.GranWindow = None
        self.GranImage = None
        self.NoiseInfo = None
        self.AboutWindow = None

# ---------------------------
# Utility equivalents
# ---------------------------

def gui_alloc_result_array(num_attrs:int, dims:List[int]) -> np.ndarray:
    """
    Equivalent of GUIAllocResultArray: allocate a multi-dimensional array
    of shape (dims[0]+1, dims[1]+1, ..., dims[num_attrs-1]+1) dtype=float64.
    """
    if len(dims) != num_attrs:
        raise ValueError("dims length must equal num_attrs")
    shape = tuple(d + 1 for d in dims)
    return np.zeros(shape, dtype=np.float64)


# ---------------------------
# Granulometry compute dispatcher (placeholders)
# ---------------------------

def MaxTreeGranulometryMDMin(tree, num_attrs, procs, dims, result, dls, dhs):
    # Placeholder: fill with synthetic data
    print("[DEBUG] MaxTreeGranulometryMDMin stub called")
    # create 2D result when num_attrs>=2, else fill flat
    if result.ndim >= 2:
        # normalize using a simple gradient for demo
        yy, xx = np.indices(result.shape[:2])
        result[:, :] = (xx + yy).astype(np.float64)
    else:
        result[:] = np.arange(result.size, dtype=np.float64)

def MaxTreeGranulometryMDDirect(tree, num_attrs, procs, dims, result, dls, dhs):
    print("[DEBUG] MaxTreeGranulometryMDDirect stub called")
    MaxTreeGranulometryMDMin(tree, num_attrs, procs, dims, result, dls, dhs)

def MaxTreeGranulometryMDMax(tree, num_attrs, procs, dims, result, dls, dhs):
    print("[DEBUG] MaxTreeGranulometryMDMax stub called")
    MaxTreeGranulometryMDMin(tree, num_attrs, procs, dims, result, dls, dhs)

def MaxTreeGranulometryMDWilkinsonK(tree, num_attrs, procs, dims, result, dls, dhs, k):
    print("[DEBUG] MaxTreeGranulometryMDWilkinsonK stub called (k={})".format(k))
    MaxTreeGranulometryMDMin(tree, num_attrs, procs, dims, result, dls, dhs)

def GUIComputeGranulometry(guiinfo:GUIInfo):
    """
    Dispatch to the chosen MaxTree granulometry implementation (stubs currently).
    """
    filter_type = guiinfo.Filter
    tree = guiinfo.Tree
    num_attrs = guiinfo.NumAttrs
    procs = guiinfo.Procs
    dims = guiinfo.Dims
    dls = guiinfo.DLs
    dhs = guiinfo.DHs
    k = guiinfo.k
    result = guiinfo.Result

    if filter_type == 0:
        MaxTreeGranulometryMDMin(tree, num_attrs, procs, dims, result, dls, dhs)
    elif filter_type == 1:
        MaxTreeGranulometryMDDirect(tree, num_attrs, procs, dims, result, dls, dhs)
    elif filter_type == 2:
        MaxTreeGranulometryMDMax(tree, num_attrs, procs, dims, result, dls, dhs)
    elif filter_type == 3:
        MaxTreeGranulometryMDWilkinsonK(tree, num_attrs, procs, dims, result, dls, dhs, k)
    else:
        raise ValueError("Unknown filter type: {}".format(filter_type))


# ---------------------------
# Granulometry image creation (scale result into an 8x magnified image)
# ---------------------------

def GUIInitGranulometryImage(info:GUIInfo):
    """
    Fills info.GranPGM (flat array) with brightness values scaled to 0-255.
    The C code uses: GranPGM[(y*8+b)*8*(Dims[0])+x*8+a] = (log(1+Result[y*(Dims[0]+1)+x]) / gmax) * 255
    We'll implement the equivalent for 2D result arrays.
    """
    dims = info.Dims
    if info.Result is None:
        return
    # Expecting result shape >=2 dims; for safety, flatten/reshape if needed
    # We'll interpret the first two axes as x,y (width,height)
    # If result has shape (w+1, h+1, ...), follow original addressing: y*(dims[0]+1)+x
    # For convenience, create a 2D grid of (dims[1], dims[0])
    w = dims[0]
    h = dims[1]
    # If result is 2D as (h+1, w+1) or (w+1,h+1), handle both:
    res = info.Result
    # try to get 2D slice
    if res.ndim == 1:
        # flatten: create a dummy 2D grid
        grid = np.zeros((h+1, w+1), dtype=np.float64)
    elif res.ndim >= 2:
        # assume shape (h+1, w+1, ...)
        if res.shape[0] == h+1 and res.shape[1] == w+1:
            grid = res
        elif res.shape[0] == w+1 and res.shape[1] == h+1:
            # transpose
            grid = res.T
        else:
            # fallback: reshape first two dims
            grid = res.reshape((h+1, w+1))
    else:
        grid = res.reshape((h+1, w+1))

    # compute gmax using log(1 + val)
    vals = np.log(1.0 + grid[:h, :w])
    gmax = np.max(vals) if np.max(vals) != 0 else 1.0

    # Create gran image upscaled by factor 8 (height = h*8, width = w*8)
    gran_h = h * 8
    gran_w = w * 8
    gran = np.zeros((gran_h, gran_w), dtype=np.uint8)

    for y in range(h):
        for x in range(w):
            v = np.log(1.0 + grid[y, x])
            px = int((v / gmax) * 255.0) if gmax > 0 else 0
            # fill 8x8 block
            gran[y*8:(y+1)*8, x*8:(x+1)*8] = px

    info.GranPGM = gran


def GUICreateGranulometryImage(info:GUIInfo):
    info.GranPGM = None
    try:
        GUIInitGranulometryImage(info)
    except Exception as e:
        print("Error creating granulometry image:", e)
        info.GranPGM = None


# ---------------------------
# PyQt Window classes
# ---------------------------

def numpy_to_qimage(gray: np.ndarray) -> QImage:
    """Convert a 2D uint8 numpy array to QImage Format_Indexed8"""
    if gray is None:
        return QImage()
    h, w = gray.shape
    # Ensure contiguous
    arr = np.ascontiguousarray(gray, dtype=np.uint8)
    img = QImage(arr.data, w, h, arr.strides[0], QImage.Format_Grayscale8)
    return img.copy()  # copy to detach from numpy memory


class DiatomWindow(QMainWindow):
    def __init__(self, guiinfo:GUIInfo):
        super().__init__()
        self.guiinfo = guiinfo
        self.setWindowTitle("Diatom window")
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)
        self.setCentralWidget(self.label)
        self.redraw()

    def redraw(self):
        img = numpy_to_qimage(self.guiinfo.DiatomPGM)
        pix = QPixmap.fromImage(img)
        self.label.setPixmap(pix)
        self.resize(pix.size())

    # Optional: override leaveEvent to mimic XStoreName behavior
    def leaveEvent(self, event):
        self.setWindowTitle("Diatom window")
        return super().leaveEvent(event)


class GranWindow(QMainWindow):
    """
    Granulometry window with interactive events:
     - mouse press -> compute selection (tile coords) and call Show2DSelGranNodes()
     - mouse move -> update title with coordinates & result value
     - mouse release -> clear pressed state
    The gran image uses a scale factor of 8 (same as C code).
    """
    def __init__(self, guiinfo:GUIInfo):
        super().__init__()
        self.guiinfo = guiinfo
        self.setWindowTitle("Granulometry window")
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)
        self.setCentralWidget(self.label)
        self.button_pressed = False
        self.prev_x = -1
        self.prev_y = -1
        self.outsidegran = True
        self.redraw()

    def redraw(self):
        if self.guiinfo.GranPGM is None:
            # blank white canvas
            w = self.guiinfo.Dims[0]*8
            h = self.guiinfo.Dims[1]*8
            arr = np.full((h, w), 255, dtype=np.uint8)
            img = numpy_to_qimage(arr)
        else:
            img = numpy_to_qimage(self.guiinfo.GranPGM)
        pix = QPixmap.fromImage(img)
        self.label.setPixmap(pix)
        self.resize(pix.size())

    def mousePressEvent(self, event):
        x = event.position().x() if hasattr(event, 'position') else event.x()
        y = event.position().y() if hasattr(event, 'position') else event.y()
        tx = int(x)//8
        ty = int(y)//8
        if 0 <= tx < self.guiinfo.Dims[0] and 0 <= ty < self.guiinfo.Dims[1]:
            self.button_pressed = True
            self.prev_x = tx
            self.prev_y = ty
            self.outsidegran = False
            # emulate refresh and selection
            self.guiinfo.DiatomWindow.redraw()
            self.redraw()
            Show2DSelGranNodes(self.guiinfo, 0, 1, tx, ty)  # default d1=0,d2=1
            TLImageSelectPixel(self, tx, ty)
            self.redraw()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        # emulate re-draw in diatom window
        self.guiinfo.DiatomWindow.redraw()
        self.button_pressed = False
        self.prev_x = -1
        self.prev_y = -1
        super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event):
        x = event.position().x() if hasattr(event, 'position') else event.x()
        y = event.position().y() if hasattr(event, 'position') else event.y()
        tx = int(x)//8
        ty = int(y)//8
        if 0 <= tx < self.guiinfo.Dims[0] and 0 <= ty < self.guiinfo.Dims[1]:
            self.outsidegran = False
            if (tx != self.prev_x) or (ty != self.prev_y):
                self.prev_x = tx; self.prev_y = ty
                # Acquire value from Result: addressing as in C: Result[y*(Dims[0]+1)+x]
                # Our Result is a numpy array shaped (h+1, w+1, ...)
                val = 0.0
                try:
                    val = float(self.guiinfo.Result[ty, tx])
                except Exception:
                    # fallback if different shape
                    val = float(self.guiinfo.Result.flat[0])
                self.setWindowTitle(f"Granulometry ({tx}, {ty}, {val:1.0f})")
                if self.button_pressed:
                    Show2DSelGranNodes(self.guiinfo, 0, 1, tx, ty)
                    TLImageSelectPixel(self, tx, ty)
                    self.redraw()
        else:
            if not self.outsidegran:
                self.outsidegran = True
                self.setWindowTitle("Granulometry window")
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        self.outsidegran = True
        self.setWindowTitle("Granulometry window")
        return super().leaveEvent(event)


# ---------------------------
# Functions that were used by the C GUI
# ---------------------------

def TLImageSelectPixel(window:GranWindow, x:int, y:int):
    """
    In C this colors the 8x8 block with the Display->Red pixel.
    Here we simply draw a red rectangle overlay in the GranPGM for visualization.
    """
    g = window.guiinfo.GranPGM
    if g is None:
        return
    h, w = g.shape
    # draw red selection by composing a color overlay on a QImage later.
    # For simplicity, set selection area to 255 (white) in this grayscale demo
    sx = x*8; sy = y*8
    if 0 <= sx < w and 0 <= sy < h:
        # set to 0 (dark) to indicate selection (in grayscale demo)
        window.guiinfo.GranPGM[sy:sy+8, sx:sx+8] = 0


def Show2DSelGranNodes(info:GUIInfo, d1:int, d2:int, x:int, y:int):
    """
    Emulate the original Show2DSelGranNodes: depending on the filter, run a
    corresponding GUIFilter and then color the Diatom image with overlays.
    For now, we call the filter functions (placeholders) and then color pixels in
    DiatomPGM to visualize result.
    """
    # build selcoord or pass x,y for min filter
    if info.Filter == 0:
        GUIFilterMDMin(info.Tree, info.NumAttrs, info.Procs, info.Dims, info.DLs, info.DHs, d1, d2, x, y)
    elif info.Filter == 1:
        selcoord = [0]*info.NumAttrs
        selcoord[d1] = x
        selcoord[d2] = y
        GUIFilterMDDirect(info.Tree, info.NumAttrs, info.Procs, info.Dims, info.DLs, info.DHs, selcoord)
    elif info.Filter == 2:
        selcoord = [0]*info.NumAttrs
        selcoord[d1] = x
        selcoord[d2] = y
        GUIFilterMDMax(info.Tree, info.NumAttrs, info.Procs, info.Dims, info.DLs, info.DHs, selcoord)
    else:
        # For filter==3 or others, call a placeholder
        print("[DEBUG] GUIFilterMDSub not implemented; skipping")

    # After filtering, color diatom pixels according to node statuses
    # We'll simply color diatom in-memory array (DiatomPGM) by setting special values:
    H = info.DiatomHeight; W = info.DiatomWidth
    diatom = info.DiatomPGM
    # For demo, we will darken pixels whose corresponding node had NodeStatus 1 (match) or 2 (parent)
    # Note: in real code, mapping from pixel->node requires GetNodeIndex; here we will apply a synthetic mapping.
    # We'll just visualize the selection by drawing a rectangle centered at x,y scaled to image dims.
    cx = int((x / max(1, info.Dims[0]-1)) * (W-1)) if info.Dims[0]>1 else 0
    cy = int((y / max(1, info.Dims[1]-1)) * (H-1)) if info.Dims[1]>1 else 0
    r = max(1, min(W, H)//20)
    for yy in range(max(0, cy-r), min(H, cy+r)):
        for xx in range(max(0, cx-r), min(W, cx+r)):
            # darken pixel for selection visualization
            diatom[yy, xx] = max(0, diatom[yy, xx] - 100)


# ---------------------------
# Implementations (placeholders) of filter functions ported from C logic
# Note: these are simplified and assume tree.nodes is a flat list and parent is index
# ---------------------------

def GUIFilterMDMin(tree:MaxTree, numattrs:int, procs:List[Proc], dims:List[int], dls:List[float], dhs:List[float], d1:int, d2:int, x:int, y:int):
    """
    Simplified translation of GUIFilterMDMin. Marks node.NodeStatus = 1 if mapping matches (x,y),
    and NodeStatus = 2 if parent had NodeStatus set.
    """
    NUMLEVELS = len(tree.nodes_per_level)
    for l in range(NUMLEVELS):
        for i in range(tree.get_number_of_nodes(l)):
            idx = tree.get_num_pixels_below_level(l) + i
            node = tree.nodes[idx]
            parent_idx = node.parent
            if idx != parent_idx:
                if tree.nodes[parent_idx].NodeStatus:
                    node.NodeStatus = 2
                attrs = node.attributes
                lm1 = procs[d1].Mapper(procs[d1].Attribute(attrs[d1]), dims[d1], dls[d1], dhs[d1])
                lm2 = procs[d2].Mapper(procs[d2].Attribute(attrs[d2]), dims[d2], dls[d2], dhs[d2])
                if (lm1 == x) and (lm2 == y):
                    node.NodeStatus = 1


def GUIFilterMDDirect(tree:MaxTree, numattrs:int, procs:List[Proc], dims:List[int], dls:List[float], dhs:List[float], selcoord:List[int]):
    """
    Simplified (interpreted) translation of GUIFilterMDDirect.
    This implementation is a simplified adaptation to demonstrate behavior.
    """
    NUMLEVELS = len(tree.nodes_per_level)
    # First pass
    for l in range(NUMLEVELS):
        for i in range(tree.get_number_of_nodes(l)):
            idx = tree.get_num_pixels_below_level(l) + i
            node = tree.nodes[idx]
            if idx != node.parent:
                node.NodeStatus = node.Area
                attrs = node.attributes
                pos = 0
                selpos = 0
                for j in reversed(range(numattrs)):
                    mm = procs[j].Mapper(procs[j].Attribute(attrs[j]), dims[j], dls[j], dhs[j])
                    pos = pos * (dims[j] + 1) + mm
                    selpos = selpos * (dims[j] + 1) + selcoord[j]
                if pos == selpos:
                    node.NodeStatus = -node.NodeStatus

    # Note: second pass (propagation upward) is complex; here we implement a simplified variant:
    for l in range(NUMLEVELS-1, -1, -1):
        for i in range(tree.get_number_of_nodes(l)):
            idx = tree.get_num_pixels_below_level(l) + i
            node = tree.nodes[idx]
            parent = tree.nodes[node.parent]
            # update parent's Pos from this node's attributes
            for j in reversed(range(numattrs)):
                parent.Pos[j] = procs[j].Mapper(procs[j].Attribute(node.attributes[j]), dims[j], dls[j], dhs[j])
            # simplified climb, not exact algorithm
            # (Detailed implementation should mirror the original C while loop)

    # Finalize
    for l in range(NUMLEVELS):
        for i in range(tree.get_number_of_nodes(l)):
            idx = tree.get_num_pixels_below_level(l) + i
            node = tree.nodes[idx]
            node.NodeStatus = 1 if node.NodeStatus < 0 else 0


def GUIFilterMDMax(tree:MaxTree, numattrs:int, procs:List[Proc], dims:List[int], dls:List[float], dhs:List[float], selcoord:List[int]):
    NUMLEVELS = len(tree.nodes_per_level)
    for l in range(NUMLEVELS-1, -1, -1):
        for i in range(tree.get_number_of_nodes(l)):
            idx = tree.get_num_pixels_below_level(l) + i
            node = tree.nodes[idx]
            parent = tree.nodes[node.parent]
            if idx != node.parent:
                attrs = node.attributes
                pos = 0
                selpos = 0
                for j in reversed(range(numattrs)):
                    lm = procs[j].Mapper(procs[j].Attribute(attrs[j]), dims[j], dls[j], dhs[j])
                    if node.NodeStatus:
                        lm = max(lm, node.Pos[j])
                    if parent.NodeStatus:
                        parent.Pos[j] = max(parent.Pos[j], lm)
                    else:
                        parent.Pos[j] = lm
                    pos = pos * (dims[j] + 1) + lm
                    selpos = selpos * (dims[j] + 1) + selcoord[j]
                node.NodeStatus = 1 if pos == selpos else 0
                parent.NodeStatus = 1


# ---------------------------
# Control window: basic controls similar to original GUI
# ---------------------------

class ControlWindow(QMainWindow):
    def __init__(self, guiinfo:GUIInfo, on_update_callback):
        super().__init__()
        self.guiinfo = guiinfo
        self.on_update = on_update_callback
        self.setWindowTitle("Control")
        self._init_ui()

    def _init_ui(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        form = QFormLayout()
        # Decision (filter)
        self.decision_box = QComboBox()
        self.decision_box.addItems(["MDMin (0)", "MDDirect (1)", "MDMax (2)", "WilkinsonK (3)"])
        self.decision_box.setCurrentIndex(self.guiinfo.Filter)
        form.addRow("Decision:", self.decision_box)

        # Attr combo placeholders
        self.attr1_spin = QSpinBox()
        self.attr1_spin.setMinimum(-1); self.attr1_spin.setMaximum(10)
        self.attr2_spin = QSpinBox()
        self.attr2_spin.setMinimum(-1); self.attr2_spin.setMaximum(10)
        form.addRow("Attr1:", self.attr1_spin)
        form.addRow("Attr2:", self.attr2_spin)

        # Width/Height
        self.width_spin = QSpinBox(); self.width_spin.setMinimum(1); self.width_spin.setValue(self.guiinfo.Dims[0])
        self.height_spin = QSpinBox(); self.height_spin.setMinimum(1); self.height_spin.setValue(self.guiinfo.Dims[1])
        form.addRow("Width:", self.width_spin)
        form.addRow("Height:", self.height_spin)

        # Min/Max floats
        self.min1 = QDoubleSpinBox(); self.min1.setValue(self.guiinfo.DLs[0]); self.min1.setDecimals(4)
        self.max1 = QDoubleSpinBox(); self.max1.setValue(self.guiinfo.DHs[0]); self.max1.setDecimals(4)
        form.addRow("Min1:", self.min1); form.addRow("Max1:", self.max1)

        layout.addLayout(form)

        # Buttons row
        btn_layout = QHBoxLayout()
        self.gran_btn = QPushButton("Granulometry")
        self.filter_btn = QPushButton("Filter")
        self.invert_btn = QPushButton("Invert")
        self.noise_btn = QPushButton("Noise")
        self.about_btn = QPushButton("About")
        self.quit_btn = QPushButton("Quit")
        btn_layout.addWidget(self.gran_btn); btn_layout.addWidget(self.filter_btn)
        btn_layout.addWidget(self.invert_btn); btn_layout.addWidget(self.noise_btn)
        btn_layout.addWidget(self.about_btn); btn_layout.addWidget(self.quit_btn)
        layout.addLayout(btn_layout)

        # Connect signals
        self.gran_btn.clicked.connect(self.on_gran)
        self.filter_btn.clicked.connect(self.on_filter)
        self.invert_btn.clicked.connect(self.on_invert)
        self.noise_btn.clicked.connect(self.on_noise)
        self.about_btn.clicked.connect(self.on_about)
        self.quit_btn.clicked.connect(self.on_quit)

        self.setCentralWidget(widget)

    def on_gran(self):
        # recompute granulometry with current params
        self.guiinfo.Dims[0] = self.width_spin.value()
        self.guiinfo.Dims[1] = self.height_spin.value()
        self.guiinfo.Filter = self.decision_box.currentIndex()
        # reallocate
        self.guiinfo.Result = gui_alloc_result_array(self.guiinfo.NumAttrs, self.guiinfo.Dims)
        GUIComputeGranulometry(self.guiinfo)
        GUICreateGranulometryImage(self.guiinfo)
        # refresh windows
        self.on_update()

    def on_filter(self):
        self.guiinfo.Filter = self.decision_box.currentIndex()
        print("[DEBUG] Filter set to", self.guiinfo.Filter)
        self.on_update()

    def on_invert(self):
        # invert DiatomPGM
        self.guiinfo.DiatomPGM = 255 - self.guiinfo.DiatomPGM
        # NOTE: in C they recompute the MaxTree and granulometry; here we skip that heavy step
        GUIComputeGranulometry(self.guiinfo)
        GUICreateGranulometryImage(self.guiinfo)
        self.on_update()

    def on_noise(self):
        print("[TODO] Noise dialog not implemented")

    def on_about(self):
        print("xmaxtree - Python port (partial) - October 2025")

    def on_quit(self):
        QApplication.instance().quit()


# ---------------------------
# Top-level launcher function GUIShow (mirrors C GUIShow)
# ---------------------------

def GUIShow(img:Optional[np.ndarray],
            shape:Optional[np.ndarray],
            width:int, height:int,
            tree:Optional[MaxTree],
            filter_mode:int, k:int, mingreylevel:int,
            numattrs:int, procs:List[Proc], dims:List[int],
            dls:List[float], dhs:List[float], sharedcm:bool=False):
    """
    Launch the GUI: create GUIInfo, open windows, and run event loop.
    """
    # Ensure QApplication is created first
    app = QApplication.instance() or QApplication(sys.argv)

    guiinfo = GUIInfo(img, shape, width, height, tree or MaxTree(), filter_mode, k,
                      mingreylevel, numattrs, procs, dims, dls, dhs, sharedcm)

    # create windows
    control = ControlWindow(guiinfo, on_update_callback=lambda: (diatom.redraw(), gran.redraw()))
    diatom = DiatomWindow(guiinfo)
    gran = GranWindow(guiinfo)

    # store references for use in functions (GUIInfo expects these to exist in some calls)
    guiinfo.ControlWindow = control
    guiinfo.DiatomWindow = diatom
    guiinfo.GranWindow = gran

    # show windows
    control.show()
    diatom.show()
    gran.show()

    # Run event loop
    sys.exit(app.exec_())


# ---------------------------
# Demo / main
# ---------------------------

def demo_create_dummy_tree(num_levels=3, nodes_per_level=None, num_attrs=2):
    if nodes_per_level is None:
        nodes_per_level = [4]*num_levels
    nodes = []
    total = sum(nodes_per_level)
    # simple linear parent: node i parent -> max(0, i-1)
    for i in range(total):
        parent = i-1 if i>0 else 0
        attrs = [float((i % (j+3)) + 0.1) for j in range(num_attrs)]
        nodes.append(Node(parent=parent, attributes=attrs, area=1, level=float(i%num_levels)))
    return MaxTree(nodes=nodes, nodes_per_level=nodes_per_level)


if __name__ == "__main__":
    # create dummy diatom image (512x512) and shape mask
    W, H = 256, 256
    diatom = np.random.randint(0, 255, (H, W), dtype=np.uint8)
    shape = (diatom > 10).astype(np.uint8)

    # Create dummy max tree and procs
    tree = demo_create_dummy_tree(num_levels=3, nodes_per_level=[10, 10, 10], num_attrs=2)
    # simple attribute and mapper functions
    def attr_func(v): return float(v)
    def mapper_func(value, dim, dl, dh):
        # map continuous value into dim bins
        if dim <= 0: return 0
        # normalize between dl..dh then bin
        if dh == dl: return 0
        t = (value - dl) / (dh := (dh - dl))
        t = max(0.0, min(0.9999, t))
        return int(t * dim)
    procs = [Proc(attribute_func=attr_func, mapper_func=mapper_func),
             Proc(attribute_func=attr_func, mapper_func=mapper_func)]

    dims = [16, 16]
    dls = [0.0, 0.0]
    dhs = [2.0, 2.0]

    GUIShow(img=diatom, shape=shape, width=W, height=H,
            tree=tree, filter_mode=0, k=5, mingreylevel=0, numattrs=2,
            procs=procs, dims=dims, dls=dls, dhs=dhs, sharedcm=False)
