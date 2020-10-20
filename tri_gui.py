from enum import Enum
from PyQt5 import QtCore, QtWidgets, QtGui
import sys
import numpy as np
from scipy.spatial.qhull import Delaunay
from PIL import Image
from easytime import Timer
import os.path
import svgwrite
from numba import jit


class TriMeWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent)

        self.setWindowTitle('Triangulate Me!')
        self.setWindowIcon(QtGui.QIcon('icon.png'))

        self.master_widget = TriMeMasterWidget()

        main_frame = QtWidgets.QFrame()
        self.setCentralWidget(main_frame)
        main_layout = QtWidgets.QVBoxLayout()
        main_frame.setLayout(main_layout)

        buttons_layout = QtWidgets.QHBoxLayout()
        main_layout.addLayout(buttons_layout)

        load_image = QtWidgets.QPushButton('Load image')
        load_image.clicked.connect(self.on_load_image)

        buttons_layout.addWidget(load_image)

        save_svg = QtWidgets.QPushButton('Save SVG')
        save_svg.clicked.connect(self.on_save_svg)

        buttons_layout.addWidget(save_svg)

        self.brush_value = QtWidgets.QDoubleSpinBox()
        self.brush_value.setRange(0, 100)
        self.brush_value.setDecimals(0)
        self.brush_value.setValue(self.master_widget.brush_value)
        self.brush_value.valueChanged.connect(self.on_setting_changed)

        buttons_layout.addWidget(QtWidgets.QLabel('Value:'))
        buttons_layout.addWidget(self.brush_value)

        self.brush_radius = QtWidgets.QDoubleSpinBox()
        self.brush_radius.setRange(1, 100)
        self.brush_radius.setDecimals(0)
        self.brush_radius.setValue(self.master_widget.brush_radius)
        self.brush_radius.valueChanged.connect(self.on_setting_changed)

        buttons_layout.addWidget(QtWidgets.QLabel('Radius:'))
        buttons_layout.addWidget(self.brush_radius)

        self.mode_draw = QtWidgets.QRadioButton('Draw')
        self.mode_draw.setChecked(self.master_widget.interact_mode == TriMeMasterWidget.InteractMode.DENSITY_DRAW)
        self.mode_draw.toggled.connect(self.on_setting_changed)
        buttons_layout.addWidget(self.mode_draw)

        self.mode_manual = QtWidgets.QRadioButton('Manual')
        self.mode_manual.setChecked(self.master_widget.interact_mode == TriMeMasterWidget.InteractMode.MANUAL_POINTS)
        self.mode_manual.toggled.connect(self.on_setting_changed)
        buttons_layout.addWidget(self.mode_manual)

        buttons_layout.addStretch(10)

        layers_layout = QtWidgets.QHBoxLayout()
        main_layout.addLayout(layers_layout)

        layers_layout.addWidget(QtWidgets.QLabel('Layers:'))

        self.layer_image = QtWidgets.QCheckBox('Image')
        self.layer_image.setChecked(self.master_widget.layer_image)
        self.layer_image.toggled.connect(self.on_setting_changed)
        layers_layout.addWidget(self.layer_image)

        self.layer_density = QtWidgets.QCheckBox('Density')
        self.layer_density.setChecked(self.master_widget.layer_density)
        self.layer_density.toggled.connect(self.on_setting_changed)
        layers_layout.addWidget(self.layer_density)

        self.layer_tri_outline = QtWidgets.QCheckBox('Triangle Outline')
        self.layer_tri_outline.setChecked(self.master_widget.layer_tri_outline)
        self.layer_tri_outline.toggled.connect(self.on_setting_changed)
        layers_layout.addWidget(self.layer_tri_outline)

        self.layer_tri_fill = QtWidgets.QCheckBox('Triangle Color')
        self.layer_tri_fill.setChecked(self.master_widget.layer_tri_fill)
        self.layer_tri_fill.toggled.connect(self.on_setting_changed)
        layers_layout.addWidget(self.layer_tri_fill)

        self.layer_manual_points = QtWidgets.QCheckBox('Manual Points')
        self.layer_manual_points.setChecked(self.master_widget.layer_manual_points)
        self.layer_manual_points.toggled.connect(self.on_setting_changed)
        layers_layout.addWidget(self.layer_manual_points)

        layers_layout.addStretch(10)

        triangulate = QtWidgets.QPushButton('Triangulate!')
        triangulate.clicked.connect(self.master_widget.triangulate)
        buttons_layout.addWidget(triangulate)

        main_layout.addWidget(self.master_widget)

        self.resize(800, 600)

        self.show()

    def on_setting_changed(self):
        self.master_widget.brush_radius = int(self.brush_radius.value())
        self.master_widget.brush_value = int(self.brush_value.value())
        self.master_widget.layer_image = self.layer_image.isChecked()
        self.master_widget.layer_density = self.layer_density.isChecked()
        self.master_widget.layer_tri_outline = self.layer_tri_outline.isChecked()
        self.master_widget.layer_tri_fill = self.layer_tri_fill.isChecked()
        self.master_widget.layer_manual_points = self.layer_manual_points.isChecked()

        if self.mode_draw.isChecked():
            self.master_widget.interact_mode = TriMeMasterWidget.InteractMode.DENSITY_DRAW
        elif self.mode_manual.isChecked():
            self.master_widget.interact_mode = TriMeMasterWidget.InteractMode.MANUAL_POINTS

        self.master_widget.update()

    def on_load_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open image', filter='Image Files (*.png *.jpg *.jpeg *.bmp);; All Files (*)')
        if os.path.isfile(path):
            self.master_widget.load_image(path)

    def on_save_svg(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save SVG', filter='SVG (*.svg);; All Files (*)')
        if path:
            self.master_widget.save_svg(path)


@jit(nopython=True)
def density_to_radius(rho):
    if rho == 0:
        r = np.inf
    else:
        r = np.sqrt(1 / (2 * np.sqrt(3) * rho))

    return r


@jit(nopython=True)
def is_viable(grid, rho_arr, p: np.ndarray):
    p_int = p.astype(np.intc)
    if p_int[0] < 0 or p_int[0] > grid.shape[1] - 1 or p_int[1] < 0 or p_int[1] > grid.shape[0] - 1:
        return False

    rho = rho_arr[p_int[1], p_int[0]]
    if rho == 0:
        return False

    r = density_to_radius(rho)
    for y in range(max(0, p_int[1] - int(r)), min(grid.shape[0] - 1, p_int[1] + int(r))):
        for x in range(max(0, p_int[0] - int(r)), min(grid.shape[1] - 1, p_int[0] + int(r))):
            if (x - p[0])**2 + (y - p[1])**2 < r**2 and grid[y, x] >= 0:
                return False

    return True


@jit(nopython=True)
def weighted_poisson_disc_sampling(rho_arr: np.ndarray, initial_samples: np.ndarray, global_seed_throws=100, bridson_k=30):
    grid = np.ones(rho_arr.shape, dtype=np.intc) * -1
    samples = []

    for i, p in enumerate(initial_samples):
        samples.append([p[0], p[1]])
        grid[int(p[1]), int(p[0])] = i

    done = False
    while not done:
        active_list = []
        # find a viable seed sample by global dart throwing
        for i in range(global_seed_throws):
            sample = np.array([np.random.rand() * grid.shape[1], np.random.rand() * grid.shape[0]])
            if is_viable(grid, rho_arr, sample):
                samples.append([sample[0], sample[1]])
                sample_index = len(samples) - 1
                grid[int(sample[1]), int(sample[0])] = sample_index
                active_list.append(sample_index)
                break
        else:
            done = True

        if done:
            break

        # run Bridson's algorithm from seed
        while active_list:
            # pick random sample from active_list
            pi = int(np.random.rand()*len(active_list))
            p = samples[active_list[pi]]

            for i in range(bridson_k):
                # random candidate sample from annulum around p with [r, 2r]
                rho = rho_arr[int(p[1]), int(p[0])]
                r = density_to_radius(rho)
                R = r + np.random.rand() * (r)
                a = 2 * np.pi * np.random.rand()

                candidate = np.array([p[0] + R * np.cos(a), p[1] + R * np.sin(a)])
                if is_viable(grid, rho_arr, candidate):
                    samples.append([candidate[0], candidate[1]])
                    candidate_index = len(samples) - 1
                    grid[int(candidate[1]), int(candidate[0])] = candidate_index

                    active_list.append(candidate_index)
            else:
                active_list.pop(pi)

    return np.array(samples)


class TriMeMasterWidget(QtWidgets.QWidget):
    class InteractMode(Enum):
        DENSITY_DRAW = 0
        MANUAL_POINTS = 1

    def __init__(self):
        super().__init__()

        self.layer_image = True
        self.layer_density = True
        self.layer_tri_outline = False
        self.layer_tri_fill = True
        self.layer_manual_points = True

        self.interact_mode = self.InteractMode.DENSITY_DRAW

        self.brush_value = 50
        self.brush_radius = 50

        self.density_orders = 5

        self.manual_points = []
        self.load_image('danny.jpg')

    def load_image(self, path):
        self.img = QtGui.QPixmap(path)
        pil_img = Image.open(path)
        self.img_arr = np.array(pil_img)
        # Default: everything white (no tris)
        self.density_array = np.ones((self.img.height(), self.img.width()), dtype=np.ubyte) * 255

        self.ps = np.zeros((0, 2))
        self.tri_indices = np.zeros((0, 3))
        self.tri_colors = np.zeros((0, 3))

        self.update()

    def save_svg(self, path):
        # disable anti-aliasing to close gaps between tris
        svg = svgwrite.Drawing(path, (self.img_arr.shape[1], self.img_arr.shape[0]), shape_rendering='crispEdges')

        for i in range(self.tri_indices.shape[0]):
            tri_points = self.ps[self.tri_indices[i]]
            color_tuple = self.tri_colors[i]

            svg.add(svg.path([
                'M', *tri_points[0],
                'L', *tri_points[1],
                'L', *tri_points[2],
                'Z'
            ], fill=f"rgb({color_tuple[0]}, {color_tuple[1]}, {color_tuple[2]})",))

        svg.save()

    def triangulate(self):
        # --- estimate number of needed points based on density map
        rho_arr = (np.power(10, -self.density_orders*self.density_array/255) - 1e-5) / (1 - 1e-5)

        # poisson disc sampling
        self.ps = weighted_poisson_disc_sampling(rho_arr, np.array(self.manual_points).reshape((-1, 2)), bridson_k=10)

        # triangulation
        if self.ps.shape[0] >= 3:
            tria = Delaunay(self.ps)
            self.tri_indices = tria.simplices
        else:
            self.tri_indices = np.zeros((0, 3))

        # triangle coloring
        n_tris = self.tri_indices.shape[0]
        self.tri_colors = np.zeros((n_tris, 3), dtype=np.ubyte)

        barycentric_samples = np.array([
            [1, 1, 1],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0.5, 0.5],
            [0.5, 1, 0.5],
            [0.5, 0.5, 1]])

        for i in range(n_tris):
            tri_points = self.ps[self.tri_indices[i]]

            color_sum = np.zeros(3)

            for b_sample in barycentric_samples:
                sample = (b_sample @ tri_points / sum(b_sample)).astype(int)
                color_sum += np.power(self.img_arr[sample[1], sample[0]].astype(float), 2)

            # RMS instead of mean, supposedly more colour accurate
            avg_color = np.sqrt(color_sum / barycentric_samples.shape[0]).astype(np.ubyte)
            self.tri_colors[i] = avg_color

        self.update()

    def paintEvent(self, e: QtGui.QPaintEvent):
        painter = QtGui.QPainter(self)

        ww = self.width()
        wh = self.height()
        iw = self.img.width()
        ih = self.img.height()

        cw, ch = self.canvas_size()

        black = QtGui.QColor('black')
        painter.fillRect(0, 0, ww, wh, black)

        painter.translate((ww - cw) / 2, (wh - ch) / 2)

        painter.save()
        img_scale = cw / iw
        painter.scale(img_scale, img_scale)
        if self.layer_image:
            painter.drawPixmap(0, 0, self.img)
        if self.layer_density:
            if self.layer_image:
                painter.setOpacity(0.5)  # draw density map semi transparent
            density_img = QtGui.QImage(self.density_array.tobytes(), iw, ih, QtGui.QImage.Format_Grayscale8)
            painter.drawImage(0, 0, density_img)
        painter.restore()

        painter.save()
        blue = QtGui.QColor('blue')
        if self.layer_tri_outline or self.layer_tri_fill:
            if self.layer_tri_outline:
                painter.setPen(blue)
            else:
                painter.setPen(QtCore.Qt.NoPen)
            for i in range(self.tri_indices.shape[0]):
                tri_points = self.ps[self.tri_indices[i]] * cw / iw
                poly = QtGui.QPolygonF([QtCore.QPointF(*tri_point) for tri_point in tri_points])
                color = QtGui.QColor(*self.tri_colors[i])
                if self.layer_tri_fill:
                    painter.setBrush(color)
                painter.drawPolygon(poly)
        painter.restore()

        painter.save()
        red = QtGui.QColor('red')
        if self.layer_manual_points:
            painter.setBrush(red)
            painter.setPen(black)
            for p in self.manual_points:
                cx = int(p[0] * cw / iw)
                cy = int(p[1] * ch / ih)
                r = 4
                painter.drawEllipse(cx - r, cy - r, 2*r, 2*r)
        painter.restore()

    def canvas_size(self):
        iw = self.img.width()
        ih = self.img.height()
        iaspect = iw / ih

        ww = self.width()
        wh = self.height()
        waspect = ww / wh

        if waspect > iaspect:
            # black bars left and right
            ch = wh
            cw = int(ch * iaspect)
        else:
            # black bars top and bottom
            cw = ww
            ch = int(cw / iaspect)

        return cw, ch

    def draw(self, mouse_x, mouse_y):
        ww = self.width()
        wh = self.height()
        iw = self.img.width()
        ih = self.img.height()
        cw, ch = self.canvas_size()

        ix = int((mouse_x - (ww - cw) / 2) * iw / cw)
        iy = int((mouse_y - (wh - ch) / 2) * ih / ch)

        brush_color = int((100 - self.brush_value) * 2.55)

        for x in range(ix - self.brush_radius, ix + self.brush_radius):
            for y in range(iy - self.brush_radius, iy + self.brush_radius):
                if x < 0 or x >= iw or y < 0 or y >= ih:
                    continue

                if (x - ix)**2 + (y - iy)**2 < self.brush_radius**2:
                    self.density_array[y, x] = brush_color

        self.update()

    def place_point(self, mouse_x, mouse_y):
        ww = self.width()
        wh = self.height()
        iw = self.img.width()
        ih = self.img.height()
        cw, ch = self.canvas_size()

        ix = np.clip((mouse_x - (ww - cw) / 2) * iw / cw, 0, iw - 1)
        iy = np.clip((mouse_y - (wh - ch) / 2) * ih / ch, 0, ih - 1)

        self.manual_points.append([ix, iy])

        self.update()

    def delete_point(self, mouse_x, mouse_y):
        ww = self.width()
        wh = self.height()
        iw = self.img.width()
        ih = self.img.height()
        cw, ch = self.canvas_size()

        r = 10

        for p in self.manual_points:
            pwx = p[0] * cw/iw + (ww - cw)/2
            pwy = p[1] * ch/ih + (wh - ch)/2

            if (mouse_x - pwx)**2 + (mouse_y - pwy)**2 < r**2:
                self.manual_points.remove(p)

        self.update()

    def mousePressEvent(self, e: QtGui.QMouseEvent):
        if self.interact_mode == TriMeMasterWidget.InteractMode.DENSITY_DRAW:
            if e.button() == QtCore.Qt.LeftButton:
                self.draw(e.x(), e.y())
        elif self.interact_mode == TriMeMasterWidget.InteractMode.MANUAL_POINTS:
            if e.button() == QtCore.Qt.LeftButton:
                self.place_point(e.x(), e.y())
            elif e.button() == QtCore.Qt.RightButton:
                self.delete_point(e.x(), e.y())

    def mouseMoveEvent(self, e: QtGui.QMouseEvent):
        if self.interact_mode == TriMeMasterWidget.InteractMode.DENSITY_DRAW:
            if e.buttons() & QtCore.Qt.LeftButton:
                self.draw(e.x(), e.y())


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = TriMeWindow()
    result = app.exec_()