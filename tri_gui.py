from PyQt5 import QtCore, QtWidgets, QtGui
import sys
import numpy as np
from scipy.spatial.qhull import Delaunay
from PIL import Image
from easytime import Timer
import os.path
import svgwrite


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

        self.show_picture = QtWidgets.QCheckBox('Show picture')
        self.show_picture.setChecked(self.master_widget.show_picture)
        self.show_picture.stateChanged.connect(self.on_setting_changed)

        buttons_layout.addWidget(self.show_picture)

        self.fill_triangles = QtWidgets.QCheckBox('Fill triangles')
        self.fill_triangles.setChecked(self.master_widget.fill_triangles)
        self.fill_triangles.stateChanged.connect(self.on_setting_changed)

        buttons_layout.addWidget(self.fill_triangles)

        buttons_layout.addStretch(10)

        triangulate = QtWidgets.QPushButton('Triangulate!')
        triangulate.clicked.connect(self.master_widget.triangulate)
        buttons_layout.addWidget(triangulate)

        main_layout.addWidget(self.master_widget)

        self.resize(800, 600)

        self.show()

    def on_setting_changed(self):
        self.master_widget.brush_radius = int(self.brush_radius.value())
        self.master_widget.brush_value = int(self.brush_value.value())
        self.master_widget.show_picture = self.show_picture.isChecked()
        self.master_widget.fill_triangles = self.fill_triangles.isChecked()
        self.master_widget.repaint()

    def on_load_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open image', filter='Image Files (*.png *.jpg *.jpeg *.bmp);; All Files (*)')
        if os.path.isfile(path):
            self.master_widget.load_image(path)

    def on_save_svg(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save SVG', filter='SVG (*.svg);; All Files (*)')
        if path:
            self.master_widget.save_svg(path)


def weighted_poisson_disc_sampling(rho_arr: np.ndarray, global_seed_throws=100, bridson_k=30):
    grid = np.ones(rho_arr.shape, dtype=int) * -1
    coordinate_grid = np.zeros((*grid.shape, 2))  # accelerator structure for circle checks
    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            coordinate_grid[y,x,0] = x
            coordinate_grid[y,x,1] = y
    samples = []

    def density_to_radius(rho):
        if rho == 0:
            r = np.inf
        else:
            r = np.sqrt(1 / (2 * np.sqrt(3) * rho))

        return r

    def is_viable(p: np.ndarray):
        p_int = p.astype(int)
        if p_int[0] < 0 or p_int[0] > grid.shape[1] - 1 or p_int[1] < 0 or p_int[1] > grid.shape[0] - 1:
            return False

        rho = rho_arr[p_int[1], p_int[0]]
        if rho == 0:
            return False

        r = density_to_radius(rho)
        ymin = max(0, p_int[1] - int(r))
        ymax = min(grid.shape[0] - 1, p_int[1] + int(r))
        xmin = max(0, p_int[0] - int(r))
        xmax = min(grid.shape[1] - 1, p_int[0] + int(r))

        in_circle = np.sum(np.power(coordinate_grid[ymin:ymax, xmin:xmax] - p, 2), axis=2) < r**2
        has_samples = grid[ymin:ymax, xmin:xmax] >= 0

        not_viable = np.any(np.logical_and(in_circle, has_samples))

        return not not_viable

    def add_sample(p: np.ndarray):
        samples.append(p)
        index = len(samples) - 1
        grid[int(p[1]), int(p[0])] = index

        return index

    done = False
    while not done:
        active_list = []
        # find a viable seed sample by global dart throwing
        for i in range(global_seed_throws):
            sample = np.array([np.random.rand() * grid.shape[1], np.random.rand() * grid.shape[0]])
            if is_viable(sample):
                sample_index = add_sample(sample)
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
                if is_viable(candidate):
                    candidate_index = add_sample(candidate)
                    active_list.append(candidate_index)
            else:
                active_list.pop(pi)

    return np.array(samples)


class TriMeMasterWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.show_picture = True
        self.fill_triangles = True

        self.brush_value = 50
        self.brush_radius = 50

        self.density_orders = 5

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
        self.ps = weighted_poisson_disc_sampling(rho_arr, bridson_k=10)

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

        self.repaint()

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
        if self.show_picture:
            painter.drawPixmap(0, 0, self.img)
            painter.setOpacity(0.5)  # draw density map semi transparent
        density_img = QtGui.QImage(self.density_array.tobytes(), iw, ih, QtGui.QImage.Format_Grayscale8)
        painter.drawImage(0, 0, density_img)
        painter.restore()

        painter.save()
        red = QtGui.QColor('red')
        painter.setBrush(red)
        painter.setPen(red)
        for i in range(self.ps.shape[0]):
            cx = int(self.ps[i, 0] * cw / iw)
            cy = int(self.ps[i, 1] * ch / ih)
            r = 10
            #painter.drawEllipse(cx - r, cy - r, 2*r, 2*r)
        painter.restore()

        painter.save()
        blue = QtGui.QColor('blue')
        for i in range(self.tri_indices.shape[0]):
            tri_points = self.ps[self.tri_indices[i]] * cw / iw
            poly = QtGui.QPolygonF([QtCore.QPointF(*tri_point) for tri_point in tri_points])
            color = QtGui.QColor(*self.tri_colors[i])
            if self.fill_triangles:
                painter.setPen(QtCore.Qt.NoPen)
                painter.setBrush(color)
            else:
                painter.setPen(blue)
            painter.drawPolygon(poly)
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

        ix = np.clip(int((mouse_x - (ww - cw) / 2) * iw / cw), 0, iw - 1)
        iy = np.clip(int((mouse_y - (wh - ch) / 2) * ih / ch), 0, ih - 1)

        brush_color = int((100 - self.brush_value) * 2.55)

        for x in range(ix - self.brush_radius, ix + self.brush_radius):
            for y in range(iy - self.brush_radius, iy + self.brush_radius):
                if x < 0 or x >= iw or y < 0 or y >= ih:
                    continue

                if (x - ix)**2 + (y - iy)**2 < self.brush_radius**2:
                    self.density_array[y, x] = brush_color

        self.repaint()

    def mousePressEvent(self, e: QtGui.QMouseEvent):
        if e.button() == QtCore.Qt.LeftButton:
            self.draw(e.x(), e.y())

    def mouseMoveEvent(self, e: QtGui.QMouseEvent):
        if e.buttons() & QtCore.Qt.LeftButton:
            self.draw(e.x(), e.y())


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = TriMeWindow()
    result = app.exec_()