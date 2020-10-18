from PyQt5 import QtCore, QtWidgets, QtGui
import sys
import numpy as np
from scipy.spatial.qhull import Delaunay


class TriMeWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent)

        self.setWindowTitle('Triangulate Me!')

        self.master_widget = TriMeMasterWidget()

        main_frame = QtWidgets.QFrame()
        self.setCentralWidget(main_frame)
        main_layout = QtWidgets.QVBoxLayout()
        main_frame.setLayout(main_layout)

        buttons_layout = QtWidgets.QHBoxLayout()
        main_layout.addLayout(buttons_layout)

        self.brush_value = QtWidgets.QDoubleSpinBox()
        self.brush_value.setRange(0, 1)
        self.brush_value.setValue(self.master_widget.brush_color / 255)
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

        self.max_density = QtWidgets.QDoubleSpinBox()
        self.max_density.setRange(0.01, 100)
        self.max_density.setValue(self.master_widget.rho_max)
        self.max_density.valueChanged.connect(self.on_setting_changed)

        buttons_layout.addWidget(QtWidgets.QLabel('Density:'))
        buttons_layout.addWidget(self.max_density)

        self.points = QtWidgets.QDoubleSpinBox()
        self.points.setRange(0, 1000)
        self.points.setDecimals(0)
        self.points.setValue(self.master_widget.N)
        self.points.valueChanged.connect(self.on_setting_changed)

        buttons_layout.addWidget(QtWidgets.QLabel('Points:'))
        buttons_layout.addWidget(self.points)

        self.show_picture = QtWidgets.QCheckBox('Show picture')
        self.show_picture.setChecked(self.master_widget.show_picture)
        self.show_picture.stateChanged.connect(self.on_setting_changed)

        buttons_layout.addWidget(self.show_picture)

        buttons_layout.addStretch(10)

        triangulate = QtWidgets.QPushButton('Triangulate!')
        triangulate.clicked.connect(self.master_widget.triangulate)
        buttons_layout.addWidget(triangulate)

        main_layout.addWidget(self.master_widget)

        self.resize(800, 600)

        self.show()

    def on_setting_changed(self):
        self.master_widget.brush_radius = int(self.brush_radius.value())
        self.master_widget.brush_color = int(self.brush_value.value() * 255)
        self.master_widget.rho_max = self.max_density.value()
        self.master_widget.N = int(self.points.value())
        self.master_widget.show_picture = self.show_picture.isChecked()
        self.master_widget.repaint()


class TriMeMasterWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.img = QtGui.QPixmap('danny.jpg')
        # Default: everything white (no tris)
        self.density_array = np.ones((self.img.height(), self.img.width()), dtype=np.ubyte) * 255

        self.show_picture = True

        self.brush_color = 0
        self.brush_radius = 50

        self.rho_max = 1
        self.N = 10

        self.ps = np.zeros((0, 2))
        self.tri_indices = np.zeros((0, 3))

    def triangulate(self):
        self.ps = np.ones((self.N, 2)) * 1e6  # initialize coordinates to (1e6, 1e6)
        nr_points = 0

        same_point_iterations = 0
        while nr_points < self.N and same_point_iterations < 1e4:
            same_point_iterations += 1
            p = np.array([np.random.rand() * self.density_array.shape[1],
                          np.random.rand() * self.density_array.shape[0]])
            img_row = int(p[1])
            img_col = int(p[0])
            a = (255 - self.density_array[img_row, img_col])/255  # 0 -> rho_min, 1 -> rho_max
            rho = a * self.rho_max / 1e4  # rho max is scaled for input

            if rho == 0:
                r = 10000
            else:
                r = np.sqrt(1 / (2 * np.sqrt(3) * rho))

            has_rmin_to_all = all(np.sum(np.power(self.ps - p, 2), axis=1) > r ** 2)
            if has_rmin_to_all and rho > 0:
                self.ps[nr_points, :] = p
                nr_points += 1
                print(nr_points, "/", self.N)
                same_point_iterations = 0

        tria = Delaunay(self.ps)
        self.tri_indices = tria.simplices

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
            painter.drawEllipse(cx - r, cy - r, 2*r, 2*r)
        painter.restore()

        painter.save()
        blue = QtGui.QColor('blue')
        painter.setPen(blue)
        for i in range(self.tri_indices.shape[0]):
            tri_points = self.ps[self.tri_indices[i]] * cw / iw
            poly = QtGui.QPolygonF([QtCore.QPointF(*tri_point) for tri_point in tri_points])
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

        for x in range(ix - self.brush_radius, ix + self.brush_radius):
            for y in range(iy - self.brush_radius, iy + self.brush_radius):
                if x < 0 or x >= iw or y < 0 or y >= ih:
                    continue

                if (x - ix)**2 + (y - iy)**2 < self.brush_radius**2:
                    self.density_array[y, x] = self.brush_color

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