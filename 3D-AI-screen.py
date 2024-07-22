import PySide6.QtCore
import numpy as np
import cv2
import time
import onnxruntime
import torch
from math import ceil
from itertools import product as product

import sys
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from OpenGL.GL import *
from OpenGL.GLU import *

# WEBCAM native resolution
CAM_WIDTH = 1920
CAM_HEIGHT = 1080

# Pupil distance in m:
PUPIL_DISTANCE_M = 0.62

# Monitor dimensions in m
MONITOR_SCREEN_WIDTH = 0.623
MONITOR_SCREEN_HEIGHT = 0.343

# Webcam preview downscale factor
WEBCAM_PREVIEW_DOWNSCALING = 2


class VideoSource:
    def __init__(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(3, CAM_WIDTH)  # set Width
        self.cap.set(4, CAM_HEIGHT)  # set Height
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # Video compression

    def get_frame(self):
        ret, frame = self.cap.read()
        return frame

    def release(self):
        self.cap.release()


class PriorBox(object):
    # SOURCE: https://huggingface.co/amd/retinaface
    def __init__(self, cfg, image_size=None):
        super(PriorBox, self).__init__()
        self.min_sizes = cfg["min_sizes"]
        self.steps = cfg["steps"]
        self.clip = cfg["clip"]
        self.image_size = image_size
        self.feature_maps = [
            [ceil(self.image_size[0] / step), ceil(self.image_size[1] / step)]
            for step in self.steps
        ]

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]
        # back to torch land
        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output


class FaceDetectorAndLocalizer:
    def __init__(self):
        # Model source: https://huggingface.co/amd/retinaface/tree/main
        # Code has been significantly refactored for improved performance
        self.device = torch.device("cpu")
        self.ort = onnxruntime.InferenceSession('RetinaFace_int.onnx',
                                                providers=['VitisAIExecutionProvider'],
                                                provider_options=[{"config_file": 'vaip_config.json'}])
        self.cfg = {
            "name": "mobilenet0.25",
            "min_sizes": [[16, 32], [64, 128], [256, 512]],
            "steps": [8, 16, 32],
            "variance": [0.1, 0.2],
            "clip": False,
        }

        self.input_size = [608, 640]  # Model input size
        self.prior = PriorBox(self.cfg, self.input_size)
        self.priors = self.prior.forward()
        self.priors = self.priors.to(self.device)
        self.prior_data = self.priors.data

        # Rescaling parameters (needed as aspect ratio of captured image model input is different)
        self.scale_tensor = torch.Tensor([self.input_size[1], self.input_size[0], self.input_size[1], self.input_size[0]])
        self.scale_tensor = self.scale_tensor.to(self.device)

        self.scale_tensor_2 = torch.Tensor(
            [self.input_size[1], self.input_size[0],
             self.input_size[1], self.input_size[0],
             self.input_size[1], self.input_size[0],
             self.input_size[1], self.input_size[0],
             self.input_size[1], self.input_size[0],]
        )
        self.scale_tensor_2 = self.scale_tensor_2.to(self.device)

        ratio = self.input_size[0] * 1.0 / self.input_size[1]
        if CAM_HEIGHT * 1.0 / CAM_WIDTH <= ratio:
            self.resize_ratio = self.input_size[1] * 1.0 / CAM_WIDTH
            self.re_h, self.re_w = int(CAM_HEIGHT * self.resize_ratio), self.input_size[1]
        else:
            self.resize_ratio = self.input_size[0] * 1.0 / CAM_HEIGHT
            self.re_h, self.re_w= self.input_size[0], int(CAM_WIDTH * self.resize_ratio)

    def pad_image(self, image, h, w, size, pad_value):
        pad_image = image.copy()
        pad_h = max(size[0] - h, 0)
        pad_w = max(size[1] - w, 0)
        if pad_h > 0 or pad_w > 0:
            pad_image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=pad_value)
        return pad_image

    def decode(self, loc, prior, variances):
        """Decode locations from prediction using prior to undo
        the encoding we did for offset regression at train time.
        Args:
            loc (tensor): location prediction for loc layers,
                Shape: [4]
            prior (tensor): Prior box in center-offset form.
                Shape: [4].
            variances: (list[float]) Variances of priorboxes
        Return:
            decoded bounding box prediction
        """
        box = torch.cat(
            (
                prior[:2] + loc[:2] * variances[0] * prior[2:],
                prior[2:] * torch.exp(loc[2:] * variances[1]),
            ),
            0,
        )
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        return box

    def decode_landm(self, pre, prior, variances):
        """Decode landm from predictions using priors to undo
        the encoding we did for offset regression at train time.
        Args:
            pre (tensor): landm prediction for loc layers,
                Shape: [10]
            prior (tensor): Prior box in center-offset form.
                Shape: [4].
            variances: (list[float]) Variances of priorboxes
        Return:
            decoded landm prediction
        """
        landms = torch.cat(
            (
                prior[:2] + pre[:2] * variances[0] * prior[2:],
                prior[:2] + pre[2:4] * variances[0] * prior[2:],
                prior[:2] + pre[4:6] * variances[0] * prior[2:],
                prior[:2] + pre[6:8] * variances[0] * prior[2:],
                prior[:2] + pre[8:10] * variances[0] * prior[2:],
            ),
            dim=0,
        )
        return landms

    def detect_and_localize(self, full_size_frame):
        # start_ms = time.time()
        # Resize first, then convert to float. This saves ~5ms!
        img = cv2.resize(full_size_frame, (self.re_w, self.re_h))
        img = self.pad_image(img, self.re_h, self.re_w, self.input_size, (0.0, 0.0, 0.0))
        img = np.float32(img)

        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.numpy()
        # preprocess_ms = time.time()

        img = np.transpose(img, (0, 2, 3, 1))

        outputs = self.ort.run(None, {self.ort.get_inputs()[0].name: img})

        loc = torch.from_numpy(outputs[0])
        landms = torch.from_numpy(outputs[2])

        max_ind = np.argmax(outputs[1].squeeze(0)[:, 1])  # Place of maximum of scores, we only care about this
        loc = loc.squeeze(0)[max_ind, :]
        landms = landms.squeeze(0)[max_ind, :]

        box = self.decode(loc, self.prior_data[max_ind, :], self.cfg["variance"])
        box = box * self.scale_tensor / self.resize_ratio
        box = box.cpu().numpy()

        landms = self.decode_landm(landms, self.prior_data[max_ind, :], self.cfg["variance"])
        landms = landms * self.scale_tensor_2 / self.resize_ratio
        landms = landms.cpu().numpy()

        # end_ms = time.time()
        # print("Time:", round((end_ms - start_ms) * 1000), round((preprocess_ms-start_ms)*1000))
        return (np.rint(box)).astype(int), (np.rint(landms)).astype(int)


class WebCamTo3DCoordinates:
    def __init__(self):
        # SOURCE: https://github.com/thsant/3dmcap/blob/master/resources/Logitech-C920.yaml
        self.fx = self.fy = 1394.6027293299926
        self.cx = 995.588675691456
        self.cy = 599.3212928484164

    def pixel_to_world(self, x, y, Z):
        X = (x - self.cx) * Z / self.fx
        Y = (y - self.cy) * Z / self.fy
        return X, Y, Z

    def convert(self, eye_coordinates):
        (x_l, y_l, x_r, y_r) = eye_coordinates
        eye_distance = round(np.sqrt((x_r - x_l) ** 2 + (y_r - y_l) ** 2))  # Euclidean distance

        # Formula calculated based on 7 data points in the range of 37cm-140cm:
        estimated_eye_distance_cm = 5000.0 / max(eye_distance - 40, 0.000001) + 10
        estimated_eye_distance_m = estimated_eye_distance_cm / 100

        eye_midpoint_x = round((x_l + x_r) / 2)
        eye_midpoint_y = round((y_l + y_r) / 2)

        real_world_estimated = self.pixel_to_world(eye_midpoint_x, eye_midpoint_y, estimated_eye_distance_m)
        return real_world_estimated


class ExponentialMovingAverage:
    def __init__(self, alpha):
        self.alpha = alpha
        self.sum = 0

    def __call__(self, val):
        self.sum = self.alpha * val + (1 - self.alpha) * self.sum
        return self.sum


class GLWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.angle_x = 0
        self.angle_y = 0
        self.coord_x = 0
        self.coord_y = 0
        self.coord_z = 0
        self.last_pos = None
        self.scene_idx = 1
        self.num_scenes = 6
        self.animation_idx = 0
        self.h_w_ratio = MONITOR_SCREEN_HEIGHT / MONITOR_SCREEN_WIDTH

    def initializeGL(self):
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glEnable(GL_DEPTH_TEST)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        self.set_off_axis_projection(-1, 1, -1, 1, 1, 1000)

    def set_off_axis_projection(self, left, right, bottom, top, near, far):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glFrustum(left, right, bottom, top, near, far)
        glMatrixMode(GL_MODELVIEW)

    def set_frustum(self, x, y, z):
        self.coord_x = x
        self.coord_y = y
        self.coord_z = z
        self.update()

    def change_scene(self):
        self.scene_idx = (self.scene_idx + 1) % self.num_scenes

    def paintGL(self):
        x_offset = 0.5 + (-self.coord_x / (MONITOR_SCREEN_WIDTH / 2)) / 2
        y_offset = self.h_w_ratio/2 + (-self.coord_y / (MONITOR_SCREEN_HEIGHT / 2)) / 2
        z = max(0.01, self.coord_z)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        gluLookAt(0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)

        glFrustum(x_offset, x_offset - 1, y_offset, y_offset - self.h_w_ratio, z, 1000)
        self.draw_scene()

    def draw_scene(self):
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDisable(GL_DEPTH_TEST)

        if self.scene_idx == 0:  # Simple 3D rectangles
            glBegin(GL_QUADS)
            glColor3f(1.0, 0.0, 0.0)
            glVertex3f(-1.0, -1.0*self.h_w_ratio, -1.0)
            glVertex3f(-1.0, 1.0*self.h_w_ratio, -1.0)
            glVertex3f(1.0, 1.0*self.h_w_ratio, -1.0)
            glVertex3f(1.0, -1.0*self.h_w_ratio, -1.0)
            glEnd()

            glBegin(GL_QUADS)
            # Left face
            glColor3f(0.0, 0.5, 1.0)
            glVertex3f(-1.0, -1.0*self.h_w_ratio, -1.0)
            glVertex3f(-1.0, -1.0*self.h_w_ratio, 1.0)
            glVertex3f(-1.0, 1.0*self.h_w_ratio, 1.0)
            glVertex3f(-1.0, 1.0*self.h_w_ratio, -1.0)
            # Right face
            glColor3f(0.0, 0.5, 1.0)
            glVertex3f(1.0, -1.0*self.h_w_ratio, -1.0)
            glVertex3f(1.0, 1.0*self.h_w_ratio, -1.0)
            glVertex3f(1.0, 1.0*self.h_w_ratio, 1.0)
            glVertex3f(1.0, -1.0*self.h_w_ratio, 1.0)
            # Top face
            glColor3f(0.0, 1.0, 0.0)
            glVertex3f(-1.0, 1.0*self.h_w_ratio, -1.0)
            glVertex3f(-1.0, 1.0*self.h_w_ratio, 1.0)
            glVertex3f(1.0, 1.0*self.h_w_ratio, 1.0)
            glVertex3f(1.0, 1.0*self.h_w_ratio, -1.0)
            # Bottom face
            glColor3f(0.0, 1.0, 0.0)
            glVertex3f(-1.0, -1.0*self.h_w_ratio, -1.0)
            glVertex3f(1.0, -1.0*self.h_w_ratio, -1.0)
            glVertex3f(1.0, -1.0*self.h_w_ratio, 1.0)
            glVertex3f(-1.0, -1.0*self.h_w_ratio, 1.0)
            glEnd()

            glBegin(GL_QUADS)
            glColor3f(0.5, 0.5, 0.8)
            glVertex3f(-0.5, -0.5*self.h_w_ratio, -0.5)
            glVertex3f(0.5, -0.5*self.h_w_ratio, -0.5)
            glVertex3f(0.5, 0.5*self.h_w_ratio, -0.5)
            glVertex3f(-0.5, 0.5*self.h_w_ratio, -0.5)
            glEnd()

            glBegin(GL_QUADS)
            glColor3f(0.25, 0.8, 0.5)
            glVertex3f(-0.3, -0.3*self.h_w_ratio, -0.3)
            glVertex3f(0.3, -0.3*self.h_w_ratio, -0.3)
            glVertex3f(0.3, 0.3*self.h_w_ratio, -0.3)
            glVertex3f(-0.3, 0.3*self.h_w_ratio, -0.3)
            glEnd()

            glBegin(GL_QUADS)
            glColor3f(0.8, 0.1, 0.8)
            glVertex3f(-0.6, -0.1*self.h_w_ratio, -0.1)
            glVertex3f(0.6, -0.1*self.h_w_ratio, -0.1)
            glVertex3f(0.6, 0.1*self.h_w_ratio, -0.1)
            glVertex3f(-0.6, 0.1*self.h_w_ratio, -0.1)
            glEnd()
        elif self.scene_idx == 1:  # 3D checkerboard box
            self.draw_checkerboard_box()
        elif self.scene_idx == 2:  # 3D checkerboard box with triangles
            self.draw_checkerboard_box()
            glBegin(GL_TRIANGLES)
            for i in range(10):
                glColor3f(0.2 + (9-i) * 0.05, 0.6 + (9-i) * 0.05, 1.0)
                glVertex3f(0.0, -0.15, -0.9 + i * 0.08)
                glVertex3f(0.15, 0.15, -0.9 + i * 0.08)
                glVertex3f(-0.15, 0.15, -0.9 + i * 0.08)

                glColor3f(1.0, 0.2 + (9-i) * 0.05, 0.6 + (9-i) * 0.05)
                glVertex3f(0.5, 0.15, -0.9 + i * 0.08)
                glVertex3f(0.5, -0.15, -0.9 + i * 0.08)
                glVertex3f(0.8, 0, -0.9 + i * 0.08)

                glColor3f(0.2 + (9-i) * 0.05, 1.0, 0.6 + (9-i) * 0.05)
                glVertex3f(-0.5, 0.15, -0.9 + i * 0.08)
                glVertex3f(-0.5, -0.15, -0.9 + i * 0.08)
                glVertex3f(-0.8, 0, -0.9 + i * 0.08)
            glEnd()

        elif self.scene_idx == 3:  # 3D checkerboard box with triangles #2
            self.draw_checkerboard_box()
            glBegin(GL_TRIANGLES)
            for i in range(10):
                glColor3f(0.2 + i * 0.05, 0.3 + i * 0.05, 1.0)
                glVertex3f(0.1 + (i - 5) / 10, 0.15, -0.9 + i * 0.08)
                glVertex3f(0.1 + (i - 5) / 10, -0.15, -0.9 + i * 0.08)
                glVertex3f(-0.2 + (i - 5) / 10, 0, -0.9 + i * 0.08)
            glEnd()

        elif self.scene_idx == 4:  # 3D checkerboard box with triangles #3
            self.draw_checkerboard_box()
            glBegin(GL_TRIANGLES)
            self.animation_idx = (self.animation_idx+1) % 360
            for i in range(8):
                rotmat = np.array([[np.cos(np.deg2rad(i * 10 + self.animation_idx)), -np.sin(np.deg2rad(i * 10 + self.animation_idx))],
                                   [np.sin(np.deg2rad(i * 10 + self.animation_idx)), np.cos(np.deg2rad(i * 10 + self.animation_idx))]])
                rotated_1 = np.matmul(rotmat, np.array([[0.1], [0.15]]))
                rotated_2 = np.matmul(rotmat, np.array([[0.1], [-0.15]]))
                rotated_3 = np.matmul(rotmat, np.array([[-0.2], [0.0]]))

                glColor3f(1.0, 0.2 + i * 0.05, 0.3 + i * 0.05)
                glVertex3f(rotated_1[0, 0], rotated_1[1, 0], -0.9 + i * 0.11)
                glVertex3f(rotated_2[0, 0], rotated_2[1, 0], -0.9 + i * 0.11)
                glVertex3f(rotated_3[0, 0], rotated_3[1, 0], -0.9 + i * 0.11)
            glEnd()
        elif self.scene_idx == 5:  # 3D checkerboard box with octahedrons
            self.draw_checkerboard_box()
            self.animation_idx = (self.animation_idx + 1) % 360
            rotation = abs(self.animation_idx-180)/4-22.5
            self.draw_octahedron(0.5, 0, -0.7,
                                 0.2, 0.4, 0.2,
                                 0.8, 0.1, 0.2,
                                 0.6, 0.1, 0.2, np.deg2rad(rotation))
            self.draw_octahedron(-0.5, 0, -0.7,
                                 0.2, 0.4, 0.2,
                                 0.2, 0.1, 0.8,
                                 0.2, 0.1, 0.6, np.deg2rad(rotation))
            self.draw_octahedron(0, 0, -0.3,
                                 0.2, 0.4, 0.2,
                                 0.3, 0.8, 0.2,
                                 0.2, 0.6, 0.2, np.deg2rad(-rotation))

    def draw_octahedron(self, x, y, z,
                        rad_x, rad_y, rad_z,
                        color_r, color_g, color_b,
                        dk_color_r, dk_color_g, dk_color_b, rotation_x_z):

        rotmat = np.array(
            [[np.cos(rotation_x_z), -np.sin(rotation_x_z)],
             [np.sin(rotation_x_z), np.cos(rotation_x_z)]])
        rotated_back = np.matmul(rotmat, np.array([[0], [-rad_z]]))
        rotated_forward = np.matmul(rotmat, np.array([[0], [rad_z]]))
        rotated_left = np.matmul(rotmat, np.array([[-rad_x], [0]]))
        rotated_right = np.matmul(rotmat, np.array([[rad_x], [0]]))
        rb_x = rotated_back[0, 0] + x
        rb_z = rotated_back[1, 0] + z
        rf_x = rotated_forward[0, 0] + x
        rf_z = rotated_forward[1, 0] + z
        rl_x = rotated_left[0, 0] + x
        rl_z = rotated_left[1, 0] + z
        rr_x = rotated_right[0, 0] + x
        rr_z = rotated_right[1, 0] + z


        glBegin(GL_TRIANGLES)
        glColor3f(dk_color_r, dk_color_g, dk_color_b)
        glVertex3f(rb_x, y, rb_z)
        glVertex3f(x, y+rad_y, z)
        glVertex3f(rr_x, y, rr_z)

        glColor3f(color_r, color_g, color_b)
        glVertex3f(rb_x, y, rb_z)
        glVertex3f(x, y-rad_y, z)
        glVertex3f(rr_x, y, rr_z)

        glColor3f(dk_color_r, dk_color_g, dk_color_b)
        glVertex3f(rb_x, y, rb_z)
        glVertex3f(x, y-rad_y, z)
        glVertex3f(rl_x, y, rl_z)

        glColor3f(color_r, color_g, color_b,)
        glVertex3f(rb_x, y, rb_z)
        glVertex3f(x, y+rad_y, z)
        glVertex3f(rl_x, y, rl_z)


        glColor3f(color_r, color_g, color_b,)
        glVertex3f(rf_x, y, rf_z)
        glVertex3f(x, y+rad_y, z)
        glVertex3f(rr_x, y, rr_z)

        glColor3f(dk_color_r, dk_color_g, dk_color_b)
        glVertex3f(rf_x, y, rf_z)
        glVertex3f(x, y-rad_y, z)
        glVertex3f(rr_x, y, rr_z)

        glColor3f(color_r, color_g, color_b,)
        glVertex3f(rf_x, y, rf_z)
        glVertex3f(x, y-rad_y, z)
        glVertex3f(rl_x, y, rl_z)

        glColor3f(dk_color_r, dk_color_g, dk_color_b)
        glVertex3f(rf_x, y, rf_z)
        glVertex3f(x, y+rad_y, z)
        glVertex3f(rl_x, y, rl_z)
        glEnd()

    def draw_checkerboard_box(self):
        glBegin(GL_QUADS)
        for i in range(-9, 11):
            for j in range(-9, 11):
                if min(abs(j / 10), abs((j - 1) / 10)) <= self.h_w_ratio:
                    if (i + j) % 2 == 0:
                        glColor3f(0.6, 0.6, 0.6)
                        glVertex3f((i - 1) / 10, min(max((j - 1) / 10, -self.h_w_ratio), self.h_w_ratio), -1.0)
                        glVertex3f((i - 1) / 10, min(max(j / 10, -self.h_w_ratio), self.h_w_ratio), -1.0)
                        glVertex3f(i / 10, min(max(j / 10, -self.h_w_ratio), self.h_w_ratio), -1.0)
                        glVertex3f(i / 10, min(max((j - 1) / 10, -self.h_w_ratio), self.h_w_ratio), -1.0)
        for i in range(-9, 1):
            for j in range(-9, 11):
                if min(abs(j/10), abs((j-1)/10)) <= self.h_w_ratio:
                    if (i + j) % 2 == 1:
                        glColor3f(0.6, 0.6, 0.6)
                        glVertex3f(1, min(max((j - 1) / 10, -self.h_w_ratio), self.h_w_ratio), (i - 1) / 10)
                        glVertex3f(1, min(max(j / 10, -self.h_w_ratio), self.h_w_ratio), (i - 1) / 10)
                        glVertex3f(1, min(max(j / 10, -self.h_w_ratio), self.h_w_ratio), i / 10)
                        glVertex3f(1, min(max((j - 1) / 10, -self.h_w_ratio), self.h_w_ratio), i / 10)
                    else:
                        glColor3f(0.6, 0.6, 0.6)
                        glVertex3f(-1, min(max((j - 1) / 10, -self.h_w_ratio), self.h_w_ratio), (i - 1) / 10)
                        glVertex3f(-1, min(max(j / 10, -self.h_w_ratio), self.h_w_ratio), (i - 1) / 10)
                        glVertex3f(-1, min(max(j / 10, -self.h_w_ratio), self.h_w_ratio), i / 10)
                        glVertex3f(-1, min(max((j - 1) / 10, -self.h_w_ratio), self.h_w_ratio), i / 10)
        for i in range(-9, 1):
            for j in range(-9, 11):
                if (i + j) % 2 == 1:
                    glColor3f(0.6, 0.6, 0.6)
                    glVertex3f((j - 1) / 10, 1*self.h_w_ratio, (i - 1) / 10)
                    glVertex3f(j / 10, 1*self.h_w_ratio, (i - 1) / 10)
                    glVertex3f(j / 10, 1*self.h_w_ratio, i / 10)
                    glVertex3f((j - 1) / 10, 1*self.h_w_ratio, i / 10)
                else:
                    glColor3f(0.6, 0.6, 0.6)
                    glVertex3f((j - 1) / 10, -1*self.h_w_ratio, (i - 1) / 10)
                    glVertex3f(j / 10, -1*self.h_w_ratio, (i - 1) / 10)
                    glVertex3f(j / 10, -1*self.h_w_ratio, i / 10)
                    glVertex3f((j - 1) / 10, -1*self.h_w_ratio, i / 10)
        glEnd()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D AI Screen")
        self.setGeometry(100, 100, 1920, 1080)
        self.gl_widget = GLWidget(self)
        self.setCentralWidget(self.gl_widget)
        self.is_fullscreen = False

    def set_frustum(self, x, y, z):
        self.gl_widget.set_frustum(x, y, z)

    def change_scene(self):
        self.gl_widget.change_scene()

    def keyPressEvent(self, event):
        if event.key() == PySide6.QtCore.Qt.Key.Key_Escape:  # press 'ESC' to quit
            videoSource.release()
            cv2.destroyAllWindows()
            sys.exit()
        elif event.key() == ord(" "):  # press 'SPACE' to change view
            view.change_scene()
        elif event.key() == 70:  # press 'f' to toggle fullscreen mode
            self.is_fullscreen = not self.is_fullscreen
            if self.is_fullscreen:
                self.showFullScreen()
            else:
                self.showNormal()


app = QApplication(sys.argv)
view = MainWindow()
view.show()

xSmoother = ExponentialMovingAverage(0.3)
ySmoother = ExponentialMovingAverage(0.3)
zSmoother = ExponentialMovingAverage(0.2)

videoSource = VideoSource()
webCamTo3D = WebCamTo3DCoordinates()

faceDetectorAndLocalizer = FaceDetectorAndLocalizer()

in_fullscreen = False

while True:
    start_time = time.time()

    frame = videoSource.get_frame()
    to_show = frame.copy()

    frame_capture_time = time.time()

    bbox, landmarks = faceDetectorAndLocalizer.detect_and_localize(frame.copy())

    (x0, y0, x1, y1) = bbox
    to_show = cv2.blur(to_show, (400, 400))
    to_show[y0:y1, x0:x1, :] = frame[y0:y1, x0:x1, :]
    color = (0, 255, 0)
    cv2.rectangle(to_show, (x0, y0), (x1, y1), color, 2)

    for i in range(5):
        cv2.circle(img=to_show, center=(landmarks[2*i], landmarks[2*i+1]), radius=2, color=(255, 255, 255), thickness=2)

    estimated_coordinates = webCamTo3D.convert(landmarks[0:4])
    (X, Y, Z) = estimated_coordinates

    X = xSmoother(X)
    Y = ySmoother(Y)
    Z = zSmoother(Z)

    view.set_frustum(X, Y, Z)

    to_show = cv2.resize(to_show, (round(CAM_WIDTH / 2), round(CAM_HEIGHT / 2)))

    cv2.imshow('Face tracking', to_show)
    midtime = time.time()

    k = cv2.waitKey(1) & 0xff
    if k == 27:  # press 'ESC' to quit
        videoSource.release()
        cv2.destroyAllWindows()
        sys.exit()

    endtime = time.time()
    print("Total time:", round((endtime - start_time) * 1000), "ms",
          "Capture part", round((frame_capture_time - start_time) * 1000), "ms",
          "Calculation part:", round((midtime - frame_capture_time) * 1000), "ms",
          "Draw&Wait part:", round((endtime - midtime) * 1000), "ms")
