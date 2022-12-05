from cv2 import circle, clipLine, cvtColor, destroyWindow, destroyAllWindows, dilate, findContours, \
    getStructuringElement, imread, imshow, moveWindow, namedWindow, normalize, putText, resize, setMouseCallback, \
    waitKey, CHAIN_APPROX_NONE, COLOR_BGR2GRAY, EVENT_MOUSEMOVE, EVENT_LBUTTONUP, EVENT_LBUTTONDOWN, \
    FONT_HERSHEY_SIMPLEX, FONT_HERSHEY_DUPLEX, IMREAD_GRAYSCALE, INTER_AREA, INTER_LINEAR, MORPH_RECT, NORM_MINMAX, \
    RETR_LIST, WINDOW_GUI_NORMAL
from dataclasses import dataclass, field, InitVar
from matplotlib.pyplot import get_cmap
from numba import njit
from numpy import arctan, arctan2, argwhere, argmax, array, bool_, concatenate, cos, deg2rad, diff, divide, dot, \
    flatnonzero, floor, index_exp, ndarray, pi, rad2deg, rint, sin, sqrt, subtract, unique, zeros, zeros_like
from numpy.random import choice
from pathlib import Path
from scipy.spatial.distance import cdist
from skimage.draw import circle_perimeter, circle_perimeter_aa, line, line_aa
import torch
from torch import Tensor
from typing import List, Tuple


@dataclass(frozen=True)
class GIS:
    """
    GIS class processes image, stores and shares map data

    image:          ndarray     Greyscale 2D image. \n
    smooth_factor:  float       Smoothness of modeled bathymetry. \n
    nav_rgb:        ndarray     RGB of image. \n
    depth_rgb:      ndarray     RGB of modeled bathymetry. \n
    edge_array:     ndarray     Pixel coordinates of boarders around/within navigable area. \n
    dist_map:       ndarray     Greyscale distance matrix of pixels from the boarders (within navigable area). \n
    depth_map:      ndarray     Greyscale depth matrix within navigable area. \n
    depth_mosaic:   ndarray     3D model of the environment with depth dimension.
    """

    image: ndarray
    max_depth: InitVar[int]
    sensor_range: InitVar[int]
    smooth_factor: float = 2.
    nav_rgb: ndarray = field(init=False)
    depth_rgb: ndarray = field(init=False)
    edge_array: ndarray = field(init=False)
    dist_map: ndarray = field(init=False)
    depth_map: ndarray = field(init=False)
    depth_mosaic: ndarray = field(init=False)

    def __post_init__(self, max_depth, sensor_range) -> None:
        """
        Initialization of environment data

        :param int max_depth:       Maximum depth of the bathymetry.
        :param int sensor_range:    Maximum sonar beam range.
        :return: None
        """

        # Binary and rgb bitmaps
        bit_maps = zeros((6,) + self.image.shape, 'uint8')
        object.__setattr__(self, 'nav_rgb', bit_maps[0:3].transpose(1, 2, 0))
        object.__setattr__(self, 'depth_rgb', bit_maps[3:6].transpose(1, 2, 0))
        bit_maps.flags.writeable = False

        self.nav_rgb[..., 0], self.nav_rgb[..., 1], self.nav_rgb[..., 2] = (
            (self.image > 0).view('uint8'),
            (~ (self.image > 0)).view('uint8'),
            zeros_like(self.image, 'uint8')
        )
        self.nav_rgb[:] *= 255
        self.nav_rgb.flags.writeable = False
        nav_map, gnd_map = self.nav_rgb[..., 0].view('bool'), self.nav_rgb[..., 1].view('bool')

        # Edges of survey area
        contours, _ = findContours(gnd_map.view('uint8'), RETR_LIST, CHAIN_APPROX_NONE)
        edge_array = concatenate(contours, 0).squeeze()[:, ::-1].T
        edge_array = edge_array[:, (edge_array != 0).all(0) & (edge_array != 511).all(0)]
        edge_array.flags.writeable = False
        object.__setattr__(self, 'edge_array', edge_array[:])

        # Distance and depth rasterized maps
        float_maps = zeros((2,) + self.image.shape, 'float64')
        object.__setattr__(self, 'dist_map', float_maps[0])
        object.__setattr__(self, 'depth_map', float_maps[1])
        float_maps.flags.writeable = False

        # Distance matrix
        edge_dist = cdist(argwhere(nav_map), edge_array.T)
        self.dist_map[nav_map] = edge_dist.min(1)
        self.dist_map.flags.writeable = False
        dist_map = self.dist_map

        # Depth matrix
        self.depth_map[:] = dist_map
        self.depth_map[nav_map] += 1
        self.depth_map[nav_map] **= (1 / self.smooth_factor)
        self.depth_map[:] = normalize(self.depth_map, None, 0, max_depth, NORM_MINMAX)
        self.depth_map.flags.writeable = False
        depth_map = self.depth_map

        # Depth rgb bitmap
        depth_color = get_cmap('autumn_r')(normalize(depth_map, None, 0, 1, NORM_MINMAX))[..., :3][..., ::-1]
        self.depth_rgb[:] = rint(depth_color * 255.).astype('uint8')
        self.depth_rgb[gnd_map] = (0, 255, 0)
        self.depth_rgb.flags.writeable = False

        # 3d environment mosaic
        depth_model = zeros((sensor_range + 1,) + self.image.shape, 'uint8')
        object.__setattr__(self, 'depth_mosaic', depth_model[:])
        depth_model.flags.writeable = False
        water_column_val = 64
        for n, z in enumerate(self.depth_mosaic):
            z[:] = (depth_map > n) * water_column_val
        self.depth_mosaic.flags.writeable = False

    ...


class INS:
    """
    INS class handles data relevant to a navigation sensor

    position:       ndarray     A sliding window of x, y coordinates:
                                    Row 0- Current values x and y \n
                                    Row 1- Previous values x and y \n
                                    Row 2- Delta change in  position \n
    steps:          ndarray     The number of steps taken through the environment\n
    angle_window:   ndarray     A sliding window of the angle of motion:
                                    Pages- Queue window \n
                                    Row 0- arc-tan2 of position delta y and x \n
                                    Row 1- Sin component of Row 0 \n
                                    Row 2- Cos component of Row 0 \n
    angle_filter:   ndarray     A moving average of the angle of motion:
                                    Row 0- arctan2 of sum of sin and cos components \n
                                    Row 1- rad2deg conversion of Row 0 \n
    rotate:         ndarray     A R2 rotation matrix \n
    body_axes:      ndarray     A set of points at end of each body axis
                                    Page 0- Body Frame \n
                                    Page 1- World Frame \n
                                    Row 0- Bow axis \n
                                    Row 1- Stern axis \n
                                    Row 2- Port axis \n
                                    Row 3- Starboard axis \n
    axes:           list        A List of points along each axis
    """

    __slots__ = ('_dims', '_gnd_map', '_steps_plt', '_axes_plt', '_position', '_steps', '_angle_window',
                 '_angle_filter', '_rotate', '_body_axes', '_axes', '_edge_dist')

    def __init__(self, ins_data: Tuple[ndarray, ndarray, ndarray]) -> None:
        """
        Initialization of data fields

        :param tuple[ndarray, ndarray, ndarray] ins_data:    Data plots relevant to INS data
        :return: None
        """

        self._gnd_map, self._steps_plt, self._axes_plt = ins_data

        self._dims = self._gnd_map.shape[0]
        self._position = zeros((3, 2), 'int64')
        self._steps = zeros(1, 'int64')
        self._angle_window = zeros((12, 3, 1), 'float64')
        self._angle_filter = zeros((2, 1), 'float64')
        self._rotate = zeros((4, 1), 'float64')
        self._body_axes = zeros((2, 4, 2), 'float64')
        self._body_axes[0] = ((-724, 0), (724, 0), (0, -724), (0, 724))
        self._axes = [array((), 'int64'), array((), 'int64'), array((), 'int64'), array((), 'int64')]
        self._edge_dist = zeros(1, 'float64')

    @staticmethod
    @njit()
    def __update_position(
            waypoint_x: int, waypoint_y: int, position: ndarray, steps: ndarray, steps_plt: ndarray
    ) -> None:
        """ Update position of agent within environment using new waypoint """

        # Retrieve waypoint coordinates
        position[0, 0], position[0, 1] = waypoint_y, waypoint_x

        # Calculate delta change in position
        subtract(position[0], position[1], position[2])

        # Plot position data
        steps_plt[position[0, 0], position[0, 1]] = 255

        # Register additional step
        steps[0] += 1

    @staticmethod
    @njit()
    def __update_heading(
            angle_window: ndarray, position: ndarray, angle_filter: ndarray, rotate: ndarray, body_axes: ndarray
    ) -> None:
        """ Update angle of motion of agent within the environment """

        # Cycle sliding window
        angle_window[1:] = angle_window[:-1]

        # Calculate the angle of delta change in position
        arctan2(position[2, 0], position[2, 1], angle_window[0, 0])
        angle_window[0, 0] += (pi / 2)
        angle_window[0, 0] %= (2 * pi)

        # Calculate sin and cos components of angles in window
        sin(angle_window[0, 0], angle_window[0, 1])
        cos(angle_window[0, 0], angle_window[0, 2])

        # Calculate the average heading angle in the window
        arctan2(angle_window[:, 1].sum(), angle_window[:, 2].sum(), angle_filter[0])
        angle_filter[0] %= (2 * pi)

        # Convert to degrees
        rad2deg(angle_filter[0], angle_filter[1])

        # Update rotation matrix with the latest heading angle estimation
        cos(angle_filter[0], rotate[0])
        sin(-angle_filter[0], rotate[1])
        sin(angle_filter[0], rotate[2])
        cos(angle_filter[0], rotate[3])

        # Rotate and offset dummy axes to new state
        dot(body_axes[0], rotate.reshape(2, 2), body_axes[1])
        body_axes[1] += position[0]
        rint(body_axes[1], body_axes[1])

    @staticmethod
    def __update_ortho_axes(
            dims: int, position: ndarray, body_axes: ndarray, gnd_map: ndarray, axis_plt: ndarray,
            nav_axes: List[ndarray], edge_dist: ndarray
    ) -> None:
        """ Update body axes lines and plot """

        # Update axes plot with new axes data
        axis_plt[:] = 0
        rect = (0, 0, dims, dims)
        pos = (position[0, 0], position[0, 1])

        # Calculate visible lines along body axes and plot axes data
        for n, axis in enumerate(body_axes[1]):
            *_, end = clipLine(rect, pos, (axis[0].astype('int64'), axis[1].astype('int64')))
            axis_line = line(*pos, *end)
            axis_viz = gnd_map[axis_line]
            axis_range = argmax(axis_viz) if axis_viz.any() else axis_viz.size
            nav_axes[n] = array((axis_line[0][:axis_range], axis_line[1][:axis_range]))
            axis_plt[nav_axes[n][0], nav_axes[n][1]] = 255

        edge_dist[0] = nav_axes[0][0].size

    @staticmethod
    @njit()
    def __log(position: ndarray) -> None:
        """ Record previous position of INS """

        position[1] = position[0]

    def update(self, x_coord: int, y_coord: int) -> Tuple[ndarray, Tuple[ndarray, ndarray]]:
        """ Globally update INS measurements """

        self.__update_position(x_coord, y_coord, self._position, self._steps, self._steps_plt)
        self.__update_heading(self._angle_window, self._position, self._angle_filter, self._rotate, self._body_axes)
        self.__update_ortho_axes(
            self._dims, self._position, self._body_axes, self._gnd_map, self._axes_plt, self._axes, self._edge_dist
        )
        self.__log(self._position)

        return self.position, self.ortho_axes

    def hot_start(self, x_coord: int, y_coord: int) -> None:
        """ Prime INS position """

        self._position[0, 0], self._position[0, 1], self._position[1, 0], self._position[1, 1] = (
            y_coord, x_coord, y_coord, x_coord
        )
        self._steps[0] -= 1

    def reset(self) -> None:
        """ Reset state of all values to zero """

        self._position[:] = 0
        self._steps[:] = 0
        self._angle_window[:] = 0
        self._angle_filter[:] = 0
        self._rotate[:] = 0
        self._body_axes[1] = 0
        self._steps_plt[:] = 0
        self._axes_plt[:] = 0
        self._axes = [array((), 'int64'), array((), 'int64'), array((), 'int64'), array((), 'int64')]

    @property
    def steps(self) -> ndarray:
        """ Return Steps """

        output, output.flags.writeable = self._steps[:], False
        return output

    @property
    def position(self) -> ndarray:
        """ Return position """

        output, output.flags.writeable = self._position[0], False
        return output

    @property
    def heading(self) -> ndarray:
        """ Return heading """

        output, output.flags.writeable = self._angle_filter[1], False
        return output

    @property
    def ortho_axes(self) -> Tuple[ndarray, ndarray]:
        """ Return axes orthogonal to yaw """

        output1, output2, output1.flags.writeable, output2.flags.writeable = self._axes[2], self._axes[3], False, False
        return output1, output2

    @property
    def edge_dist(self) -> ndarray:
        """ Return Steps """

        output, output.flags.writeable = self._edge_dist[:], False
        return output


class SONAR:
    """
    Sonar class handles data relevant to a sonar sensor

    range:          Total beam range
    angle:          Total beam angle
    depth:          Depth at current position
    swath_width:    Width of sonar beam
    coverage:       Percentage cover of the environment
    overlap:        Percentage of repeated samples in sensor area
    arg_sensor:     List with current and previous sensor coords
    swath coord:    Coordinates of beam arc for perspective plot
    swath center:   center of beam along axis
    """

    __slots__ = ('_dims', '_range', '_angle', '_nav_map', '_depth_map', '_nav_rgb', '_depth_rgb',
                 '_sensor_plt', '_samples_plt', '_swath_plt', '_survey_plt', '_depth_mosaic', '_depth',
                 '_swath_width', '_coverage', '_overlap', '_arg_sensor', '_swath_coord', '_swath_center')

    def __init__(
            self, sonar_data: Tuple[ndarray, ndarray, ndarray, ndarray, ndarray, ndarray, ndarray, ndarray, ndarray],
            sensor_range: int, sensor_angle: int
    ) -> None:

        """ Initialization of data fields """

        (self._nav_map, self._nav_rgb, self._depth_map, self._depth_rgb,
         self._depth_mosaic, self._sensor_plt, self._samples_plt, self._swath_plt, self._survey_plt) = sonar_data

        self._range = sensor_range
        self._angle = deg2rad(sensor_angle) / 2
        self._dims = self._nav_map.shape[0]
        self._depth = zeros(1, 'float64')
        self._swath_width = zeros(1, 'float64')
        self._coverage = zeros((3, 1), 'float64')
        self._overlap = zeros((3, 1), 'float64')
        self._arg_sensor = [zeros((2, 1), 'int64'), zeros((2, 1), 'int64')]

        beam_radius = array(circle_perimeter(0, 0, self._range))
        beam_radius = unique(beam_radius, False, False, False, 1).squeeze()
        degrees = arctan2(beam_radius[0], beam_radius[1])
        degrees += (pi / 2.0)
        degrees %= (2.0 * pi)
        port_angle, starboard_angle = pi + self._angle, pi - self._angle
        beam_sector = (port_angle >= degrees) & (degrees >= starboard_angle)

        self._swath_coord = beam_radius[:, beam_sector]
        self._swath_center = zeros((2, 1), 'int64')

    @staticmethod
    @njit()
    def __update_beam(
            position: ndarray, ortho_axes: Tuple[ndarray, ndarray], angle: float,
            _range: int, depth: ndarray, swath_width: ndarray, depth_map: ndarray
    ) -> Tuple[ndarray, ndarray, int]:
        """ Calculations to construct sonar beam in environment """

        # Calculate beam axis line
        beam_axis = concatenate((ortho_axes[0][:, 1:][:, ::-1], ortho_axes[1]), 1)

        # Update depth measurement
        depth[0] = depth_map[position[0], position[1]]

        # Calculate swath of sonar beam
        beam_depth = zeros(beam_axis.shape[1], 'float64')
        for n, coord in enumerate(beam_axis.T):
            beam_depth[n] = depth_map[coord[0], coord[1]]

        beam_extent = (position - beam_axis.T) ** 2
        beam_extent = sqrt(beam_extent[:, 0] + beam_extent[:, 1])
        beam_range = sqrt(beam_depth ** 2 + beam_extent ** 2)
        beam_angle = arctan(beam_extent / beam_depth)
        beam_center = int(beam_angle.argmin())
        ensonified = (beam_angle <= angle) & (beam_range <= _range)

        # Calculate Acoustic Shadow
        port = beam_angle[:(beam_center + 1)][::-1], ensonified[:(beam_center + 1)][::-1]
        starboard = beam_angle[beam_center:], ensonified[beam_center:]

        sweep_angle = 0
        for n, theta in enumerate(port[0]):
            if theta >= sweep_angle:
                sweep_angle = theta
            else:
                port[1][n] = False

        sweep_angle = 0
        for n, theta in enumerate(starboard[0]):
            if theta >= sweep_angle:
                sweep_angle = theta
            else:
                starboard[1][n] = False

        # Update swath measurement
        swath_width[:] = ensonified.sum()
        beam_breaks = zeros(ensonified.size + 1)
        beam_breaks[1:] = ensonified

        beam_echos = flatnonzero(diff(beam_breaks))
        if beam_echos.size % 2:
            beam_echos = concatenate((beam_echos, array([beam_breaks.size - 1])))

        return beam_echos.reshape(-1, 2), beam_axis, beam_center

    @staticmethod
    def __update_sensor(
            beam_echos: ndarray, beam_axis: ndarray, _range: int, nav_map: ndarray, sensor_plt: ndarray
    ) -> None:
        """ Draw beam on sensor plot and calculate coverage """

        # Update sensor plot
        sensor_plt[:] = 0
        for edges in beam_echos:
            start = (beam_axis[0][edges[0]], beam_axis[1][edges[0]])
            stop = (beam_axis[0][edges[1] - 1], beam_axis[1][edges[1] - 1])
            *echo, _ = line_aa(start[0], start[1], stop[0], stop[1])
            sensor_plt[echo[0], echo[1]] = nav_map[echo[0], echo[1]] * 255

    @staticmethod
    @njit()
    def __update_measurements(
            beam_axis: ndarray, beam_center: int, swath_center: ndarray, swath_coord: ndarray,
            overlap: ndarray, coverage: ndarray, arg_sensor: List[ndarray],
            nav_map: ndarray, depth_rgb: ndarray, depth_mosaic: ndarray,
            sensor_plt: ndarray, samples_plt: ndarray, survey_plt: ndarray
    ) -> Tuple[ndarray, ndarray, ndarray]:
        """ Update sonar measurements """

        swath = argwhere(sensor_plt).T

        # Update Overlap measurement
        overlap_sum = 0
        for echo in swath.T:
            if samples_plt.view(bool_)[echo[0], echo[1]]:
                overlap_sum += 1

        overlap[0], overlap[1] = overlap_sum, swath[0].size

        if overlap[1] > 0:
            divide(overlap[0], overlap[1], overlap[2])
            overlap[2] *= 100.0

        # Update samples and survey plot
        for echo in swath.T:
            survey_plt[echo[0], echo[1]] = (255, 255, 255)

        if arg_sensor[1].size > 2:
            for echo in arg_sensor[1].T:
                survey_plt[echo[0], echo[1]] = depth_rgb[echo[0], echo[1]]
                samples_plt[echo[0], echo[1]] = 255

        # Update coverage measurement
        coverage[0], coverage[1] = samples_plt.view(bool_).sum(), nav_map.sum()
        divide(coverage[0], coverage[1], coverage[2])
        coverage[2] *= 100.0

        # Calculate swath in 3d map
        swath_img = zeros((33, beam_axis.shape[1]), 'uint8')
        for dth, contour in enumerate(depth_mosaic):
            for wth, coord in enumerate(beam_axis.T):
                swath_img[dth, wth] = contour[coord[0], coord[1]]

        swath_center[1] = beam_center
        swath_arc = (swath_coord + swath_center).T
        swath_clipped = (swath_arc[:, 1] >= 0) & (swath_arc[:, 1] < swath_img.shape[1])

        return swath, swath_img, swath_arc[swath_clipped]

    @staticmethod
    def __update_swath(dims: int, swath_img: ndarray, swath_arc: ndarray, beam_center: int, swath_plt: ndarray) -> None:
        """ Draw Swath plot """

        port, starboard = beam_center - 30, beam_center + 30
        port = 0 if port < 0 else dims if port > dims else port
        starboard = 0 if starboard < 0 else dims if starboard > dims else starboard

        for sector in swath_arc:
            beam = line(0, beam_center, sector[0], sector[1])
            for sample in zip(*beam):
                if swath_img[sample[0], sample[1]]:
                    swath_img[sample[0], sample[1]] = 255
                else:
                    break

        resize(swath_img[:, port:starboard], swath_plt.shape, swath_plt)

    @staticmethod
    @njit()
    def __log(arg_sensor: List[ndarray]) -> None:
        """ Record previous sensor swath """

        arg_sensor[1] = arg_sensor[0]

    def update(self, position: ndarray, ortho_axes: Tuple[ndarray, ndarray]) -> None:
        """ Globally update SONAR measurements """

        beam_echos, beam_axis, beam_center = self.__update_beam(
            position, ortho_axes, self._angle, self._range, self._depth, self._swath_width, self._depth_map
        )
        self.__update_sensor(beam_echos, beam_axis, self._range, self._nav_map, self._sensor_plt)
        swath, swath_img, swath_arc = self.__update_measurements(
            beam_axis, beam_center, self._swath_center, self._swath_coord,
            self._overlap, self._coverage, self._arg_sensor,
            self._nav_map, self._depth_rgb, self._depth_mosaic,
            self._sensor_plt, self._samples_plt, self._survey_plt
        )
        self._arg_sensor[0] = swath
        self.__update_swath(self._dims, swath_img, swath_arc, beam_center, self._swath_plt)
        self.__log(self._arg_sensor)

    def reset(self) -> None:
        """ Reset state of all values to zero """

        self._depth[:] = 0
        self._swath_width[:] = 0
        self._coverage[:] = 0
        self._overlap[:] = 0
        self._sensor_plt[:] = 0
        self._samples_plt[:] = 0
        self._swath_plt[:] = 0
        self._arg_sensor = [zeros((2, 1), 'int64'), zeros((2, 1), 'int64')]
        self._survey_plt[:] = self._nav_rgb

    @property
    def sensor_range(self):
        """ Return sensor range """

        return self._range

    @property
    def arg_sensor(self):
        """ Return sonar field coordinated """

        return self._arg_sensor[1]

    @property
    def depth(self):
        """ Return depth """

        output, output.flags.writeable = self._depth[:], False
        return output

    @property
    def swath_width(self):
        """ Return swath width """

        output, output.flags.writeable = self._swath_width[:], False
        return output

    @property
    def coverage(self):
        """ Return coverage """

        output, output.flags.writeable = self._coverage[:], False
        return output[2]

    @property
    def overlap(self):
        """ Return overlap """

        output, output.flags.writeable = self._overlap[:], False
        return output[2]


class ASV:
    """
    ASV class handles navigation and operation of sensor platform. Aggregates data and provides control to the agent.

    Attributes:
        data_plot:  ndarray     Real-time bitmaps representing sensory data.
        step_map:   ndarray     Real-time bitmap of previous steps.
        axes_map:   ndarray     Real-time bitmap of orientation axes.
        sensor_map: ndarray     Real-time bitmap of sonar field
        sample_map: ndarray     Real-time bitmap of samples collected
        survey_rgb: ndarray     Real-time RGB image with sample data overlay
        swath_pov:  ndarray     Real-time grey image of sonar swath field within water column
        ins:        obj         INS object
        sonar:      obj         Sonar object
        actions:    dict        Possible actions to be taken
    """

    __slots__ = ('_dims', '_data_plot', '_step_map', '_axes_map', '_sensor_map', '_sample_map', '_survey_rgb',
                 '_swath_pov', '_wtr_map', '_gnd_map', '_ins', '_sonar', '_start_coord', '_waypoints', '_start_bearing',
                 '_bearing', '_steps', '_end_steps', '_sectors', '_compass', '_action_keys', '_steering')

    def __init__(self, gis_data: GIS, sensor_range: int, sensor_angle: int, compass_sectors: int) -> None:
        """
        Initialize sensors and drive functions

        :param GIS gis_data:            The GIS data container.
        :param int sensor_range:        The beam range of sonar. Min = 1, Max = 200
        :param int sensor_angle:        The beam angle of sonar. Min = 1, Max = 180.
        :param int compass_sectors:     The sensitivity of turning actions, multiples of 4. Min = 4, Max = 24.
        :return None:
        """

        # Sensory data binary bitmaps
        self._dims = gis_data.image.shape[0]
        dims = self._dims
        quads = (
            index_exp[0:dims, 0:dims],
            index_exp[0:dims:, (dims + 1):(2 * dims + 1)],
            index_exp[(dims + 1):(2 * dims + 1), 0:dims],
            index_exp[(dims + 1):(2 * dims + 1), (dims + 1):(2 * dims + 1)]
        )
        data_maps = zeros((2 * dims + 1, 2 * dims + 1), 'uint8')
        self._step_map, self._axes_map, self._sensor_map, self._sample_map = (
            data_maps[quads[0]], data_maps[quads[1]], data_maps[quads[2]], data_maps[quads[3]]
        )
        self._data_plot = data_maps[:]
        data_maps.flags.writeable = False
        self._data_plot[dims, :], self._data_plot[:, dims] = 127, 127
        self._data_plot.flags.writeable = False

        # Survey Image and swath plot
        data_images = zeros((4,) + gis_data.image.shape, 'uint8')
        self._survey_rgb, self._swath_pov = data_images[0:3].transpose(1, 2, 0), data_images[3]
        data_images.flags.writeable = False
        self.survey_rgb[:] = gis_data.nav_rgb

        # Map defining non-navigable space
        self._wtr_map = gis_data.nav_rgb[..., 0].view('bool')
        self._gnd_map = gis_data.nav_rgb[..., 1].view('bool')

        # Create Sensor objects and share ASV data buffers
        ins_data = (gis_data.nav_rgb[..., 1].view('bool'), self._step_map, self._axes_map)
        self._ins = INS(ins_data)

        sonar_data = (
            gis_data.nav_rgb[..., 0].view('bool'), gis_data.nav_rgb, gis_data.depth_map, gis_data.depth_rgb,
            gis_data.depth_mosaic, self._sensor_map, self._sample_map, self._swath_pov, self._survey_rgb
        )
        self._sonar = SONAR(sonar_data, sensor_range, sensor_angle)

        # Calculate compass params
        self._start_coord = array([0, 0], 'int64')
        self._start_bearing, self._bearing = 0, 0
        self._steps, self._end_steps = 0, 0
        self._sectors = compass_sectors
        self._waypoints = array([], 'int64')

        tics = unique(array(circle_perimeter(0, 0, 724)), axis=1).squeeze()
        degrees = arctan2(tics[0], tics[1])
        degrees += (pi / 2.0)
        rad2deg(degrees, out=degrees)
        degrees %= 360.0
        degrees_ord = degrees.argsort()
        floor(degrees.T[degrees_ord], out=degrees)
        degrees_major = argwhere(diff(degrees, prepend=-1) > 0).ravel()[::(360 // self._sectors)]

        self._compass = tics[:, degrees_ord][:, degrees_major]
        self._action_keys = {0: ord('a'), 1: ord('ÿ'), 2: ord('d')}
        self._steering = {}

        ...

    def __update_bearing(self) -> None:
        """ Calculate new bearing from current position """

        rect = (0, 0, self._dims, self._dims)
        pos = (self._ins.position[0], self._ins.position[1])
        *_, end = clipLine(rect, pos, tuple(self._compass[:, self._bearing] + self._ins.position))[1:3]
        bearing_line = line(*pos, *end)
        bearing_viz = self._gnd_map[bearing_line]
        bearing_range = argmax(bearing_viz) if bearing_viz.any() else bearing_viz.size
        bearing_coord = array((bearing_line[0][:bearing_range], bearing_line[1][:bearing_range])).T
        self._waypoints = bearing_coord[1:] if bearing_coord.size > 2 else bearing_coord
        self._steps = 0
        self._end_steps = self._waypoints.shape[0] - 1

    def __turn(self, move_key: int) -> Tuple[ndarray, bool, bool]:
        """ Changing bearing """

        turning = True

        # Update bearing
        left = ord('a')
        adjustment = -1 if move_key == left else 1
        self._bearing = (self._bearing + adjustment) % self._compass.shape[1]
        self.__update_bearing()
        waypoints, moving, _ = self.__move(move_key)

        return waypoints, moving, turning

    def __move(self, move_key: int) -> Tuple[ndarray, bool, bool]:
        """ Move forward if not blocked """

        moving, turning = True, False

        # Move forward
        if self._steps < self._end_steps:
            waypoint = self._waypoints[self._steps]
            self._steps += 1
        else:
            waypoint = self._waypoints[self._end_steps]
            moving = False

        return waypoint, moving, turning

    def start(self, start: int, end: int) -> None:
        """ Start with input coordinates and calculate bearing """

        if (start < 0) and (end < 0):
            nav_pts = argwhere(self._wtr_map)
            rand_pts = choice(nav_pts.shape[0], 2, False)
            start, end = nav_pts[rand_pts[0]][::-1], nav_pts[rand_pts[1]][::-1]
            # start, end = nav_pts[0][::-1], nav_pts[-1][::-1]
            # start, end = array([127, 382]), array([382, 127])

        delta = end - start
        angle = arctan2(delta[1], delta[0])
        angle += (pi / 2)
        angle %= (2 * pi)

        self._start_coord[:] = start
        self._start_bearing = int(rint(angle / (((360 // self._sectors) * pi) / 180))) % self._compass.shape[1]

        # Add moving function to dict
        left, forward, right = self._action_keys.values()
        self._steering[left], self._steering[forward], self._steering[right] = self.__turn, self.__move, self.__turn

    def update(self, action: int) -> Tuple[bool, bool]:
        """ Perform action and record data """

        action_key = self._action_keys[1]  # Default forward

        if action in self._action_keys.keys():
            action_key = self._action_keys[action]  # Map DQN input to key
        elif action in self._action_keys.values():  # Take keyboard input
            action_key = action

        waypoint, moving, turning = self._steering[action_key](action_key)

        if moving:
            position, ortho_axes = self._ins.update(waypoint[1], waypoint[0])
            self._sonar.update(position, ortho_axes)
        else:
            self._ins._steps += 1

        return moving, turning

    def reset(self) -> None:
        """ Start in initial state and calibrate INS """

        self._ins.reset()
        self._sonar.reset()
        self.start(-1, -1)
        self._ins.hot_start(*self._start_coord)
        self._bearing = self._start_bearing
        self.__update_bearing()

    @property
    def data_plot(self) -> ndarray:
        """ Return data plot """
        return self._data_plot

    @property
    def step_map(self) -> ndarray:
        """ Return step map """
        return self._step_map

    @property
    def axes_map(self) -> ndarray:
        """ Return axes map """
        return self._axes_map

    @property
    def sensor_map(self) -> ndarray:
        """ Return sensor map """
        return self._sensor_map

    @property
    def sample_map(self) -> ndarray:
        """ Return sample map """
        return self._sample_map

    @property
    def survey_rgb(self) -> ndarray:
        """ Return survey RGB """
        return self._survey_rgb

    @property
    def swath_pov(self) -> ndarray:
        """ Return swath pov """
        return self._swath_pov

    @property
    def ins(self) -> INS:
        """ Return INS object """
        return self._ins

    @property
    def sonar(self) -> SONAR:
        """ Return SONAR object """
        return self._sonar

    @property
    def action_keys(self) -> Tuple[int, ...]:
        """ Return steering action keys """
        return tuple(self._action_keys)


class Env:
    """ Class to manage Bathymetry environment. Similar to a gym environments """

    __slots__ = ("_maps", '_device', "_architecture", '_training', 'max_steps', '_GIS', '_ASV', '_obs_shape',
                 '_obs_history', '_obs_rgb', '_obs_grey', '_obs_seq', '_show_aux_plots', '_edges', '_aux_plots',
                 '_captions', '_crash')

    def __init__(self, args) -> None:
        """
        Bathymetric Environment Initialization

        :param ? args:          Input arguments

        """

        # Check if inputs are valid
        if args.image_file == '':
            raise ValueError('No image file is specified')

        self._maps = [file for file in Path(args.image_file).iterdir()]

        img_path = self._maps[choice(len(self._maps))]
        if not img_path.exists():
            raise ValueError('No image file exists')
        image = imread(str(img_path), IMREAD_GRAYSCALE)

        image_shape = (512, 512)
        if image.shape != image_shape:
            raise ValueError('Image shape must be 512px by 512px')
        if (args.max_depth <= 0) or (args.max_depth > 100):
            raise ValueError('Max_depth invalid: Should 1 - 100')
        if (args.sensor_range <= 4) or (args.sensor_range > 100):
            raise ValueError('Sensor_range invalid: Should be 5 - 100')
        if (args.sensor_angle <= 0) or (args.sensor_angle > 180):
            raise ValueError('Sensor_angle invalid: Should 1 - 180')
        if (args.history_length <= 0):
            raise ValueError('History invalid: Should be above 1')

        self._device = args.device
        self.max_steps = args.max_episode_length
        self._training = True
        self._crash = 0

        # GIS and ASV instances
        self._GIS = GIS(image, args.max_depth, args.sensor_range)
        self._ASV = ASV(self._GIS, args.sensor_range, args.sensor_angle, compass_sectors=12)

        # Agent observation buffers
        if args.architecture in ('canonical', 'data-efficient'):
            self._obs_shape, self._obs_history = (64, 64), args.history_length
        else:
            self._obs_shape, self._obs_history = (128, 128), args.history_length

        data_images = zeros((4,) + image.shape, 'uint8')
        self._obs_rgb, self._obs_grey = data_images[0:3].transpose(1, 2, 0), data_images[3]
        data_images.flags.writeable = False

        obs_image = zeros((self._obs_history,) + self._obs_shape, 'uint8')
        self._obs_seq = obs_image[:]
        obs_image.flags.writeable = False

        # Create display for auxiliary information
        show_plots = args.auxiliary_plots
        self._show_aux_plots = show_plots

        if self._show_aux_plots:
            self._aux_plots = self._ASV.data_plot.copy()

            dims = image.shape[0] + 1
            edge_array = self._GIS.edge_array
            edges = zeros((edge_array.shape[0], 4 * edge_array.shape[1]), dtype='int32')
            concatenate((edge_array[0], edge_array[0], edge_array[0] + dims, edge_array[0] + dims), out=edges[0])
            concatenate((edge_array[1], edge_array[1] + dims, edge_array[1], edge_array[1] + dims), out=edges[1])
            self._edges = tuple(edges)

            # Info to display
            self._captions = (
                ("Edge distance: {:6.2f}", self._ASV.ins.edge_dist, (int(dims * 0.1), 20)),
                ("Total distance: {}", self._ASV.ins.steps, (int(dims * 0.6), 20)),
                ("Heading: {:6.2f}", self._ASV.ins.heading, (int(dims * 1.5), 20)),
                ("Depth: {:6.2f}", self._ASV.sonar.depth, (int(dims * 0.1), int(dims * 1.04))),
                ("Swath Width: {:6.2f}", self._ASV.sonar.swath_width, (int(dims * 0.6), int(dims * 1.04))),
                ("Percent cover: {:6.2f}%", self._ASV.sonar.coverage, (int(dims * 1.05), int(dims * 1.04))),
                ("Percent overlap: {:6.2f}%", self._ASV.sonar.overlap, (int(dims * 1.55), int(dims * 1.04)))
            )

    def step(self, action: int) -> Tuple[Tensor, float, bool, dict]:
        """ Get action from key input and return observation """

        reward, done = 0., False

        moving, turning = self._ASV.update(action)
        info = {'moving': moving, 'turning': turning}

        percent_coverage = self._ASV.sonar.coverage[0] / 100
        percent_overlap = self._ASV.sonar.overlap[0] / 100
        steps = self._ASV.ins.steps[0]
        # step_tax = (steps / self.max_steps) ** 3

        # Calculate reward
        reward += percent_coverage * (1 - percent_overlap)  # - step_tax * percent_overlap

        if not moving:
            self._crash += 1
            reward -= self._crash / 6

        if (percent_coverage >= 0.95) or (steps > self.max_steps) or self._crash > 6:
            done = True
            self._crash = 0

        observation, shape, history,  = self._obs_seq, self._obs_shape, self._obs_history
        rgb_buffer, grey_buffer = self._obs_rgb, self._obs_grey

        # Construct observation using GIS and sensory data
        rgb_buffer[..., 0] = self._ASV.sample_map
        rgb_buffer[..., 1] = self._GIS.nav_rgb[..., 1]
        rgb_buffer[..., 2] = 0
        rgb_buffer[..., 2][tuple(self._ASV.sonar.arg_sensor)] = 255
        # rgb_buffer[..., 2][tuple(self._ASV.ins._axes[0])] = 255
        cvtColor(src=rgb_buffer, dst=grey_buffer, code=COLOR_BGR2GRAY)

        # cvtColor(self._ASV.survey_rgb, COLOR_BGR2GRAY, grey_buffer)

        # Cycle observation window and append new step
        observation[0:history - 1] = observation[1:history]
        resize(grey_buffer, shape, observation[history - 1], interpolation=INTER_AREA)

        return torch.tensor(observation, dtype=torch.float32, device=self._device).div_(255), reward, done, info

    def render(self) -> int:
        """ Display screen of current state """

        # Write data to auxiliary plots and display image
        if self._show_aux_plots:
            self._aux_plots[:] = self._ASV.data_plot
            self._aux_plots[self._edges] = 127

            for c in self._captions:
                putText(self._aux_plots, c[0].format(c[1][0]), c[2], FONT_HERSHEY_SIMPLEX, 0.5, 127, 2)

            imshow("Observation Plot", self._obs_seq.max(0))
            imshow("Auxiliary Plots", self._aux_plots)
            imshow("Swath Plot", self._ASV.swath_pov)

        # Display survey plot image
        imshow("Survey Plot", self._ASV.survey_rgb)

        # Display Screen and read key input
        action_key = 0xFF & waitKey(1)

        return action_key

    def reset(self) -> Tensor:
        """ Start in a random state and calibrate INS """
        self._ASV.reset()
        for i in range(self._obs_history):
            observation, *_ = self.step(choice(3))
        return observation

    def make(self, random: bool = True) -> None:
        """ Create display and record initial bearing """

        # Open Survey display
        namedWindow("Survey Plot", WINDOW_GUI_NORMAL)
        moveWindow("Survey Plot", 0, 0)

        # Open Auxiliary display
        if self._show_aux_plots:
            namedWindow("Observation Plot", WINDOW_GUI_NORMAL)
            moveWindow("Observation Plot", 128, 0)
            namedWindow("Auxiliary Plots", WINDOW_GUI_NORMAL)
            moveWindow("Auxiliary Plots", 702, 0)
            namedWindow("Swath Plot", WINDOW_GUI_NORMAL)
            moveWindow("Swath Plot", 0, 542)

        if random:
            # self._ASV.start(-1, -1)
            self._ASV.reset()
        else:
            def start(event: int, x: int, y: int, flags: int, start_params) -> None:
                """ Capture user input for initial bearing of agent ASV """

                if event == EVENT_LBUTTONDOWN:
                    if start_params['start'] is None:
                        if start_params['nav_map'][y, x]:
                            start_params['start'] = array([x, y])
                    elif start_params['end'] is None:
                        if start_params['nav_map'][y, x]:
                            start_params['end'] = array([x, y])
                    if (start_params['start'] is not None) and (start_params['end'] is not None):
                        start_params['ASV'].start(start_params['start'], start_params['end'])

                        # Remove start function and continue run
                        setMouseCallback('Survey Plot', lambda *args: None)
                        start_params['flag'][:] = True

            params = {
                'flag': array([False]),
                'nav_map': self._GIS.nav_rgb[..., 0].view('bool'),
                'ASV': self._ASV,
                'start': None,
                'end': None
            }

            # Use start function to get user input for ASV initialization
            setMouseCallback('Survey Plot', start, params)

            while not params['flag']:
                _ = self.render()

    def train(self) -> None:
        """ Set training flag to true """
        self._training = True
        return None

    def eval(self) -> None:
        """ Set training flag to false """
        self._training = False
        return None

    @staticmethod
    def close() -> None:
        """ Close all windows """

        destroyAllWindows()
        destroyAllWindows()
        waitKey(1)
        return None

    def show_map(self) -> None:
        """ Display survey area """

        # Draw to display
        dim = self._GIS.image.shape[0]
        quads = (index_exp[0:dim, 0:dim], index_exp[0:dim:, (dim + 1):(2 * dim + 1)],)
        screen = zeros((dim, (dim * 2 + 1), 3), 'uint8')

        screen[:, dim] = 127
        screen[quads[0]] = self._GIS.nav_rgb
        screen[quads[0]][tuple(self._GIS.edge_array)] = (0, 0, 255)
        screen[quads[1]] = self._GIS.depth_rgb

        # Raster features
        nav_map = self._GIS.nav_rgb[..., 0].view('bool')
        map_area, depth_min, depth_mean, depth_max = (
            nav_map.sum(),
            int(self._GIS.depth_map[nav_map].min()),
            int(self._GIS.depth_map[nav_map].mean()),
            int(self._GIS.depth_map[nav_map].max())
        )

        # Write text to display
        captions = (
            (f'Total survey area: {map_area} pixels', (int(dim * 0.2), 20)),
            (f'Depth units (min, mean, max): {(depth_min, depth_mean, depth_max)}', (int(dim * 1.2), 20))
        )
        for c in captions:
            putText(screen, c[0], c[1], FONT_HERSHEY_DUPLEX, 0.5, (64, 64, 64), 2)

        # Display images and escape clause
        imshow('GIS map', screen)
        print('\nUse esc key to exit')

        while True:
            if (waitKey(0) & 0xFF) == 27:
                destroyWindow('GIS map')
                waitKey(1)
                break

        return None

    @property
    def actions(self) -> Tuple[int, ...]:
        """ Return tuple of action keys """
        return self._ASV.action_keys

    @property
    def action_space(self) -> int:
        """ Return number of action keys """
        return len(self._ASV.action_keys)

    @property
    def observation_space(self) -> Tuple[Tuple[int, int], int]:
        """ Return observation buffer shape tuple """
        return self._obs_shape, self._obs_history

    @property
    def percentage_cover(self) -> float:
        """ Return percentage cover of area """
        return self._ASV.sonar.coverage

    ...
