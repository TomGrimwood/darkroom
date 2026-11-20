"""The sensors on a stage, the frames rigs of them go on, and what the sensors report.

    sensorkit.frames(stage)                       # what a stage offers to attach to
    sensorkit.sensors(stage)                      # prims usdsensors marks as sensors
    sensorkit.imu.samples(motion, 200)            # from a pose sampled over time
    sensorkit.sky.sky_share(view, rotation)             # how much of an antenna's view is sky

Sensors and their rigs are files, read and written with `usdsensors`; a rig is attached under
a frame (any Xform in a `Frames` scope, whoever declares it; a carrier's rigs go on its
`SensorMount`). Knows nothing of environments, characters or trajectories.
"""

from . import camera, gnss, imu, lidar, motion, site, sky
from .frames import FRAMES, SENSOR_MOUNT, asset_path, frames, identifier
from .kinds import DEBUG, KINDS, RENDERED, Sensor, kind_of, mark, sensors

__all__ = [
    "DEBUG",
    "FRAMES",
    "KINDS",
    "RENDERED",
    "SENSOR_MOUNT",
    "Sensor",
    "asset_path",
    "camera",
    "frames",
    "gnss",
    "identifier",
    "imu",
    "kind_of",
    "lidar",
    "mark",
    "motion",
    "sensors",
    "site",
    "sky",
]
