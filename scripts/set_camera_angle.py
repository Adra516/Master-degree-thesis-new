#!/usr/bin/env python3
"""
set_camera_angle.py
-------------------
Publishes a single pan/tilt command after a delay, then exits.
Runs once at launch to point the camera toward the table.
"""
import rospy
from std_msgs.msg import Float64

rospy.init_node("set_camera_angle")
robot_name = rospy.get_param("~robot_name", "locobot")
pan   = float(rospy.get_param("~pan",   0.0))
tilt  = float(rospy.get_param("~tilt",  0.9))
delay = float(rospy.get_param("~delay", 4.0))

pan_pub  = rospy.Publisher(f"/{robot_name}/pan_controller/command",  Float64, queue_size=1, latch=True)
tilt_pub = rospy.Publisher(f"/{robot_name}/tilt_controller/command", Float64, queue_size=1, latch=True)

rospy.loginfo("[camera] Waiting %.1fs then setting pan=%.2f tilt=%.2f", delay, pan, tilt)
rospy.sleep(delay)

pan_pub.publish(Float64(data=pan))
tilt_pub.publish(Float64(data=tilt))
rospy.loginfo("[camera] Camera angle set: pan=%.2f tilt=%.2f", pan, tilt)

rospy.sleep(1.0)  # let latched message propagate
