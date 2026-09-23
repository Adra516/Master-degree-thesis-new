"""Gazebo ground truth, for measurement only. Every query returns None when
Gazebo is not there (a real robot, or a sim without the service), so nothing
may depend on it for behaviour.

Ported near-verbatim from rework-attempt-1's src/locobot_prog/ros/
ground_truth.py, using this project's own rotations.py (already ported for
arm_diagnostics.py in step 3 — same world_to_body full-orientation fix).
"""
import rospy

import rotations


class GroundTruth:

    def __init__(self, robot_name="locobot", timeout=2.0):
        self.robot = robot_name
        self._srv = None
        try:
            from gazebo_msgs.srv import GetModelState
            rospy.wait_for_service("/gazebo/get_model_state", timeout=timeout)
            self._srv = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)
        except (rospy.ROSException, ImportError):
            pass

    @property
    def available(self):
        return self._srv is not None

    def object_in_base(self, model):
        """(x, y, z) of a Gazebo model's origin in the robot's base frame, or
        None. For the table objects the origin is their base, on the axis.
        Uses the robot's full orientation (see rotations.world_to_body)."""
        if self._srv is None or not model:
            return None
        try:
            r, o = self._srv(self.robot, ""), self._srv(model, "")
        except rospy.ServiceException:
            return None
        if not (r.success and o.success):
            return None
        p, q = r.pose.position, r.pose.orientation
        return tuple(float(v) for v in rotations.world_to_body(
            (p.x, p.y, p.z), (q.x, q.y, q.z, q.w),
            (o.pose.position.x, o.pose.position.y, o.pose.position.z)))
