"""Arm and gripper actuators: publish straight to the controllers' command topics.

Ported near-verbatim from rework-attempt-1's src/locobot_prog/arm/actuators.py,
which was itself "moved verbatim from locobot_motion_node.py (_send_arm,
_publish_arm_trajectory, _send_gripper and the settle-hold pattern)". Adapted
here only to drop the locobot_prog.* package prefix (this project uses flat
importable modules in scripts/, not a catkin_python_setup library).

GRASP_READY_POSE matches the FK-measured, horizontal-approach ready pose
already validated in locobot_motion_node.py this session (EE (0.309, 0.344),
retracted 86mm short of the nearest target's near face) — unchanged here.
"""
import rospy
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from grasp_geometry import FINGER_OPEN

ARM_JOINTS = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]

# Named joint configurations (6-DOF, ARM_JOINTS order).
CAMERA_CLEAR_POSE = [0.0, -1.20, 1.00, 0.0, 0.80, 0.0]
LIFT_POSE         = [0.0, -0.80, 1.00, 0.0, 0.50, 0.0]
GRASP_READY_POSE  = [0.0, -1.00, 1.25, 0.0, -0.25, 0.0]
ARM_POSES = {
    "camera_clear": CAMERA_CLEAR_POSE,
    "grasp_ready":  GRASP_READY_POSE,
    "lift":         LIFT_POSE,
}


class ArmActuator:

    def __init__(self, robot_name, move_group):
        self.move_group = move_group
        self.pub = rospy.Publisher(f"/{robot_name}/arm_controller/command",
                                   JointTrajectory, queue_size=1, latch=True)

    def send_joints(self, positions, move_time=2.0):
        """Joint-space move; blocks until it should have finished."""
        traj = JointTrajectory()
        traj.joint_names = ARM_JOINTS
        pt = JointTrajectoryPoint()
        pt.positions = positions
        pt.time_from_start = rospy.Duration(move_time)
        traj.points = [pt]
        self.pub.publish(traj)
        rospy.sleep(move_time + 0.2)

    def hold(self, move_time):
        """Settle-hold: re-command the arm's own MEASURED joint values.

        After an open-loop move the effort controllers keep chasing a goal they
        cannot fully reach under gravity (integral wind-up), so the arm keeps
        fighting and wobbles. Re-commanding the actual joints drives the
        position error to ~0 so it holds still.
        """
        self.send_joints(self.move_group.get_current_joint_values(), move_time)

    def publish_trajectory(self, robot_traj):
        """Run a MoveIt RobotTrajectory by publishing it straight to the arm
        controller's command topic, bypassing MoveIt's execute() (the
        FollowJointTrajectory ACTION interface).

        Why: the action interface enforces per-joint path tolerances and ABORTS
        the whole trajectory the moment forearm_roll lags past its tolerance —
        which the low-gain roll joint does on every descent in this sim. The
        command topic has no such monitoring: the controller chases the
        setpoints and forearm_roll settles as best it can. Wrist roll doesn't
        affect the gripper's XYZ position (only which way the fingers face,
        irrelevant for a roughly cylindrical object), so this trades a perfect
        roll angle for a move that actually completes.

        Returns the trajectory's nominal duration in seconds.
        """
        jt = robot_traj.joint_trajectory
        if not jt.points:
            return 0.0
        self.pub.publish(jt)
        return jt.points[-1].time_from_start.to_sec()


class Gripper:

    def __init__(self, robot_name):
        self.position = FINGER_OPEN        # left_finger joint position, from joint_states
        self.pub = rospy.Publisher(f"/{robot_name}/gripper_controller/command",
                                   JointTrajectory, queue_size=1, latch=True)
        rospy.Subscriber(f"/{robot_name}/joint_states", JointState,
                         self._on_joint_states, queue_size=1)

    def _on_joint_states(self, msg):
        if "left_finger" in msg.name:
            self.position = msg.position[msg.name.index("left_finger")]

    def send(self, position, move_time=1.0):
        """Command both fingers (the controller requires both; right mirrors left)."""
        traj = JointTrajectory()
        traj.joint_names = ["left_finger", "right_finger"]
        pt = JointTrajectoryPoint()
        pt.positions = [position, -position]
        pt.time_from_start = rospy.Duration(move_time)
        traj.points = [pt]
        self.pub.publish(traj)
        rospy.sleep(move_time + 0.2)
