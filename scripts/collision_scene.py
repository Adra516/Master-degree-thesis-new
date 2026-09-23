"""The MoveIt collision scene for one grasp.

Ported near-verbatim from rework-attempt-1's src/locobot_prog/arm/
collision_scene.py ("moved from locobot_motion_node.py"). Like cobot's
addCollisionObject(), scene building is kept out of the grasp sequence — the
boxes themselves are computed in grasp_geometry.py (table_box, target_box,
obstacle_boxes); this class only puts them into MoveIt and takes them out
again.

Two-phase semantics: during the pre-grasp EVERY object is an obstacle,
including the target; before the straight-in grasp the target is removed so
the jaws may close around it.
"""
import rospy
from geometry_msgs.msg import PoseStamped

TARGET_OBSTACLE = "target_obstacle"   # the target while it is still an obstacle
ATTACHED_OBJECT = "target_object"     # the target once it is in the gripper


class CollisionScene:

    def __init__(self, robot_name, scene):
        self.ns = robot_name
        self.scene = scene                 # moveit_commander.PlanningSceneInterface
        self._obstacles = []               # names of the table + neighbour boxes
        self._target_in_scene = False

    def _pose(self, x, y, z):
        pose = PoseStamped()
        pose.header.frame_id = f"{self.ns}/base_link"
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = x, y, z
        pose.pose.orientation.w = 1.0
        return pose

    def _add(self, box):
        """box: a dict {name, x, y, z, sx, sy, sz} (grasp_geometry's box shape)."""
        self.scene.add_box(box["name"], self._pose(box["x"], box["y"], box["z"]),
                          size=(box["sx"], box["sy"], box["sz"]))

    def set_obstacles(self, boxes):
        """Replace the obstacle set (table first, then neighbouring objects).

        Added BEFORE the arm moves, so the first plan does not start in
        collision. The target is not in this list: see add_target()."""
        if self._obstacles:
            self._remove_obstacles()
            rospy.sleep(0.15)
        for box in boxes:
            self._add(box)
            self._obstacles.append(box["name"])
            rospy.loginfo("[arm] Obstacle %-16s centre=(%.3f,%.3f,%.3f) size=(%.3f,%.3f,%.3f)",
                          box["name"], box["x"], box["y"], box["z"],
                          box["sx"], box["sy"], box["sz"])
        rospy.sleep(0.30)

    def _remove_obstacles(self):
        for name in self._obstacles:
            self.scene.remove_world_object(name)
        self._obstacles = []

    def add_target(self, box):
        """The target as an obstacle, for the pre-grasp phase."""
        self._add(box)
        self._target_in_scene = True
        rospy.loginfo("[arm] Target-as-obstacle added (pre-grasp): xy=(%.3f,%.3f) "
                      "foot=%.3f h=%.3f", box["x"], box["y"], box["sx"], box["sz"])

    def remove_target(self):
        """Release the target from collision (it becomes the grasp target)."""
        if self._target_in_scene:
            self.scene.remove_world_object(TARGET_OBSTACLE)
            self._target_in_scene = False

    def attach(self, x, y, z, size_xyz):
        """Attach the grasped object to the gripper, so later moves plan around it."""
        self.scene.attach_box(
            f"{self.ns}/ee_gripper_link", ATTACHED_OBJECT,
            pose=self._pose(x, y, z), size=size_xyz,
            touch_links=[f"{self.ns}/left_finger_link",
                         f"{self.ns}/right_finger_link",
                         f"{self.ns}/gripper_link"])

    def detach(self):
        self.scene.remove_attached_object(f"{self.ns}/ee_gripper_link", name=ATTACHED_OBJECT)

    def clear(self):
        """Tear down everything this grasp added. remove_attached_object only
        DETACHES (leaving the box in the world at the gripper's last pose), so
        the world object is deleted too — a stale box would otherwise trip the
        collision-aware pre-grasp of the NEXT grasp."""
        self.detach()
        self.scene.remove_world_object(ATTACHED_OBJECT)
        self._remove_obstacles()
        self.remove_target()
