# An axis-aligned box in the MoveIt collision scene, in <robot_name>/base_link.
# Object orientation is unknown from the camera, so boxes are never rotated;
# callers pad the footprint instead.
string name
geometry_msgs/Point center
geometry_msgs/Vector3 size      # full extents (m)
