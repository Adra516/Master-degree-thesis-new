# Everything the arm needs to execute one grasp, so the arm node derives no
# geometry of its own. Metres, radians, <robot_name>/base_link.
#
# Execution is two-phase: the arm first stops at a standoff `standoff_offset`
# back from grasp_point along the approach axis (same height, same
# orientation), then moves straight in.
bool valid
string reason                               # why it is invalid, or how it was chosen
geometry_msgs/Point grasp_point             # where the jaws close
geometry_msgs/Quaternion grasp_orientation  # level gripper, +x pointing at the object
float64 standoff_offset                     # pre-grasp distance back from grasp_point
float64 standoff_offset_min                 # closest the standoff may move in on a retry
                                            # while staying outside the target's collision box
float64 gripper_open                        # per-finger joint position for the approach
float64 gripper_close                       # per-finger joint position to close to
bool full_close                             # gripper_close is the full close (object width
                                            # at the grip height is not trusted)
geometry_msgs/Vector3 object_size           # x = depth, y = width, z = height; used for
                                            # the attached collision box
