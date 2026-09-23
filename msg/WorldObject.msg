# One object in the world model, in <robot_name>/base_link. Metres, radians.
string name                   # detector label
string catalog_key            # catalogue object it is; "" = not catalogued / ambiguous
bool is_target
geometry_msgs/Point position
geometry_msgs/Vector3 size    # x = depth, y = width, z = height
float64 distance              # horizontal distance from the base
float64 yaw
string grasp_axis             # along_x | along_y: the face the jaws would close across
bool gripper_fits
bool in_arm_reach
