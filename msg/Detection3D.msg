# One object seen by the camera. Position in <robot_name>/base_link, lengths in
# metres, angles in radians — except bbox, which is in image pixels.
#
# Deliberately carries no "is this the target" flag: identity is decided by
# the world model, not by perception.
string class_name              # raw detector label, lowercase
float64 confidence
int32 track_id                 # -1 = untracked
float64[4] bbox                # x1 y1 x2 y2 (pixels)
float64 depth                  # median depth of the bbox centre region; 0 = no valid depth
geometry_msgs/Point position   # object centre
geometry_msgs/Vector3 size     # x = depth extent, y = width, z = height
float64 ray_yaw                # YOLO3D viewing-ray orientation prior, REFINED by
                               # detection_shim.py's estimate_grasp_orientation
                               # (contour + symmetry + 180-deg flip resolution)
                               # before publishing — not the raw theta_ray alone.
