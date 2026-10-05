# Everything the world model knows about where the target is. Each source is
# reported separately; the caller decides which to trust (see
# target_state.measured_position). <robot_name>/base_link, metres, seconds.
string target_name
string catalog_key               # "" = the name is not in the catalogue
bool locked                      # a detection of it has been accepted
float64 lock_age                 # seconds since the last accepted detection
geometry_msgs/Point lock_position
geometry_msgs/Vector3 lock_size  # x = depth, y = width, z = height
float64 lock_orientation         # refined yaw (detection_shim.py), radians —
                                 # meaningless (0.0) for a catalogued round object
int32 recent_count               # accepted detections in the last 1.5 s
geometry_msgs/Point recent_median
bool in_frame                    # it is in the latest camera frame
geometry_msgs/Point frame_position
geometry_msgs/Vector3 frame_size
float64 frame_orientation         # same, from the current frame's own detection
float64[4] frame_bbox            # its detection box in that frame, PIXELS [x1, y1, x2, y2]
time frame_stamp                 # the camera frame it was detected in
bool map_valid                   # the lock reprojects from the map frame
geometry_msgs/Point map_position
bool calibrated                  # enough far-vantage size samples
geometry_msgs/Vector3 calibrated_size
