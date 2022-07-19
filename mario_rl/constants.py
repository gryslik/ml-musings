import os
import pathlib

model_name = "mario"

# Paths
base_path = os.path.expanduser(pathlib.Path().absolute())
model_path = os.path.expanduser(base_path + "/models/")
fail_path = os.path.expanduser(base_path + "/episodes/failures/")
success_path = os.path.expanduser(base_path + "/episodes/successes/")
video_path = os.path.expanduser(base_path + "/episodes/videos/")
log_path = os.path.expanduser(base_path+"/training_log.txt")
# log filename
log_fn = "training_log.txt"