from asyncio import base_events
import os
import pathlib

model_name = "eagle-large"

# Paths
base_path = os.path.expanduser(pathlib.Path().absolute())
model_path = os.path.expanduser(base_path + "/" + model_name)
fail_path = os.path.expanduser(base_path + "/episodes/failures/")
success_path = os.path.expanduser(base_path + "/episodes/successes/")
video_path = os.path.expanduser(base_path + "/episodes/videos/")