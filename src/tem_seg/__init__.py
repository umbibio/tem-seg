import os

from .config import settings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = settings.environment.tf_cpp_min_log_level


__version__ = "0.3.1.dev1"
