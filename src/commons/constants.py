DATA_PATH = r"data"

import sys
DIR_SEP = ''
if sys.platform == "linux" or sys.platform == "linux2" or sys.platform == "darwin":
    DIR_SEP = '/'
else:
    # Windows
    DIR_SEP = '\\'
