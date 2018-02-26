

import regrid_swaths as rs
from datetime import datetime

latres=1.0
lonres=1.0

rs.make_gridded_swaths(datetime(2005,1,1),latres=latres,lonres=lonres)
