
_iga='IJ_AVG_S__'
GC_trac_avg = { 'time':'taus','lev':'press','lat':'lats','lon':'lons',
    # IJ_AVGs: in ppbv, except isop (ppbC)
    _iga+'NO':'NO', _iga+'O3':'O3', _iga+'MVK':'MVK', _iga+'MACR':'MACR',
    _iga+'ISOPN':'isopn', _iga+'IEPOX':'iepox', _iga+'NO2':'NO2', _iga+'NO3':'NO3',
    _iga+'NO2':'NO2', _iga+'ISOP':'isop', _iga+'CH2O':'hcho',
    # Biogenic sources: atoms C/cm2/s
    'BIOGSRCE__ISOP':'E_isop_bio',
    # Other diagnostics:
    'PEDGE_S__PSURF':'psurf',
    'BXHGHT_S__BXHEIGHT':'boxH', # metres
    'BXHGHT_S__AD':'AD', # air mass in grid box, kg
    'BXHGHT_S__AVGW':'avgW', # Mixing ratio of H2O vapor, v/v
    'BXHGHT_S__N_AIR_':'N_air', # Air density: molec/m3
    'DXYP__DXYP':'area', # gridbox surface area: m2
    'TR_PAUSE__TP_LEVEL':'tplev',
    'TR_PAUSE__TP_HGHT':'tpH', # trop height: km
    'TR_PAUSE__TP_PRESS':'tpP', # trop Pressure: mb
    # Many more in trac_avg_yyyymm.nc, not read here yet...
    }
