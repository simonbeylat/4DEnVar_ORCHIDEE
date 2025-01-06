import warnings
warnings.filterwarnings("ignore")
from FourDEnVar_simplify import FourDEnVAR

path='../data/VCMAX/output.nc'
print('Simple Case')

OBJ=FourDEnVAR(path_netcdf=path,natmsite=1,size=100)
find_min, xa=OBJ.do_iteration()
print('____________________________________________________________')
print('Complexe Case')
path='../data/5P/output.nc'
OBJ=FourDEnVAR(path_netcdf=path,natmsite=1,size=300)
find_min, xa=OBJ.do_iteration()
