import numpy as np
import pandas as pd
from scipy.interpolate import interpn
from scipy.integrate import RK45
from datetime import date
from argparse import ArgumentParser
import netCDF4 as nc
import time
warnings.filterwarnings("ignore")



# Global variables
domain = np.array([-6, 36.25, 30.9, 45.9375]) #Coordinates of the boundaries of the simulation domain - MedSea, for cropping the CMEMS fields
mindepth = 0 #Minimum depth of the velocity fields
maxdepth = 1 #Maximum depth of the velocity fields


m2deg=1e-3/111.19 #Meters to degrees. Conversion factor for the velocity
sec2day=60*60*24 #Seconds to day. Conversion factor for the velocity

survival = 250 #Advection duration - 5*lambda (50 days)


# Import
# Vectors describing the geometry of the velocity fields. Coordinates of the vertexes of the oceanographic grid
lon = np.genfromtxt('lon.csv', delimiter=',')
lat = np.genfromtxt('lat.csv', delimiter=',')
depth = np.genfromtxt('depth.csv', delimiter=',')

de1 = np.flatnonzero(depth<=mindepth)[-1]
de2 = np.flatnonzero(depth>maxdepth)[0]+1

X = lon
Y = lat
Z = depth[de1:de2]

# Setting/importing particle starting positions 
coasts = pd.read_csv(r'/Simulations/Scripts/Sources_coasts.csv', delimiter=';')

no_part = (sum(coasts['particles'])) # Number of particles per source type
nsources = len(coasts)



class Advection:
    """Definition of the class Advection for ODE integration
    
    -- Parameters --
    
    X,Y,Z : Geographic coordinates of the vertexes of the grid. X is a NumPy array containing the longitudes,
        Y contains the latitudes while Z the depth (in meters) of each vertex.
        
    Ugrid : Masked NumPy array with dimensions (len(Y),len(X),len(Z)) containing the zonal velocities.

    Vgrid : Masked NumPy array with dimensions (len(Y),len(X),len(Z)) containing the meridional velocities.
    
    nsites : Integer equal to the number of particles released.
    
    z : NumPy array with depth of each particle
    
    -- __call__ method --
    Builds the callable for ODE integration.
    
    Variables:
    
    x,y : NumPy arrays with the initial longitudes (x) and latitudes (y) of each particle.
    points : NumPy array with dimensions (nsites, 3) containing initial lat, lon and depth of each particle
    u,v : NumPy arrays with zonal (u) and meridional (v) velocities, interpolated at each particle position 
      at each function call (so as to have a realistic representation of a "continuous" particle advection)
    
    -- return --
    Returns the RHS of the ODEs for each particle.
    
    
    """ 
    def __init__(self,X,Y,Z,Ugrid,Vgrid,nsites,z):
        self.Ugrid = Ugrid
        self.Vgrid = Vgrid
        self.X = X
        self.Y = Y
        self.Z = Z
        self.nsites = nsites
        self.z = z
    def __call__(self,t,y):
        nsites = self.nsites
        x,y = y[0:(nsites)],y[(nsites):]
        z = self.z
        Ugrid,Vgrid = self.Ugrid,self.Vgrid
        X,Y,Z = self.X,self.Y,self.Z
       
        points = np.array([x,y,z]).T
        
        u = interpn((X,Y,Z),Ugrid,points,'linear',bounds_error=False, fill_value=0)
        v = interpn((X,Y,Z),Vgrid,points,'linear',bounds_error=False, fill_value=0)

        return np.concatenate((u, v), axis=0)




if __name__ == "__main__": 
    
    p = ArgumentParser(description="""choose starting and ending dates""")
    p.add_argument('-ye_st', choices = ('2015','2016'), action="store", dest="ye_st", help='start year for the run')
    p.add_argument('-mo_st', choices = ('01','02','03','04','05','06','07','08','09','10','11','12'), action="store", dest="mo_st", 
                   help='start month for the run')
    p.add_argument('-da_st', action="store", dest="da_st", help='start day for the run')
    p.add_argument('-ye_end', choices = ('2015','2016'), action="store", dest="ye_end", help='end year for the run')
    p.add_argument('-mo_end', choices = ('01','02','03','04','05','06','07','08','09','10','11','12'), action="store", dest="mo_end", 
                   help='end month for the run')
    p.add_argument('-da_end', action="store", dest="da_end", help='end day for the run')
    
    
    args = p.parse_args()
    ye_st = int(args.ye_st)
    mo_st = int(args.mo_st)
    da_st = int(args.da_st)
    ye_end = int(args.ye_end)
    mo_end = int(args.mo_end)
    da_end = int(args.da_end)

    start_day = date.toordinal(date(ye_st,mo_st,da_st))
    end_day = date.toordinal(date(ye_end,mo_end,da_end))
    duration = end_day - start_day +1

    
    start = time.time()

    for days in range(duration):
        start_day = date.fromordinal(day1+days)
    
        filesave = r'/Simulations/Coste_py/'+str(start_day.year)+'/'+str(start_day.month)+'/'+str(start_day.year)+str(start_day.month)+str(start_day.day)+'.npy'
    
        starting = np.zeros([no_part,4]) #Country, coord_x, coord_y, depth

        for cc in range(nsources):
            
            # A prescribed number of particles is released from each source with a random displacement in a 100 m radius from it
            starting[sum(coasts['particles'][0:cc]):sum(coasts['particles'][0:cc+1]),1] = np.ones([coasts['particles'][cc]])*coasts['coord_x'][cc]+np.random.rand(coasts['particles'][cc])*0.0012658
            starting[sum(coasts['particles'][0:cc]):sum(coasts['particles'][0:cc+1]),2] = np.ones([coasts['particles'][cc]])*coasts['coord_y'][cc]+np.random.rand(coasts['particles'][cc])*0.0012658
            starting[sum(coasts['particles'][0:cc]):sum(coasts['particles'][0:cc+1]),3] = np.random.rand(coasts['particles'][cc])


            #for cc in range(nsources):

               #start_df['Country'][sum(coasts['particles'][0:cc]):sum(coasts['particles'][0:cc+1])] = [coasts['Country'][cc]]*coasts['particles'][cc]

        x0 = starting[:,1].copy()
        y0 = starting[:,2].copy()
        z0 = starting[:,3].copy()

        nsites = x0.shape[0]
    
        xt = np.zeros(no_part) #Latitudes of initial particle positions. Particles' new x-coordinates
        # will be appended along the columns

        yt = np.zeros(no_part) #Longitudes of initial particle positions. Particles' new y-coordinates
        # will be appended along the columns

        xt=x0.reshape(no_part,1) # IC
        yt=y0.reshape(no_part,1) # IC

        y_CI = np.array([xt,yt]).flatten() #IC for ODE solver

        timestart = time.time() # Comment if not interested to look into the computational time of the whole simulation
    
        for i in range(survival): # Advection loop
        
            today = date.fromordinal(date.toordinal(start_day)+i)

            if (today.year>=1987 and today.year<=2003):
                filename=r'/home/guerrini/medsea/fields_'+ str(today.year)+'/'+'%02d' % today.year +'%02d' % today.month+'%02d' % today.day+'_dm-INGV--RFVL-MFSs4b3-MED-b20130712_re-fv04.00.nc'
            elif (today.year>=2004 and today.year<=2013):
                filename=r'/home/guerrini/medsea/fields_'+ str(today.year)+'/'+'%02d' % today.year +'%02d' % today.month+'%02d' % today.day+'_dm-INGV--RFVL-MFSs4b3-MED-b20140620_re-fv05.00.nc'
            elif (today.year==2014):
                filename=r'Simulations/fields_'+ str(today.year)+'/'+'%02d' % today.year +'%02d' % today.month+'%02d' % today.day+'_dm-INGV--RFVL-MFSe1r1-MED-b20160115_re-fv06.00.nc'
            elif (today.year==2015):
                filename=r'/home/guerrini/medsea/fields_'+ str(today.year)+'/'+'%02d' % today.year +'%02d' % today.month+'%02d' % today.day+'_dm-INGV--RFVL-MFSe1r1-MED-b20160501_re-fv07.00.nc'
            elif (today.year==2016):
                filename=r'/home/guerrini/medsea/fields_'+ str(today.year)+'/'+'%02d' % today.year +'%02d' % today.month+'%02d' % today.day+'_d-INGV--RFVL-MFSe1r1-MED-b20180115_re-sv04.10.nc'
            elif (today.year==2017):
                filename=r'/home/guerrini/medsea/fields_'+ str(today.year)+'/'+'%02d' % today.year +'%02d' % today.month+'%02d' % today.day+'_d-INGV--RFVL-MFSe1r1-MED-b20190101_re-sv04.10.nc'
            else:
                filename=r'/home/guerrini/medsea/fields_'+ str(today.year)+'/'+'%02d' % today.year +'%02d' % today.month+'%02d' % today.day+'_d-CMCC--RFVL-MFSe1r1-MED-b20190601_re-sv05.00.nc'
    
           # This part reads CMEMS fields. Note that filenames change depending on the year they have been released
            
            U = nc.Dataset(filename).variables['vozocrtx']
            V= nc.Dataset(filename).variables['vomecrty']

    
            Ugrid = U[0,:,:,:]*m2deg*sec2day
            Ugrid=np.transpose(Ugrid, (1,2,0))[:,:,de1:de2]#[la1:la2,lo1:lo2,de1:de2]
            Ugrid=np.transpose(Ugrid, (1,0,2))
            Ugrid.data[Ugrid.data>=2]=0 # Removing unrealistic values/no data placeholders

            Vgrid = V[0,:,:,:]*m2deg*sec2day
            Vgrid=np.transpose(Vgrid, (1,2,0))[:,:,de1:de2]#[la1:la2,lo1:lo2,de1:de2]
            Vgrid=np.transpose(Vgrid, (1,0,2))
            Vgrid.data[Vgrid.data>=2]=0 # Removing unrealistic values/no data placeholders
   

            f = Advection(X,Y,Z,Ugrid,Vgrid,nsites,z0)
    
            t0 = 0
            t1 = 1


            ode = RK45(f,t0,y_CI,t1)

    
            while ode.status == 'running':
                ode.step()

    
          # Appending new particle locations
            xt = np.append(xt,ode.y[:nsites].reshape(nsites,1),axis=1)
            yt = np.append(yt,ode.y[nsites:].reshape(nsites,1),axis=1)
    
          #New ICs for the next iteration
            y_CI = np.array([xt[:,-1],yt[:,-1]]).flatten()
        

    

        np.save(filesave,np.array([xt,yt]).T)   
        timeend = time.time()
        print('Lagrangian simulation...'+str((days+1)*100/duration)+'% done. Day '+str(days+1)+' done in '+str(timeend-timestart)+'s')
    
    
    end = time.time() # Comment if not interested to look into the computational time of the whole simulation
    delta_time = end-start # Comment if not interested to look into the computational time of the whole simulation

    
    print('Duration of the computation: ',delta_time,'s')

