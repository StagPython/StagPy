Snapshots using command line
============================

For the examples here, simply copy and paste the command line in your
shell, working in the directory where the StagYY par file is located. 
You can also use the examples on the data available in the Examples
directory. 

The next use of stagpy is to create images of snapshots of the
different fields.

::

   % stagpy field -o T

will plot the last snapshot of the temperature field.

::

   % stagpy field -s : -o T,v3

will plot all the snapshots of the temperature and the vertical/radial velocity
on separate figures.

::

   % stagpy field -s 3:8:2 -o T+stream

will plot the temperature field with isocontours of the stream function from
the third to the eighth snapshot, every two snapshots.

::

   % stagpy field -o T -s 1:5 --vmin=0.8 --vmax=1.0
   
will plot the temperature field from the first to the fifth snapshot while keeping the range of the colorbar fixed between 0.8 and 1. 
