Profiles using command line
===========================

For the examples here, simply copy and paste the command line in your
shell, working in the directory where the StagYY par file is located. 
You can also use the examples on the data available in the Examples
directory. 


Profiles are accessed using the rprof command::

    % stagpy rprof -s 4:6

In this example, mean temperature profiles of snapshot 4 and 5 are
plotted on two graph.

::

    % stagpy rprof -s 4:6 +a

plots the average radial profiles of snapshots 4 and 5.

::

    % stagpy rprof -o Tmin,Tmean,Tmax.vzabs,vhrms -t 500:

plots all temperature and velocity profiles that have been saved starting from
time-step 500. The list of variables you want follow the same logic as time
series variables.

::

    % stagpy rprof +g -o

plots grid spacing profile for the last snapshot available. The ``-o`` flag
turns off output of other radial profiles (``Tmean`` by default).
