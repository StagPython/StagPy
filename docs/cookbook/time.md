Time series using command line
==============================

For the examples here, simply copy and paste the command line in your
shell, working in the directory where the StagYY par file is located.
You can also use the examples on the data available in the Examples
directory.

The command

```sh title="shell"
stagpy time
```

will give you by default one figure with three subplots. The first subplot
contains the time series of the Nusselt number at the top and bottom
boundaries. The second subplot contains the time series of the rms velocity.
The third subplot is the time series of the mean temperature. This is
equivalent to typing

```sh title="shell"
stagpy time -o Nu_top,Nu_bot.Vrms.Tmean
```

The command

```sh title="shell"
stagpy time --tstart 0.02 --tend 0.03
```

will give you the same plots but starting at time 0.02 and ending at
time 0.03.

```sh title="shell"
stagpy time -o Vrms-Tmin,Tmean,Tmax.dTdt
```

creates two figures. The first one contains the time series of the rms
velocity. The second one contains two subplots, the first one with the time
series of the minimal, maximal and average temperature; the second one with the
time derivative of the mean temperature. The variable names can be found by
running the `stagpy var` command. The variables you want on the same
subplot are separated by commas `,`, the variables you want on different
subplots are separated by dots `.`, and the variables you want on different
figures are separated by dashes `-`.

```sh title="shell"
stagpy time +compstat --tstart 0.02
```

will create a file containing the average and standard deviation of the time
series variables from t=0.02 to the end. This is useful when your system has
reached a statistical steady state.
