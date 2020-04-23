# OPD100 profile reading example
Python based OPD100 profile reading example

This is a sample code to provide an example how to read the reference profile and live profile of the OPD100.

This code is reduced to it's bare minimum and does not cover all edge cases
one would expect in production environments.

# Features
- Loading of reference profile
- Loading of live profile
- Loading only z-values of live profile
- Plotting results to .png file
- Saving results as .json file

All features can be enabled/disabled via global variables in the script.


# Requirements
The script is compatible with both Python2 and Python3.

If plotting is desired, you also need to have matplotlib installed
```
python -mpip install matplotlib
```


# Usage Example
```
python opd100_loader.py --ip 192.168.0.34 --tcp-port 80 --device-port 4
```


# License
> SPDX-License-Identifier: Apache-2.0
>
> Copyright (C) ifm electronic gmbh 
>
> THE PROGRAM IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND.
