# PyAspidistra
Fit density profiles to Pleiades members form the DANCe data set using emcee and MultiNest.

There are two main folders: emcee and MultiNest. Each folder cointains: the code for the density profiles, 
and the running master code to fit the data, this both these samplers. Thus, the folders contain:

* Main file runSpatial******.py. It controls the running using the models in its correspondinf "Models" folder.
* Models folder. It contains all models related to that particular flavour.

All our models use the membership probability of indivudual objects and the densities are truncated in the area of 
the DANCe survay.

The emcee folder contains models in which only the radial information is inferred. This is only the parameters of the different profiles.
The MultiNest folder contains different options for models: only radial profile, radial profile + centre, radial profile + centre + ellipticity. There is also Kent distribution and Kent+profiles.

Each runSpatial***.py must be called with two arguments: the name of the desnity profile (Plummer, EFF, King) and the radial projected distance in pc at which the 
survey is truncated. 3 degrees correspond to approx 7pc while the 6 degreees to 14 pc.

Each model should be able to run on real or synthetic data. To do this, modify the flag in runSpatial***.py.

NOTE: So far the ellipticity model does not correctly fit the ellipticity, unless it is zero.

To do:

* Implement Kent distribution to properly infer the eccentricity and avoid any projection related problems. DONE
* Couple Kent distribution with the density profiles. DONE for Plummer and King.

Issues:

* The eccentricity is somehow underestimated.
