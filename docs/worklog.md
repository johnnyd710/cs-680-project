## nov-6-2018
- raw data density plots for inboard/outboard vibrations of all datasets, with outliers greater than 2 standard deviations removed: ['1-1-17', '1-7-17', '1-16-17', '2-7-17', 'failure-14th']  
![Raw-Density-In-All](../figs/F1S-F1SFOBV-Overall-dist.png)
![Raw-Density-Out-All](../figs/F1S-F1SFOBV-Overall-dist.png)
![Raw-Density-In-All](../figs/F1S-F1SMOBV-Overall-dist.png)
![Raw-Density-Out-All](../figs/F1S-F1SMOBV-Overall-dist.png)   
- **January 16th** in bearing vibration ~100 times higher than usual... causing density plots to look weird. Motor plots look great tho.
- Same with **Jan 7th** out bearing vibration, there is a half hour stretch that is ~10-100 times times than usual, causing density plot to look weird.  
![Raw-Bar-In](../figs/F1S-F1SFIBV-Overall-bar.png)
![Raw-Bar-Out](../figs/F1S-F1SFIBV-Overall-bar.png) 
![Raw-Bar-In](../figs/F1S-F1SMIBV-Overall-bar.png)
![Raw-Bar-Out](../figs/F1S-F1SMIBV-Overall-bar.png)  
- It is more obvious in the bar plots that something is throwing off the vibration signals on those days.
![Raw-Density-In](../figs/density_F1S_F1SFIBV.png)
![Raw-Density-Out](../figs/density_F1S_F1SFOBV.png)
- so for now we will work with these three datasets only. Jan 1st, Feb 7th, and the day of failure, the 14th.

## nov-7-2018
![std-healthy](../figs/F1SFIBV_std.png)
![std-failure](../figs/F1SFOBV_std.png)
- standard deviations for bearing vibrations

![std-healthy](../figs/motor-vib-density.png)
![std-failure](../figs/motor-vib-density-out.png)
- getting bad results so far... options to try:
    - autoencoders
    - frequency analysis
    - use somtf.py and modify it yourself so you have more control... because i dont really understand the other implementation and its hard to change.
- #TODO make autoencoders accept a sequence from one variable!
![auto](../figs/ae-outlier-training.svg)
- first try with autoencoder , one second split, scale all relative
![auto](../figs/ae-outlier-training.png)
- first try with autoencoder , ten second split, scale all relative