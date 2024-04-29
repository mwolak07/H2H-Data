# H2H-Data  

Convenient python data interface for the H2H project.  

## Install  
```
pip install .
python -m build
python -m pip install .
```

## TODO  
### Tests  
- Histogram of trail lengths after nan cropping  
  - Probably cut out trials below 1 std. deviation  
- Histogram of handover time  
	- Cut handovers outside of 1 std.dev  
	- Repeat for other method of finding it  
- Histogram for roles after cropping.  
- Histogram for trial types after cropping.  
- Drop all pertubation trials  
	- Look for abnormalities in results.  
- Split data by small/large object  
	- Look for abnormalities in results.
