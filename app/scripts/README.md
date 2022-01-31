This directory serves as a collection for scripts that perform certain, specific tasks.

This section contains the following scripts:
* `lof.py` runs the Local Outlier Factor against the three core datasets: Normal.csv, Abnormal.csv, and Combined.csv. All the datasets can be found in this section.
* `lof_custom.py` allows you to input and run Local Outlier Factor against your own dataset

This section also contains `.spec` files, whose names correspond to their target python script. Use these `.spec` files with `pyinstaller` to compile the scripts into Windows executables:

```sh
pyinstaller --onefile <target_specfile.spec>
```

After compilation, you can find your executables in `/dist`.