# AMT
Automatic measurements of Physical Traits of Fish.

Create a Python software to automatically measure fish's traits giving a photo.

The image should have a black background and good quality.
The name of the picture should contain the TL (Total Lenght in physical dimensions, cm, in, etc.) of the fish, with the tag "TL". With that number, the software computes the transformation from pixels to cm, inches, etc. If an error occurs, the distances will be shown in pixels. example of correctly formatted picture name: S schlegeli 15.7 SL-18.8 TL.jpg



+ **see_mee_demo.ipynb:** jupyter notebook with a demo.
+ **AMT.py:** isolated python script to measure fish's traits.
+ **auxiliar/helper.py:** python script with the basic functions for image processing.
+ **auxiliar/classes:** folder with auxiliar classes.
+ **report_ml_*: ** folder with machine learning reports.
+ **report.pdf:** theoretical summary.



## Current state of the project.
<img src="http://garisplace.com/img/fish_example.png" />

## All desired traits.
<img src="http://garisplace.com/img/fish_traits.jpeg" />


# Requeriments. 
(Python 3.7). To setup (Tested on ubunto 16):


clone the repository
Install anaconda.

On terminal:

+ `conda create -n AMT python=3.7`
+ `conda install -c conda-forge xgboost==0.90`
+ `pip install scikit-learn==0.21.3`
+ `pip install matplotlib==3.1.0`
+ `conda install -c conda-forge opencv==3.4.2`


# Requeriments by txt file
Use `conda create --name AMT --file requirements.txt`

# To use:
<i>remember to place the pictures you would like analize in <b>images</b> folder.</i>

Run AMT.py example terminal: 
+ `conda activate AMT`
+ `python AMT.py 'A nigricans 16.7 SL-20.8 TL.jpg' yes True`
+ `python AMT.py 'A nigricans 16.7 SL-20.8 TL.jpg' no False`

The first parameter  (yes/no) activates or deactivates the fin detection, this speeds up the process greatly

The second parameter (True/False) changes the eye detection selection, try both if the eye is not detected

The image should be
place in the "./image/" folder

# Note:
Fin and mouth detection were implemented using my machine learning diagnostic tool. Available  [here](https://github.com/gariciodaro/MLDiagnosisTool)



# See the demostration on see_me_demo.html

