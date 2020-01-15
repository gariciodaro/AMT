# AMT
Automatic measurements of Physical Traits of Fish.

The image should have a black background and good quality.
The name of the picture should contain the TL (Total Lenght in physical dimensions, cm, in, etc.) of the fish, with the tag "TL". With that number, the software computes the transformation from pixels to cm, inches, etc. If an error occurs, the distances will be shown in pixels. example of correctly formatted picture name: S schlegeli 15.7 SL-18.8 TL.jpg

# Advanced project 2
Create a Python software to automatically measure fish's traits giving a photo.

<ul>
<li>see_mee_demo.ipynb: jupyter notebook with a demo</li>
<li>AMT.py: isolated python script to measure fish's traits</li>
<li>auxiliar/helper.py: python script with the basic functions for image processing.</li>
<li>auxiliar/classes: folder with auxiliar classes.</li>
<li>report_ml_*: folder with machine learning reports</li>
<li>report.pdf: theoretical summary</li> 
</ul>

# Requeriments. 
(Python 3.7). To setup (Tested on ubuntu 16):

<ul>
<li>clone the repository</li>
<li>Install anaconda.</li>


<ul>
<li>On terminal:</li>
</ul>
<li><b>conda create -n AMT python=3.7</b></li>
<li><b>conda activate AMT </b></li>
<li>on ubuntu: <b>conda install -c conda-forge xgboost==0.90</b>. on Windows <b>conda install -c anaconda py-xgboost==0.90</b></li>
<li><b>pip install scikit-learn==0.21.3</b></li>
<li><b>pip install pandas==0.24.2</b></li>
<li><b>pip install matplotlib==3.1.0</b></li>
<li><b>conda install -c conda-forge opencv==3.4.2</b></li>
</ul>

# Requeriments by txt file (Only on ubuntu)
Use <b>conda create --name AMT --file requirements.txt</b>

# To use:
<i>remember to place the pictures you would like analize in <b>images</b> folder.</i>

Run AMT.py example terminal: 
<p><b> conda activate AMT</b></p>
<p><b> python AMT.py 'S quoyi 12.3 SL-15.2 TL.jpg' yes True</b></p>
<p><b> python AMT.py 'S quoyi 12.3 SL-15.2 TL.jpg' no False</b></p>

<p>The first parameter  (yes/no) activates or deactivates the fin detection, this speeds up the process greatly</p>
<p>The second parameter (True/False) changes the eye detection selection, try both if the eye is not detected</p>

The image should be
place in the "./image/" folder

# See the demostration on see_me_demo.html

