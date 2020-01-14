# AMT
Automatic measurements of Physical Traits of Fish

# Advanced project 2
Create a Python software to automatically measure fish's traits giving a photo.

<ul>
<li>working_notebook.ipynb: jupyter notebook where i am doing the development</li>
<li>FL_draw.py: isolated python script to meausere fish's traits</li>
<li>auxiliar/helper.py: python script with the basic functions for image processing.</li>
</ul>

# Requeriments. 
(Python 3.7). To setup (Tested on ubunto 16):

<ul>
<li>clone the repository</li>
<li>Install anaconda.</li>


<ul>
<li>On terminal:</li>
</ul>
<li><b>conda create -n AMT python=3.7</b></li>
<li><b>conda install -c conda-forge xgboost==0.90</b></li>
<li><b>pip install scikit-learn===0.21.3</b></li>
<li><b>pip install matplotlib===3.1.0</b></li>
<li><b>conda install -c conda-forge opencv==3.4.2</b></li>
</ul>

# Requeriments by txt file
Use <b>conda create --name AMT --file requirements.txt</b>

# To use:
<i>remember to place the pictures you would like analize in <b>images</b> folder.</i>

Run AMT.py example terminal: 
<p><b> conda activate AMT</b></p>
<p><b> python AMT.py 'A nigricans 16.7 SL-20.8 TL.jpg' yes True</b></p>
<p><b> python AMT.py 'A nigricans 16.7 SL-20.8 TL.jpg' no False</b></p>

<p>The first parameter  (yes/no) activates or deactivates the fin detection, this speeds up the process greatly</p>
<p>The second parameter (True/False) changes the eye detection selection, try both if the eye is not detected</p>

The image should be
place in the "./image/" folder

# See the demostration on see_me_demo.html

