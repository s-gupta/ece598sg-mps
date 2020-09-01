## ECE598SG (Special Topics in Robot Learning)
### Programming Assignment 1
You must do `(A or B) and C`. If you took CS543 / ECE 549 in Spring 2020, you
must do `B and C`. <br/>
A. [Semantic Segmentation](./sseg/README.md) <br/>
B. [Character RNN](./char-rnn/README.md) <br/>
C. [Linear Quadratic Regulators](./lqr/README.md) <br/>



#### Instructions
1. Assignment is due at 11:59:59PM on Tuesday September 15, 2020.
2. Please see
[policies](http://saurabhg.web.illinois.edu/teaching/ece598sg/fa2020/policies.html)
on [course
website](http://saurabhg.web.illinois.edu/teaching/ece598sg/fa2020/index.html).
3. Submission instructions:
   1. A single report for all questions in PDF format, to be submitted to
   gradescope.  Course code is `MZN3XY`. This report should contain all that
   you want us to look at.
   2. You also need to submit code for all questions in the form of a single .zip
   file. Submit a single zip file to Compass2g.
   3. We reserve the right to take off points for not following submission
   instructions.
4. Problems A. and B. will involve training neural network models. You will
benefit from use of GPUs for these problems. You can access GPUs through
[campus
cluster](http://saurabhg.web.illinois.edu/teaching/ece598sg/fa2020/compute.html),
and also through Google Colab. Please see course website for instructions on
how to use campus cluster. Instructions to use Google Colab can be found below.
5. Lastly, be careful not to work of a public fork of this repo. Make a *private*
clone to work on your assignment. 

#### Google Colab
1. Access https://colab.research.google.com/ and login through for Illinois Google account.
2. Create a new notebook.
3. Select GPU Runtime: `Runtime` > `Change Runtime Type` > `Hardware Accelerator` > `GPU`.
4. You can copy over the starter code into this new notebook, and start development.
5. You can install dependencies by executing `!pip install -r requirements.txt`.
6. For the datasets, you have 2 options:
   1. You can download and unzip the data onto this colab instance (using
   `!wget` and `!tar -xf` commands through the notebook cells. This is likely
   the faster of the two options, though you will have to do this every time
   you open this notebook.
   2. You can copy the data into your Google Drive, and access it as follows.
   This may be slower, but the data will be persistent, so you only need to set
   it up once. 
   ```
   from google.colab import drive
   drive.mount('/content/gdrive/')
   import os
   os.chdir("/content/gdrive/My Drive/path_to_folder")
   ```
7. Keep in mind that you need to keep your browser window open while running
Colab. Colab does not allow long-running jobs but it should be sufficient for
the requirements of this assignment. Colab also likely won't save the state of
your notebooks for too long, so make sure you are saving and downloading models
that you train. 
8. Lastly, Colab might restrict the number of GPU hours you can use, so design
your experiments carefully so as to use your compute time wisely.



