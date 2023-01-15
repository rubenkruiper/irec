# Intelligent Regulatory Compliancy (iReC)
Scripts and data to reproduce some of the work done for the iReC project, a collaboration between Northumbria University (NU) and Heriot-Watt University (HWU) that was funded by NU and the Building Research Establishment (BRE).

--------------------
How to get started
--------------------

Download and install the free version of the [Anaconda package manager](https://www.anaconda.com/products/distribution) for your system. If needed, there are many tutorials online on how to get started with Anaconda and Jupyter Notebook; [see this one for example](https://youtu.be/2WL-XTl2QYI).

After installing anaconda, open a terminal/console window (mac/linux) or Anaconda prompt (Windows) and verify your installation by running: `conda -V` 

The terminal should return the version of Anaconda that is now installed on your system. 
Next run `conda install -c anaconda git -y`

Navigate to the directory on your computer where you'd like to create a folder with the code for the iReC project, e.g., some specific folder for coding projects or simply `cd ~/Documents/` 
Then clone this repository  `git clone https://github.com/rubenkruiper/irec.git`
Sign in to your GitHub account if prompted.
Navigate into the new folder `cd irec`

1. Create a separate iReC environment that runs python 3.9: 
  * `conda create --name irec python=3.9 -y`
  * `conda activate irec`

2. Install dependencies:
  * `pip install -r requirements.txt` 
  * `conda install -c conda-forge pdftotext -y`

3. Start a notebook, which will open up a browser window:
  * `jupyter notebook`
  * Inside the browser window double-click on one of the notebooks to start it. Numbering indicates the order in which notebooks should be run, e.g., start with  `1. Term Extraction.ipynb`. 
  * Run the cells from top to bottom, either by selecting `Cell > Run All` in the drop-down menu or pressing `Shift+Enter` on the top cell and working your way down.


If you have any questions, please let me know!
