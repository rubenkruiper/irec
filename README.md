# Intelligent Regulatory Compliancy (iReC)
Scripts and data to reproduce some of the work done for the iReC project, a collaboration between Northumbria University (NU) and Heriot-Watt University (HWU) that was funded by NU and the Building Research Establishment (BRE).

--------------------
How to get started
--------------------

Download and install the free version of the [Anaconda package manager](https://www.anaconda.com/products/distribution) for your system. If needed, there are many tutorials online on how to get started with Anaconda and Jupyter Notebook; [see this one for example](https://youtu.be/2WL-XTl2QYI).

After installing anaconda, open a terminal/console window (mac/linux) or Anaconda powershell prompt (Windows) and verify your installation by running `conda -V`. This should return the version of Anaconda that is now installed on your system. 

Next run `conda install -c anaconda git`.

Navigate to the directory on your computer where you'd like to create a folder with the code for the iReC project, e.g., `cd ~/Documents/` or some specific folder for coding projects. Then clone this repository  `git clone https://github.com/rubenkruiper/irec.git`.

Navigate into the new folder `cd irec`.

1. Create a separate iReC environment that runs python 3.9: 
  * `conda create --name irec python=3.9`
  * `conda activate irec`

2. Install dependencies:
  * `pip install -r requirements.txt` 
  * `conda install -c conda-forge pdftotext`

3. Start a notebook, which will open up a browser window:
  * `jupyter notebook`
  * Inside the browser window navigate into one of the folders, e.g., `term_extraction/` and double-click on the  `Term Extraction.ipynb` notebook to start it.
  * Run the cells from top to bottom, either by selecting `Cell > Run All` in the drop-down menu or pressing `Shift+Enter` on the top cell and working your way down.


If you have any questions, please let me know!
