<img style="float: right;" src="https://imaging.epfl.ch/resources/logo-for-gitlab.svg">


# Nucleoid distance on mitochondria
This code was developed by the [EPFL Center for Imaging](https://imaging.epfl.ch/) for the [Laboratory of Experimental Biophysics](https://www.epfl.ch/labs/leb/).


## Installation

### Step 1: Create and Activate a Virtual Environment
Using venv:\
```
python3 -m venv mito
source mito/bin/activate
```
Using conda:\
```
conda create -n mito
conda activate mito
conda install pip
```

### Step 2: Install Dependencies
```
pip install -r requirements.txt
```
### Step 3: Run the Code
```
python main.py
```

## Parameters to Tune and Variables to Change
Scroll down to the `__main__` section of `main.py`. Note that the names of variables containing coordinates always end in `_um` for micrometers or `_px` for pixels.

### pixel_size_um
**Description:** Adjust this if your data has a different pixel size.\
**Format:** The values should be the voxel size in micrometers, in the order z, y, x.

### ../test_folder
**Description:** Specify the folder containing your experiments.\
**Format:** Each experiment should have a separate folder, with each folder containing a file named `<folder_name>_decon.ome.tiff`.

### min_length_px
**Description:** Segments shorter than this value will be discarded.

### knots2data_ratio
**Description:** The ratio between the number of knots for the spline and the number of data points/pixels for a track.\
**Effect:** Larger values result in smoother approximations, while smaller values more closely follow the original segmentation.

### extension_um
**Description:** The amount by which the splines are extended at the ends.\
**Effect:** Capture nucleoids near the ends of mitochondria.

### half_window_size_um
**Description:** Half the size of the normal plane images that are extracted.\
**Effect:** Larger values will capture nucleoids further away.

### scipy.signal.find_peaks
**Description:** All parameters of this function can be adjusted to find the peaks corresponding to nucleoids.
