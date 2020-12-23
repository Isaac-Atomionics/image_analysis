# Image Analysis Scripts

## Image_analysis.py
Im_Analysis_AI.py
This script opens the AI images (a,b,c) in the same folder and performs some fitting and finally outputs a image (d) with plots and important parameters. 
This code can be run directly from Spyder and the images can be displayed in the python console in spyder. If the console doesnt display the image, you need to enable it by following this steps.

**You can control which image be changing the unique 5 digit image numbers in the code

Things that need to be changed manually:

1. savepath - remember put the script in that folder.

2. image_no&folder_date - 5 unique digits and date the image was taken, the script works only for images in the same folder as the script and the image date must be correct

3. ROI_x and ROI_z

4. cbarlim - if you want 

If you get this error: "TypeError: unsupported operand type(s) for -: 'NoneType' and 'NoneType'", its a filenaming problem, where the code is not reading the images properly and returning nontype.
FIX: check and make sure the image number and folderdate is correct


If you get this error: "IndexError: index 0 is out of bounds for axis 0 with size 0",
it's probably because the image is really noisy and the sum_x/sum_z is negative or veryveryhigh.
FIX: try another image

#############################################################
CHANGELOG:
-Changed plot to 2x3 and renumbered the subplots so that the structure is same as before.
-Added text at one of the subplots so we can display whatever parameters we want.
-Reduced ROI line thickness and added a bo
-Added colour bar scaling, image number, ROI and savepath to the top for easy access
-Changed gaus initial guesses according to Ana's guide
-Horizontal axes of graphs updated to reflect actual pixel in focus
-Added titles and labels to graphs and axes
-Display some important parameters in the plot
-Added px_eff = px/magnif to account for maginification
-Added equations for
N_OD, Nx, Nz, Npxsum, sigmatotal, sigma0
-Updated parameters to be saved in CSV