import os
from math import pi
from math import sqrt
import numpy as np
import cv2
from scipy.optimize import curve_fit
import csv
#from os import path
import matplotlib.pyplot as plt


#%%IMPORTANT PARAMETERS TO SET!!!!!!!!@@@@@@@@@@@@@@
folder_date = "2021-01-11" #Input the date that the image is from!!!!
batch_no = ["00031","00032","00033","00034","00035","00036","00037","00038","00039"]
#Array of image numbers to be analysed

num_run = 1 #Should be run, since the code has not evolved to perform batch processing yet

#Mention below the starting and end points of x and Z ROI. This wil also be used to see the ROI on the plot
ROI_x = [500,900]
ROI_z = [500,900]

cbarlim=(0,1.0) #set your colour bar limit for scaled image
savepath = "C:/Users/Atomionics/Desktop/test/cloud_expansion_11012021/"

#%% Defining Variables
folder_to_save_files = savepath 
if not os.path.exists(folder_to_save_files):
    os.mkdir(folder_to_save_files)

a = "a"
b = "b"
c = "c"

ROI_x_start = ROI_x[0]
ROI_x_end = ROI_x[1]
ROI_z_start = ROI_z[0]
ROI_z_end = ROI_z[1]
ROI_x_size = ROI_x_end-ROI_x_start
ROI_z_size = ROI_z_end-ROI_z_start

kB = 1.38e-23
muB = 9.27e-24
h = 6.63e-34
 
### Rubidium properties
amu = 1.66053873e-27    
isotope = 85

if isotope == 85:
    mass = 85*amu
    lam = 780.2e-9
    gam = 6.067
    Isat = 16.69
    scatleng = 0 # Bohr radii
    scattXsection = 8*pi*(scatleng*5.29e-11)**2  # m^2
    threebodyloss = 0  # m^6/s
elif isotope == 87:
    mass = 87*amu
    lam = 780.2e-9
    gam = 6.065
    Isat = 16.69
    scatleng = 0 # Bohr radii
    scattXsection = 8*pi*(scatleng*5.29e-11)**2  #m^2
    threebodyloss = 0  # m^6/s


### Supposedly imported from the GUI 

px_size = 5.5e-6
binning = 1
magnif = 0.38
px = px_size*binning
px_eff = px/magnif
tof =40e3
I = 0.06*Isat
IoverIs = I/Isat
delta = 0

ROI_sum=0

#%% Importation of the 3 images

dirInfo = os.listdir(savepath)    
sizeDirInfo = len(dirInfo)

if sizeDirInfo == 0:
    lastImageNum = 1
else:
    lastImageName = dirInfo[(sizeDirInfo-1)]
    # lastImageNum = float(lastImageName[15:20]) + 1
    
#%% Define functions

def gaus(x,b,a,x0,sigma):
    return b+a*np.exp(-(x-x0)**2/(2*sigma**2))

#%%%
def batch_proc():
    img_at = cv2.imread(savepath + folder_date + "-img_%05s_a.png" % (image_no),0) # With atom                    
    img_las = cv2.imread(savepath + folder_date + "-img_%05s_b.png" % (image_no),0) # Laser alone
    img_bck = cv2.imread(savepath + folder_date + "-img_%05s_c.png" % (image_no),0) # Background

            
### Creation of the 2D array of optical depths

# Substraction of the background
    img_las = img_las - img_bck
    img_at = img_at -img_bck
    ###Correction of the zeros
    imin = min(np.min(img_las[img_las > 0]), np.min(img_at[img_at > 0]))
    img_las[img_las <= 0] = imin
    img_at[img_at <= 0] = imin
    # a = np.asarray(img_las)
    # b = np.asarray(img_at)
    # c = a/((b.astype('float')+1)/256)
    # d = c*(c<255)+255*np.ones(np.shape(c))*(c>255)
    # e =  d.astype('uint8')
    # 
    # OD=e
    #OD = np.divide(img_at,img_las)
    # Calculation of the optical depth
    OD = np.log(np.divide(img_las,img_at))
    print("OD done") 
    # print(OD[1200,700])
    # Negative transformation if gaussian value < background
    # if OD[int(len(OD[:,0])/2),int(len(OD[0,:])/2)] < OD[0,0]:
    #     for i in range(0, len(OD[:,0])):
    #         for j in range(0, len(OD[0,:])):
    #             OD[i,j] = np.max(OD) - OD[i,j]
            
    #    with np.printoptions(threshold=np.inf):
      

      
    #%% Fitting along the x axis (integrated over all z values)
    # Gaussian fitting to get the width
    sum_x = np.sum(OD, axis = 0)
    n = len(sum_x)
    x = np.linspace(1,n,n)
    x_val_to_find = imin + (np.max(sum_x)-imin)*1/np.exp(1)
    a_x = np.where(sum_x>=x_val_to_find)
    if np.max(sum_x)<=0:
        in_max_x = n/2
        print("np.max(sum_x) is negative!")
    else:
        in_max_x = np.where(sum_x==np.max(sum_x))[0][0]
    fit_x,pcov = curve_fit(gaus,x,sum_x,p0=[imin,np.max(sum_x),in_max_x,len(a_x[0])/2])
    # Measure of the fitting accuracy for the horizontal cross-section
    # err_x = sqrt(abs(pcov[3,3]))/fit_x[3]
    print("OD x  fit done")       
    #%% Fitting along the z axis (integrated over all x values)
    # Gaussian fitting to get the width
    sum_z = np.sum(OD, axis = 1)
    n = len(sum_z)
    z = np.linspace(1,n,n)
    z_val_to_find = imin + (np.max(sum_z)-imin)*1/np.exp(1)
    a_z = np.where(sum_z>=z_val_to_find)
    if np.max(sum_z)<=0:
        in_max_z = n/2
        print("np.max(sum_z) is negative!")
    else:
        in_max_z = np.where(sum_z==np.max(sum_z))[0][0]
    fit_z,pcov = curve_fit(gaus,z,sum_z,p0=[imin,np.max(sum_z),in_max_z,len(a_z[0])/2])
    # Measure of the fitting accuracy for the vertical cross-section
    # err_z = sqrt(abs(pcov[3,3]))/fit_z[3]
    print("OD z  fit done")  
    #%% Global peak optical depths
      
    center_x = int(round(fit_x[2]))
    center_z = int(round(fit_z[2]))
    center_x = ROI_x_start + int(ROI_x_size/2)
    center_z = ROI_z_start + int(ROI_z_size/2)
    # Cross sections centered in the previous peaks
    cross_x1 = OD[center_z,:]
    cross_x = cross_x1[ROI_x_start:ROI_x_end]
    cross_z1 = OD[:,center_x]
    cross_z = cross_z1[ROI_z_start:ROI_z_end];
    x = np.linspace(1,len(cross_x), len(cross_x))
    z = np.linspace(1,len(cross_z), len(cross_z))
    # Gaussian fitting of those cross sections
    cross_x_fit,pcov = curve_fit(gaus,x,cross_x,p0=[imin, np.max(cross_x),len(cross_x)/2,1])
    cross_z_fit,pcov = curve_fit(gaus,z,cross_z,p0=[imin, np.max(cross_z),len(cross_z)/2,1])
    print("OD second  fit done")  
    ROI_sum=0
    for row in range(ROI_x_start,ROI_x_end):
        for col in range(ROI_z_start, ROI_z_end):
            ROI_sum = OD[row, col] + ROI_sum
    print("ROI Sum = " + str(ROI_sum))
    
    points_x = [ROI_x_start, ROI_x_end]
    points_z = [ROI_z_start+(ROI_z_size/2), ROI_z_start+(ROI_z_size/2)]
    points_z2 = [ROI_z_start, ROI_z_end]
    points_x2= [ROI_x_start + (ROI_x_size/2), ROI_x_start + (ROI_x_size/2)]
        
    #%%Plotting
    plt.figure(figsize=(15,10))
    plt.subplot(2,3,5)
    plt.title("Colour Unscaled")
    plt.xlabel("Position (px)")
    plt.ylabel("Position (px)")
    plt.imshow(OD)
    plt.colorbar()
    fit_xl = gaus(x, *cross_x_fit)
    
    plt.subplot(2,3,4)
    plt.title("Fitting on x-axis")
    plt.xlabel("Position (px)")
    plt.ylabel("OD")
    xpixels=np.linspace(ROI_x_start,ROI_x_end,ROI_x_size)
    plt.plot(xpixels,cross_x,'--b')
    plt.plot(xpixels,fit_xl,'--r')
      
    fit_zl = gaus(z, *cross_z_fit)
    plt.subplot(2,3,2)
    plt.xlabel("Position (px)")
    plt.ylabel("OD")
    plt.title("Fitting on z-axis")
    zpixels=np.linspace(ROI_z_start,ROI_z_end,ROI_z_size)
    plt.plot(zpixels,cross_z,'--b')
    plt.plot(zpixels,fit_zl,'--r')
     
    plt.subplot(2,3,1)
    plt.title("Colour Scaled \n OD ROI_Sum = %i"  %ROI_sum,  fontsize=12)
    plt.xlabel("Position (px)")
    plt.ylabel("Position (px)")
    plt.plot(points_x, points_z, linestyle='dashed',linewidth=0.5, color='red')
    plt.plot(points_x2, points_z2, linestyle='dashed',linewidth=0.5, color='red')
    plt.plot([ROI_x_start,ROI_x_end], [ROI_z_start,ROI_z_start], linestyle='solid',linewidth=0.5, color='red')
    plt.plot([ROI_x_start,ROI_x_start], [ROI_z_start,ROI_z_end], linestyle='solid',linewidth=0.5, color='red')
    plt.plot([ROI_x_end,ROI_x_end], [ROI_z_end,ROI_z_start], linestyle='solid',linewidth=0.5, color='red')
    plt.plot([ROI_x_end,ROI_x_start], [ROI_z_end,ROI_z_end], linestyle='solid',linewidth=0.5, color='red')

    plt.imshow(OD)
    plt.clim(cbarlim) # colourbar limit (,)
    plt.colorbar() 

    # Calculation of the peak optical depth
    ODpk_x = cross_x_fit[0] + cross_x_fit[1]
    ODpk_z = cross_z_fit[0] + cross_z_fit[1]
    ODpk = (ODpk_x + ODpk_z)/2
    # Number of atoms
    Imeas = 0.955
    detun = 0
    IoverIs_sp1 = Imeas/Isat
    sigma0 = 3*lam**2/(2*pi)
    sigmatotal=sigma0/(1+2*IoverIs_sp1+4*(detun/gam)**2)
    sigma_x = cross_x_fit[3]*px_eff
    sigma_z = cross_z_fit[3]*px_eff     
    N_OD = 2*ODpk*pi*sigma_x*sigma_z/sigmatotal

    x_center = np.where(fit_xl==np.max(fit_xl))[0][0]+ROI_x[0]
    z_center = np.where(fit_zl==np.max(fit_zl))[0][0]+ROI_z[0]

    areax = np.sum(sum_x-imin)
    areaz = np.sum(sum_z-imin)
    Nx = areax*px_size**2/sigmatotal
    Nz = areaz*px_size**2/sigmatotal
    Npxsum = ROI_sum*px_size**2/sigmatotal

    plt.subplot(2,3,3)
    plt.text(-0.1, 0.9, "Image #: %05s" % (image_no), fontsize=30)
    plt.text(-0.1, 0.8, "Date: " +folder_date, fontsize=30)
    plt.text(-0.1, 0.6, "Important Parameters", fontsize=20, )
    plt.text(-0.1, 0.50, "Peak OD = " + str(round(ODpk,3)), fontsize=15)
    plt.text(-0.1, 0.40, "$\sigma_{x}$ = " + str(round(sigma_x,6)*10**6)+"mm", fontsize=15)
    plt.text(-0.1, 0.30, "$\sigma_{z}$ = " + str(round(sigma_z,6)*10**6)+"mm", fontsize=15)
    plt.text(-0.1, 0.20, "$N_{OD}$ = " + str(round(N_OD/(10**6),1)) + "*$10^{6}$ atoms", fontsize=15) 
    plt.text(-0.1, 0.10, "$N_{x}$ =  " + str(round(Nx/(10**6),1)) + "$*10^{6}$ atoms", fontsize=15)
    plt.text(-0.1, 0.00, "$N_{z}$ =  " + str(round(Nz/(10**6),1)) + "$*10^{6}$ atoms", fontsize=15)
    plt.text(-0.1, -0.10, "$N_{px sum}$ =  " + str(round(Npxsum/(10**6),1)) + "$*10^{6}$ atoms", fontsize=15)
    plt.text(-0.1, -0.20, "X Center = " + str(x_center) +" px", fontsize=15)
    plt.text(-0.1, -0.30, "Z Center = " + str(z_center) +" px", fontsize=15)
    plt.axis('off')
               
       #save image
    filename = savepath + folder_date + "-img_%05s_d.png" % (image_no)
    plt.savefig(filename, dpi=300, bbox_inches='tight' ) 
    # os.system(filename)
    print(image)
      #%% Export parameters of interest
    timeofflight = image*2/1000 
    headline = ['Image #', 'filename','Time of Flight','X Center','Z Center','sigma_x','sigma_z','ROI X','ROI Z',"N_OD","N_x","N_z","N_pxsum","Peak OD"]
      # If the csv file does not exist yet, creates it with its header
    if not os.path.exists(savepath + folder_date + "-Data00000.csv"):
        with open(savepath + folder_date +  '-Data00000.csv', 'x', newline = '') as file:
            header = csv.writer(file, dialect = 'excel', quoting = csv.QUOTE_NONE)
            header.writerow(headline)
      # Gets the run number as the last line number of the csv file
    with open(savepath + folder_date + '-Data00000.csv', 'r') as file:
        reader = csv.reader(file, dialect = 'excel')
        rows = list(reader)
        line_num = len(rows)
    parameters = [image_no,filename, timeofflight,x_center,z_center, sigma_x,sigma_z,str(ROI_x[0])+"-"+str(ROI_x[1]),str(ROI_z[0])+"-"+str(ROI_z[1]), N_OD, Nx, Nz, Npxsum, ODpk]
      # Adds the new set of parameters following the existing lines
    with open(savepath + folder_date + '-Data00000.csv', 'a', newline = '') as file:
        writer = csv.writer(file, dialect = 'excel', quoting = csv.QUOTE_NONE)
        writer.writerow(parameters)

#%%
for image in range(len(batch_no)):
    image_no = batch_no[image]
    batch_proc()
    plt.show()
