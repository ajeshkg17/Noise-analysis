# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 20:14:18 2022

@author: admin-nisel120

The dataset is located in the "folder" variable,
 and the parameters such as the filename,
  file prefix, delimiter, and sample rate are set.
   The temperature range is set to "AutoRange," 
   which means that the code will automatically detect the range of temperatures in the dataset.
    The method for analyzing the data is also specified as "psd_welch_mean." 
    The code also has some options for skipping the start or end of the data, 
    trimming the time, and using a rolling average. 
    However, these options are not currently being used in the code.

    Bug fix:V2.1
    If the we apply more than one data removal oparation only the last one is applied as all oparations take raw data to signal data variable
"""
print("\n______________________Start____________________\n")
import multiprocessing
from scipy.signal import decimate, firwin, kaiserord
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import welch, get_window
from scipy.stats import kurtosis
#which system are you using
if os.path.exists("/Users/admin-nisem543"):
    print("running in Ajesh's Mac book pro")
    mac         = True
    lab_pc      = False
    kajal_pc    = False
elif os.path.exists(""):
    print("running in Kajal's PC")
    mac         = False
    lab_pc      = False
    kajal_pc    = True
else  : 
    print("running in Ajesh's PC")
    kajal_pc    = False
    mac         = False
    lab_pc      = True
delimiter   = ","
lineterminator = "\n"
skip_tail_raw   = 0     # Remove few data points from the end
skip_start_raw  = 0     # Remove few data points from the beginning
trim_length     = 0     # Trimms the date. Keeps from endng till the trim_length towards the start eg:800seconds
remove_comma    = False # Some data has comma and space mixed as delimitor
temperature_range = "AutoRange"
# folder      = r"C:\Users\admin-nisel120\ownCloud5\MAX PLANK\Data\Data\PPMS14T\Ajesh_2022\FGT26052022\Device4\Device4_on-24-12-2022\Data"
# filename    ='10K_10mS_3rd-Order'
# fileprefix  = "K_10mS_3rd-Order" 
# delimiter   = ","
# lineterminator = "\n"
# temperature_range = np.hstack((np.arange(10,360,10)))
# sample_rate = 104.6

# folder      = r"C:\Data_analysis\old\New folder"
# fileprefix  = "K"
# delimiter   = "\t"
# lineterminator = "\t\n"
# sample_rate = 1674
# temperature_range = np.hstack(([10],np.arange(25,325,25)))

# folder      = r"C:\Users\admin-nisel120\ownCloud5\MAX PLANK\Data\Data\PPMS9T\Ajesh\2022\GoldWire_Data_29-12-2022"
# filename    ='10K_10mS_3rdOrder'
# fileprefix  = "K_10mS_3rdOrder"
# delimiter   = ","
# lineterminator = "\n"
# temperature_range = [10,40,60,80,100,140,180,220,260,280,300,340]
# sample_rate = 104.6

# #FINAL MEASUREMENT ON DEVICE4 FGT3_26052021
# if mac: folder      = "/Users/admin-nisem543/seafile/MAX PLANK/Data/Data/PPMS14T/Ajesh_2022/FGT26052022/Device4_on_16-01-2023_5Prob/Data"
# else folder      = r"C:\Users\admin-nisel120\ownCloud5\MAX PLANK\Data\Data\PPMS14T\Ajesh_2022\FGT26052022\Device4_on_16-01-2023_5Prob\Data"
# filename    ='250K_10mS_K'
# fileprefix  = "K_10mS_K"
# # filename    ='205K_10mS'
# # fileprefix  = "K_10mS"
# sample_rate = 104.6


############################################################ FGT3-S25 D1
# # First measurement on FGT3-S25. done on 20th jan morning
# folder      = r"C:\Users\admin-nisel120\ownCloud5\MAX PLANK\Data\Data\PPMS14T\Ajesh_2022\FGT3_S25_#047\Data"
# #second measurement on FGT3-S25. done on 20th jan morning
# folder      = r"C:\Users\admin-nisel120\ownCloud5\MAX PLANK\Data\Data\PPMS14T\Ajesh_2022\FGT3_S25_#047\Data repeat on 20jan2023"
# #Third measurement on FGT3-S25. Done on 20th jan night
# if mac: folder      = "FGT3_S25_#047/Data repeat on 20th night"
# else  : folder      = r"C:\Users\admin-nisel120\ownCloud5\MAX PLANK\Data\Data\PPMS14T\Ajesh_2022\FGT3_S25_#047\Data reapeat on 20th night"
# Data taken from all 3 measuements are combined in Data_combined folder
# if mac: folder      = "/Users/admin-nisem543/Seafile/MAX PLANK/Data/PPMS/FGT3_S25_#047/D1/Combined"
# if kajal_pc : folder = ""
# if lab_pc  : folder      = r"C:\Users\admin-nisel120\ownCloud5\MAX PLANK\Data\Data\PPMS14T\Ajesh_2022\FGT3_S25_#047\Data_combined"
# fileprefix  = "K_5mS"
# row_sample_rate = 104.6
# method          =  "MSA_n2_norm___f_scaled___"#"MSA_n2_norm_lowpass"#"MSA_n2_norm" #"psd_welch_mean"#___skip_start_600s

#____Measured on Oxford System_____
# measurement 1 Long measurement
# if mac: folder      = "/Users/admin-nisem543/Seafile/MAX PLANK/Data/PPMS/Oxford Cryostat/FGT3_S25_#47/D1/Data"
# if kajal_pc : folder = ""
# if lab_pc  : folder      = r""
# fileprefix  = "K"
# row_sample_rate = 837.1
# method          =  "MSA_n2_norm___f_scaled___skip_start_600s___trim_time_3600s___skip_tail_3600s"#"MSA_n2_norm_lowpass"#"MSA_n2_norm" #"psd_welch_mean"#___skip_start_600s

#** measurement 2 __5hr sequence
if mac: folder      = "/Users/admin-nisem543/Seafile/MAX PLANK/Data/PPMS/Oxford Cryostat/FGT3_S25_#47/D1/Data_03-03-2023"
if kajal_pc : folder = ""
if lab_pc  : folder      = r""
fileprefix  = "K_2mS"
#row_sample_rate = 0 #To test the automatic sample rate calculator
#method          =  "MSA_n2_norm___f_scaled___skip_start_3600s"
method          =  "MSA_n2_norm___f_scaled___skip_start_7200s"
#method          =  "MSA_n2_norm___f_scaled___skip_start_10800s"#"MSA_n2_norm_lowpass"#"MSA_n2_norm" #"psd_welch_mean"#___skip_start_600s
#method          =  "psd_welch_median_1000s___f_scaled___skip_start_7200s"
#method          =  "psd_welch_mean_1000s___f_scaled___skip_start_7200s"
#method          =  "psd_welch_1000s___f_scaled___skip_start_7200s"
###method          =  "psd_welch_median_1000s___lowpass___f_scaled___skip_start_7200s"
#method          =  "psd_welch_median_1000s___lowpass___f_scaled___skip_start_10800s"
#method          =  "psd_welch_median_1000s___lowpass___f_scaled___skip_start_3600s"
#method          =  "psd_welch_median_100s___lowpass___f_scaled___skip_start_3600s"
#method          =  "psd_welch_median_100s____lowpass___f_scaled___skip_start_3600s"
#method          =  "psd_welch_median_100s_noverlap_50%_hamming___lowpass___f_scaled___skip_start_3600s"
lineterminator = ", \n"
temperature_range =[115,120,125,130,135,138,140,142,144,146,148,150,152,154,156. ,158. ,160., 162, 164., 166., 168., 170., 172,174,176,178,180, 182. ,184., 186., 188., 190., 192., 194. ,196. ,200., 202., 204. ,206., 208.,210. ,212., 214., 216. ,218, 220, 225., 230., 235., 240., 250. ,255. ,260.,265., 270., 275., 280., 285., 290., 295., 300.]

#************************************************************* FGT3-S25_D5
#if mac: folder      = "/Users/admin-nisem543/Documents/FGT3_S25_#47_9T_Noise/D5_2-Feb_2023_night"
#else  : folder      = r""
#fileprefix  = "K_2,6mS"
#row_sample_rate = 837.1
#method          =  "MSA_n2_norm___lowpass___f_scaled___round3"#"MSA_n2_norm_lowpass"#"MSA_n2_norm" #"psd_welch_mean"#___skip_start_600s


#************************************************************* Carbon resistor
#if mac: folder      = "/Users/admin-nisem543/Seafile/MAX PLANK/Data/PPMS/Carbon resistor/Data_05-03-2023"
#if kajal_pc : folder = ""
#if lab_pc  : folder      = r""
#fileprefix  = "K_2,6mS"
#fileprefix  = "K_10,6mS"
#remove_comma = True #data has both comma and space separating columns. we remove all comma. Also the last space.
#method          =  "MSA_n2_norm___lowpass___f_scaled___skip_start_600s"
#method          =  "MSA_n2_norm___lowpass___f_scaled___skip_start_1600s"#"MSA_n2_norm_lowpass"#"MSA_n2_norm" #"psd_welch_mean"#___skip_start_600s
#method          =  "psd_welch_mean_100s___lowpass___f_scaled___skip_start_1600s"#"MSA_n2_norm_lowpass"#"MSA_n2_norm" #"psd_welch_mean"#___skip_start_600s
#method          =  "psd_welch_median_100s___lowpass___f_scaled___skip_start_1600s"
#method          =  "MSA_n2_norm___f_scaled___skip_start_600s___trim_time_2000s"
#method          =  "MSA_n2_norm___f_scaled___skip_start_1600s___trim_time_2000s"
########## measurement 2
#if mac: folder      = "/Users/admin-nisem543/Seafile/MAX PLANK/Data/PPMS/Carbon resistor/Data_06-03-2023"
#if kajal_pc : folder = ""
#if lab_pc  : folder      = r""
#fileprefix  = "K_2,6mS"
#fileprefix  = "K_10,6mS"
#remove_comma = True # "K_10,6mS" data has both comma and space separating columns. we remove all comma. Also the last space.
#method          =  "MSA_n2_norm___f_scaled___skip_start_600s___trim_time_1000s"
#method          =  "MSA_n2_norm___f_scaled___skip_start_1600s___trim_time_1000s"
#method          =  "MSA_n2_norm___lowpass___f_scaled___skip_start_600s___trim_time_1000s"
#method          =  "MSA_n2_norm___lowpass___f_scaled___skip_start_1600s___trim_time_1000s"                 #"MSA_n2_norm_lowpass"#"MSA_n2_norm" #"psd_welch_mean"#___skip_start_600s
#method          =  "psd_welch_mean_100s___lowpass___f_scaled___skip_start_1600s___trim_time_1000s"                           #"MSA_n2_norm_lowpass"#"MSA_n2_norm"#"psd_welch_mean"#___skip_start_600s
#method          =  "psd_welch_median_100s___lowpass___f_scaled___skip_start_1600s___trim_time_1000s"

#___________________________________________________________________________________________________
print("The method deployed is: ",method)
#### Basic variables
# temperature_range = [200]
tosecond        = 1/104.6/573440
#skip_few_minuts
skip_tail       = False
skip_start      = False
trim_time       = False
rollingavg      = True
if "skip_tail" in method :
    skip_tail        = True
    skip_tail_time = method[method.find("skip_tail_")+len("skip_tail_"):]    #time to skip
    skip_tail_time = float(skip_tail_time[:skip_tail_time.find("s")])          #time to skip
    print(f"will skip tail end of {skip_tail_time}s")
if "skip_start" in method :
    skip_start       = True
    skip_start_time = method[method.find("skip_start_")+len("skip_start_"):] #time to skip
    skip_start_time = float(skip_start_time[:skip_start_time.find("s")])       #time to skip
    print(f"will skip start of {skip_start_time}s")
if "trim_time" in method :
    trim_time        = True
    trim_length_time = method[method.find("trim_time_")+len("trim_time_"):]      #time to skip
    trim_length_time = float(trim_length_time[:trim_length_time.find("s")])                #time to skip
    print(f"will trim_time of {trim_length_time}s")
#skip_tail_raw   = int( 3600*row_sample_rate)    # Remove few data points from the end
#skip_start_raw  = int( 600*row_sample_rate)     # Remove few data points from the beginning
#trim_length     = int( 3600*row_sample_rate)     # Trimms the date. Keeps from endng till the trim_length towards the start eg:800seconds
#Check 70K,110K data and edit it 
# temperature_range   = [300]



def fourier_transform_doubleside(signal,sample_rate,method):
    print("FFT on full time domain calculating....")
    # print(signal[:100])
    fourier     = np.fft.fft(signal) 
    N           = int(np.size(signal))
    frequency   = np.fft.fftfreq(N,1/sample_rate) #positive side fft of time domain
    #frequency = (k / N) * sample_rate
    #Assuming even number of data points
    magnitude   = np.abs(fourier)
    amplitude   = magnitude/N
    psd     = 2*amplitude**2
    psd[0]  = amplitude[0]**2
    # print(frequency[:100])
    # print("psd", psd[:100])
    return [frequency[:],psd]


def psd_welch_old(signal,sample_rate,method):
    """
    average ="mean"
    average = method[method.find("psd_welch_")+len("psd_welch_"):]
    average = average[:average.find("_")]
    if average in ["mean","median"]:
        nperseg_time = method[method.find(f"psd_welch_{average}_")+len(f"psd_welch_{average}_"):] # extract the number + s_
        nperseg_time = float(nperseg_time[:nperseg_time.find("s")])                               # Remove s_and_the_rest
        nperseg1     = int((nperseg_time-1)*sample_rate)
        print(f"welch {average} is running... with segment time of {nperseg_time}s")
        f, Pxx_den = welch(signal, sample_rate, nperseg=nperseg1, average=average)
    else:
        nperseg_time = method[method.find(f"psd_welch_")+len(f"psd_welch_"):]
        nperseg_time = float(nperseg_time[:nperseg_time.find("s")])
        nperseg1     = int((nperseg_time-1)*sample_rate)
        print(f"simple welch is running... with segment time of {nperseg_time}s")
        f, Pxx_den  = welch(signal, sample_rate, nperseg=nperseg1)
    # plt.semilogy(f, Pxx_den, label='mean')
    # plt.semilogy(f_med, Pxx_den_med, label='median')
    # plt.ylim([0.5e-3, 1])
    # plt.xlabel('frequency [Hz]')
    # plt.ylabel('PSD [V**2/Hz]')
    # plt.legend()
    # plt.show()
    frequency   = f
    psd         = Pxx_den  
    return [frequency[:],psd]
    """

def psd_welch(signal, sample_rate, method):
    # Parse the method string to extract the parameters
    # psd_welch_mean_100s_noverlap_50%_hamming
    # psd_welch_mean_100s_noverlap_50%_hann
    method_welch = method.split("___")[0]    # str is for copying the string
    print( f"Using {method_welch}")
    method_parts = method_welch.split('_')
    average = method_parts[2]
    nperseg_time = float(method_parts[3][:-1])
    if len(method_parts)-1 >= 4:
        if method_parts[4] == "noverlap":
            noverlap_percent = float(method_parts[5][:-1])
        else:
            noverlap_percent = None
    else:
        noverlap_percent = None
    if len(method_parts)-1 >=6:
        window_name = "hamming" if method_parts[6] == "hamming" else "hann"
    else:
        window_name ="hann"


    
    # Convert the time-based parameters to samples
    nperseg = int(nperseg_time * sample_rate)
    noverlap_time =  noverlap_percent/100*nperseg_time if noverlap_percent != None else None
    noverlap = int(noverlap_percent/100 * nperseg) if noverlap_percent is not None else None
    
    # Get the window function
    window = get_window(window_name, nperseg, fftbins = True) 
    
    # Compute the PSD using Welch's method
    print(f"Welch's method is running with {average} averaging, segment length {nperseg_time}s, window function {window_name}, and noverlap {noverlap_time}s")
    f, Pxx_den = welch(signal, sample_rate, nperseg=nperseg, window=window, noverlap=noverlap, average=average)
    
    return f, Pxx_den


def rolling_average(data):
    exponential = np.array([])
    i   = 0
    index = 0
    #Make a exponential window sequence for rolling average
    while index <= np.size(data):
        power   = int(round(np.power(1.2,i)))
        if power == 0: power =1
        if power <1000 and rollingavg: i   = i +1
        exponential = np.append(exponential,power)
        index   = power+index
    window_list = exponential
    result          = np.array([])
    stddev_list     = np.array([])
    if isinstance(window_list,np.ndarray):
        window_sum  = 0 #Sum of the windows so far
        for window_size in window_list:
            total   = 0
            window_size = int(window_size)
            if window_sum+window_size >= np.size(data):
                    # Window is crossing the size of the array
                    break
            # Total of window sizes so far is the current index of the data set being avaraged 
            total   = np.sum(data[window_sum:window_sum+window_size]) # eg: [400,401,402...411] window_size is 11 here
            #stddev  = np.std(total_list)
            #stddev_list = np.append(stddev_list,stddev)
            result  = np.append(result,total/window_size)
            window_sum  = window_sum + window_size
        stddev_list = result
    else:
        print("ERROR: WINDOW LIST IS NOT AN NP ARRAY")
    return [result[:],stddev_list]

#######################
def lowpass(signal,fs,method):

    # Input signal
    x = signal

    # Decimation factors
    dec_factors = [4, 4, 2]

    # Stop-band frequency
    stop_freq = 15 # Hz

    width = 10

    # Attenuation at stop band
    attenuation = 100 # dB

    # Initialize output signal
    y = x

    # Pad the signal with zeros
    #y = np.pad(y, (taps, taps), mode='edge')

    # Loop over decimation stages
    for i, dec_factor in enumerate(dec_factors):
        # Calculate cutoff frequency
        cutoff = stop_freq * dec_factor**i / fs
        # Calculate number of taps
        if i == 0:
            taps = 35
        elif i == 1:
            taps = 25
        else:
            taps = 321
        # Design antialiasing filter
        n, beta = kaiserord(attenuation, width)
        b = firwin(n, cutoff, window=('kaiser', beta), scale=False)
        # Apply antialiasing filter
        y = np.convolve(y, b, mode='same')
        # Decimate signal
        y = decimate(y, dec_factor)

        # Pad the signal at both ends to reduce the sudden jumps
        if "padding" in method :
            padding_size = int(np.ceil(n / 2))
            y = np.pad(y, (taps, taps), mode='edge')
    # Final sampling rate
    fs_final = fs / np.prod(dec_factors)
    return (fs_final,y)


#### Data analysis function
def analyse_signal(    filename, folder, method, tosecond, lineterminator, delimiter, skip_tail, skip_start, trim_time, skip_tail_raw, skip_start_raw, trim_length, lowpass, psd_welch, fourier_transform_doubleside):
    print("\n analysing : ", filename)

    #### Reding the file into Data array
    filelocation    = os.path.join(folder , filename + ".txt")
    analysis_base   = os.path.join(folder,"Analyse")
    analysis_imagelocation = os.path.join(analysis_base, f"_{method}",filename+f"{method}.png")
    analysis_filelocation2 = os.path.join(analysis_base, f"_{method}",filename+ f"_{method}_analysed_reduced.txt")
    analysis_filelocation = os.path.join(analysis_base, f"_{method}","full_data",filename + f"_{method}_analysed.txt")
    if not os.path.exists(analysis_base):
        print("making base analysis directory: \n", )
        os.mkdir(analysis_base)
    if not os.path.exists(os.path.dirname(analysis_imagelocation)):
        print("making analysis directory: \n", os.path.dirname(analysis_imagelocation))
        os.mkdir(os.path.dirname(analysis_imagelocation))
    if not os.path.exists(os.path.dirname(analysis_filelocation)):
        print("making analysis directory: \n", os.path.dirname(analysis_filelocation))
        os.mkdir(os.path.dirname(analysis_filelocation))
    
    #Open the data file
    if os.path.exists(os.path.join(os.path.dirname(filelocation),filename[:-4]+".npy")):
        data = np.load(os.path.join(os.path.dirname(filelocation),filename[:-4]+".npy"))
    else:
        with open(filelocation,"r") as f:
            data    = f.read()
            # Split the data into rows
            rows    = data.split(lineterminator)
            rows    = rows[0:-1]
            # Split each row into columns
            # Split each row into columns and convert the values to floats
            if remove_comma:
                rows = [row.replace(",","")[:-1] for row in rows]
                delimiter = " "
            columns = [[float(x) for x in row.split(delimiter)] for row in rows]
            # Convert the columns to a numpy array
            data    = np.array(columns)[:]
        #Save the binary file to acess fast 
        np.save(os.path.join(os.path.dirname(filelocation),filename[:-4])+".npy",data)
        #Save the standard csv file to acess fast 
        np.savetxt(os.path.join(os.path.dirname(filelocation),filename[:-4])+".csv",data)
    #Calculate the data rate
    measurement_timeintervel = (data[1,0]-data[0,0])*tosecond
    row_sample_rate =1/measurement_timeintervel
    print(f"the row sampling rate is {row_sample_rate}Hz")

    #### Skipping few minues
    if True: 
        data[:,0]       = (data[:,0])*tosecond
        starting_time   = data[0,0] 
        print(f"measurement starting time : {starting_time:.0f}s")
        if skip_tail: #Skipping end
            skip_tail_raw = int(skip_tail_time*row_sample_rate)                      # number of rows to skip
            print(f" full data is from {data[0,0]-starting_time:.0f}s to {data[-1,0]-starting_time:.0f}s")
            data        = data[:-skip_tail_raw,:]
            print(f"after skipping tail, remaining from {data[0,0]-starting_time:.0f}s to {data[-1,0]-starting_time:.0f}s")
        if skip_start: #Skip begining
            skip_start_raw = int(skip_start_time*row_sample_rate)                    # number of rows to skip
            print(f"current data is from {data[0,0]-starting_time:.0f}s to {data[-1,0]-starting_time:.0f}s")
            data        = data[skip_start_raw:,:]
            print(f"after skipping start, remaining data is from {data[0,0]-starting_time:.0f}s to {data[-1,0]-starting_time:.0f}seconds")
        if trim_time:
            trim_length = int(trim_length_time*row_sample_rate)                          # number of rows to skip
            print(f"current data is from {data[0,0]-starting_time:.0f}s to {data[-1,0]-starting_time:.0f}s")
            data        = data[-trim_length:,:]
            print(f"remaining data after trimming from {data[0,0]-starting_time:.0f}s to {data[-1,0]-starting_time:.0f}seconds")
        
        time        = data[:,0]
        signal_x    = data[:,1]
        signal_y    = data[:,2]


#################################################
    #### Low pass filter 
    if "lowpass" in method:
        desimation_factor = [4,4,2]
        (sample_rate,signal_x) = lowpass(signal_x,row_sample_rate,method)
        print("low pass filter applied\n New sampling rate is:",row_sample_rate)
        (sample_rate,signal_y) = lowpass(signal_y,row_sample_rate,method)
        signal_length = signal_x.shape[0]
        time = np.linspace(time[0], time[-1], signal_length)
        print("low pass filter applied\n New sampling rate is:",sample_rate)
    else:
        sample_rate = row_sample_rate
########################################################

    #### Plotting the time domain signal
    plot_graph = True
    if plot_graph:
        fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
        ax1.plot(time[::10], signal_x[::10],label= "singal_x")
        ax1.set_title("signal_x")
        ax2.plot(time[::10],signal_y[::10],label= "singal_y")
        ax2.set_title("signal_y")
        #plt.savefig(os.path.join(os.path.dirname(analysis_filelocation),f"time_{filename[:-4]}_.png"))

    #### Calculating the PSD  
    if "MSA_n2_norm" in method:
        psd = fourier_transform_doubleside
    if "psd_welch" in method:
        psd = psd_welch
    # acf = np.correlate(signal, signal, mode='full')[:np.size(signal)]
    # Use the Wiener-Khinchin theorem to convert the ACF to the PSD
    # Carry out the fourier_transform to optain the PSD
    [frequencypsd, MSA_norm_x]  =  psd(signal_x,sample_rate,method)
    [frequencypsd, MSA_norm_y]  =  psd(signal_y,sample_rate,method)
    # back ground substraction
    background_substracted      =  abs(MSA_norm_x-MSA_norm_y)
    if "f_scaled" in method:
        MSA_norm_x = MSA_norm_x*frequencypsd
        print("PSD Scaled")
    array   = np.vstack((frequencypsd,MSA_norm_x,MSA_norm_y,background_substracted))
    header = "Frequency,MSA_norm_x,MSA_norm_y,background_substracted"
    #np.savetxt(analysis_filelocation,np.transpose(array),header = header, delimiter=",")
    #np.save(analysis_filelocation[:-4]+".npy",array)

    #### Reduce the number of frequencis in PSD
    print("rolling average calculating....")
    [reduced_frequency,stddev_list_frequency]   = rolling_average(frequencypsd)
    [reduced_MSA_norm_x,stddev_list_x]  = rolling_average(MSA_norm_x)
    [reduced_MSA_norm_y,stddev_list_y]  = rolling_average(MSA_norm_y)
    [reduced_bg_substracted,stddev_list_bg_substracted] = rolling_average(background_substracted)
    print(".....calculation over")

    #### Save the PSD                    
    array   = np.vstack((reduced_frequency,reduced_MSA_norm_x,stddev_list_x,reduced_MSA_norm_y,stddev_list_y,reduced_bg_substracted,stddev_list_bg_substracted))
    header = "reduced_frequency,reduced_MSA_norm_x,stddev_list_x,reduced_MSA_norm_y,stddev_list_y,reduced_bg_substracted,stddev_list_bg_substracted"
    np.savetxt(analysis_filelocation2,np.transpose(array),header = header, delimiter=",")


    # Plot PSD
    print("Plotting PSD....")
    if plot_graph: 
        ax3.loglog(reduced_frequency,reduced_MSA_norm_x)
        ax3.loglog(reduced_frequency,reduced_MSA_norm_y)
        ax3.set_title("Temperature is "+str(filename)[:-len(fileprefix)])
        ax3.set_xlabel("Frequency (Hz)")
        ax3.set_ylabel("PSD")
        ax4.loglog(frequencypsd[:],MSA_norm_x[:])
        ax4.loglog(frequencypsd[:],MSA_norm_y[:])
        ax4.set_title("Temperature is "+str(filename)[:-len(fileprefix)])
        ax4.set_xlabel("Frequency (Hz)")
        ax4.set_ylabel("PSD")
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        plt.savefig(os.path.join(os.path.dirname(analysis_filelocation),f"PSD_{filename[:-4]}_.png"))
        plt.savefig(analysis_imagelocation)

        # plt.loglog(frequencypsd,background_substracted)
        # plt.title("Temperature is "+str(filename)[:-len(fileprefix)])
        # plt.xlabel("Frequency (Hz)")
        # plt.ylabel("PSD")
        # plt.show()
        print(".....Plotting completed")
        plt.close()
    return

if __name__=="__main__":


    #### Creating the file list
    filelist                = []
    if temperature_range    == "AutoRange":
        filelist = []
        temperature_range = np.array([])
        # Create a list of files 
        directory_filelist = os.listdir(folder)
        for filename in directory_filelist:
            if filename.endswith(fileprefix+'.txt'):
                temperature = float(filename[:filename.find(fileprefix)])
                temperature_range = np.append(temperature_range,temperature)
    temperature_range   = np.sort(temperature_range) 
    print(temperature_range)
    filelist    = []
    for temperature in temperature_range: 
        if temperature % 1 ==0 :
            temperature = int(temperature)
        filename    = str(temperature)+fileprefix
        filelist.append(filename)
    # filelist    = ['310K_10mS_3rd-Order']
    # filelist    = ['2K']
    print(filelist)

    #load and analyse all the files
    for index, filename in enumerate(filelist):
        analyse_signal(filename, folder, method, tosecond, lineterminator, delimiter, skip_tail, skip_start, trim_time, skip_tail_raw, skip_start_raw, trim_length, lowpass, psd_welch, fourier_transform_doubleside)

    print("\n Program has ended with no error")
