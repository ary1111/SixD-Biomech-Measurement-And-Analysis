# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 01:45:46 2020

@author: Liang (Undergraduate Research), Adam Ryason (Research Engineer)
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
import math
import seaborn as sns

class Bio_data_analysis:

    def __init__(self, filename, Spread, Peak_width, Smooth_sample_size):
        self.filename = filename
        self.bio_data = pd.read_csv(filename)
        self.flen = int((self.bio_data.shape[0]-1)/100)
        self.width = Peak_width
        self.Spread = Spread
        self.Smooth_sample_size = 50
        stdXF, stdYF, stdZF, stdXT, stdYT, stdZT = self.calculate_stdev(self.bio_data)
        meanXF, meanYF, meanZF, meanXT, meanYT, meanZT = self.calculate_mean(self.bio_data)
        self.forceX_pos_limit = meanXF + Spread*stdXF
        self.forceX_neg_limit = meanXF - Spread*stdXF
        
        self.forceY_pos_limit = meanYF + Spread*stdYF
        self.forceY_neg_limit = meanYF - Spread*stdYF
        
        self.forceZ_pos_limit = meanZF + Spread*stdZF
        self.forceZ_neg_limit = meanZF - Spread*stdZF
        
        self.torqueX_pos_limit = meanXT + Spread*stdXT
        self.torqueX_neg_limit = meanXT - Spread*stdXT
        
        self.torqueY_pos_limit = meanYT + Spread*stdYT
        self.torqueY_neg_limit = meanYT - Spread*stdYT
        
        self.torqueZ_pos_limit = meanZT + Spread*stdZT
        self.torqueZ_neg_limit = meanZT - Spread*stdZT
    
    #calcualte the mean of forces and Torques in certain time range
    #inputs: start time(float), end time(float), each row represent 0.01s
    #outputs: float array of [forceX,forceY,forceZ,torqueX,torqueY,torqueZ]
    def calculate_mean(self,data_set):
        mean_forceX = np.mean(data_set.FT1.values)
        mean_forceY = np.mean(data_set.FT2.values)
        mean_forceZ = np.mean(data_set.FT3.values)
        
        mean_torqueX = np.mean(data_set.FT4.values)
        mean_torqueY = np.mean(data_set.FT5.values)
        mean_torqueZ = np.mean(data_set.FT6.values)
    
    
        return(mean_forceX,mean_forceY, mean_forceZ,
                 mean_torqueX, mean_torqueY, mean_torqueZ)
    
    def calculate_stdev(self,data_set):
    
        
        stdev_forceX = np.std(data_set.FT1.values)
        stdev_forceY = np.std(data_set.FT2.values)
        stdev_forceZ = np.std(data_set.FT3.values)
        
        stdev_torqueX = np.std(data_set.FT4.values)
        stdev_torqueY = np.std(data_set.FT5.values)
        stdev_torqueZ = np.std(data_set.FT6.values)
        return (stdev_forceX, stdev_forceY, stdev_forceZ, stdev_torqueX, stdev_torqueY, stdev_torqueZ)
    
    def smooth_data_convolve_my_average(arr, span):
        re = np.convolve(arr, np.ones(span * 2 + 1) / (span * 2 + 1), mode="same")
    
        # The "my_average" part: shrinks the averaging window on the side that 
        # reaches beyond the data, keeps the other side the same size as given 
        # by "span"
        re[0] = np.average(arr[:span])
        for i in range(1, span + 1):
            re[i] = np.average(arr[:i + span])
            re[-i] = np.average(arr[-i - span:])
        return re
    
    #Calculate magnitudes of forces and toeques
    def calculate_2_norm(self,start,end):
        #slice the data set
        data_set = self.bio_data.iloc[start:end]
        two_norm_force = []
        two_norm_torque = []
        for row in data_set.itertuples(index=False):
            two_norm_force.append(math.sqrt(row[0]**2+row[1]**2+row[2]**2))
            two_norm_torque.append(math.sqrt(row[3]**2+row[4]**2+row[5]**2))
        two_norm_force = np.array(two_norm_force)
        two_norm_torque = np.array(two_norm_torque)
        return two_norm_force, two_norm_torque
    
    
    
    def plot_in_range(self,start, end, data,POS_HEIGHT,NEG_HEIGHT, s = 0):    
            if s == 0:
                plt.clf()
                y = data
                peaks,_ = find_peaks( y, height = POS_HEIGHT,distance = self.width )
                neg_peaks, _ = find_peaks( -y, height = -NEG_HEIGHT,distance = self.width)
                plt.axhline(y=POS_HEIGHT,  color="gray",linestyle="-")
                plt.axhline(y=NEG_HEIGHT,  color="gray",linestyle="-")
                plt.subplot(111)
                plt.plot(np.arange(start/100,end/100,0.01),y, lw = 1,label="forceX")
                plt.plot(peaks/100+start/100, y[peaks], "x")
                plt.plot(neg_peaks/100+start/100, y[neg_peaks], "x")
                #plt.xlim(start,end)        
                if(len(peaks) == 0):
                    peaks_values = [0.0]
                else:
                    peaks_values = [0.0]*len(peaks)
                if(len(neg_peaks) == 0):
                    neg_peaks_values = [0.0]
                else:
                    neg_peaks_values = [0.0]*len(neg_peaks)
                for i in range(len(peaks)):
                    peaks_values[i] = y[peaks[i]] 
                for j in range(len(neg_peaks)):
                    neg_peaks_values[j] = y[neg_peaks[j]]
                return plt, peaks_values, neg_peaks_values
            else:
                plt.clf()
                y = self.smooth_data_convolve_my_average(data,self.Smooth_sample_size)
                peaks,_ = find_peaks( y, height = POS_HEIGHT,distance = self.width)
                neg_peaks, _ = find_peaks( -y, height = -NEG_HEIGHT,distance = self.width)
                plt.axhline(y=POS_HEIGHT,  color="gray",linestyle="-")
                plt.axhline(y=NEG_HEIGHT,  color="gray",linestyle="-")
                plt.subplot(111)
                plt.plot(np.arange(start,end),y, lw = 1,label="forceX")
                plt.plot(peaks+start, y[peaks], "x")
                plt.plot(neg_peaks+start, y[neg_peaks], "x")
                #plt.xlim(start,end)
                if(len(peaks) == 0):
                    peaks_values = [0.0]
                else:
                    peaks_values = [0.0]*len(peaks)
                if(len(neg_peaks) == 0):
                    neg_peaks_values = [0.0]
                else:
                    neg_peaks_values = [0.0]*len(neg_peaks)
                for i in range(len(peaks)):
                    peaks_values[i] = y[peaks[i]] 
                for i in range(len(neg_peaks)):
                    neg_peaks_values[i] = y[neg_peaks[i]]
                return plt, peaks_values, neg_peaks_values
                
    def plot_forceX(self,start_time, end_time, smooth = 0):
        start = int(start_time * 100)
        end = int(end_time * 100)
        if(start <0 or start > self.bio_data.shape[0]):
            print("start time out of range")
        elif(end <0 or end > self.bio_data.shape[0]):
            print("end time out of range")
        else:
            data_set = self.bio_data.iloc[start:end]
            x_plt, x_peaks,x_neg_peaks = self.plot_in_range(start,end,data_set.FT1.values, 
                                                             self.forceX_pos_limit,self.forceX_neg_limit, s = smooth)
            x_plt.xlabel("Time in s")
            x_plt.ylabel("ForceX in N/m")
            x_plt.savefig("ForceX in ["+str(start_time)+" to "+str(end_time)+"s]"+self.filename[0:-4])  
            return x_peaks,x_neg_peaks
    
    def plot_forceY(self,start_time, end_time, smooth = 0):
        start = int(start_time * 100)
        end = int(end_time * 100)
        if(start <0 or start > self.bio_data.shape[0]):
            print("start time out of range")
        elif(end <0 or end > self.bio_data.shape[0]):
            print("end time out of range")
        else:
            data_set = self.bio_data.iloc[start:end]
            y_plt, y_peaks,y_neg_peaks =  self.plot_in_range(start,end,data_set.FT2.values, 
                                                             self.forceY_pos_limit,self.forceY_neg_limit, s = smooth)
            y_plt.xlabel("Time in s")
            y_plt.ylabel("ForceY in N/m")
            y_plt.savefig("ForceY in ["+str(start_time)+" to "+str(end_time)+"s]"+self.filename[0:-4])
            return y_peaks,y_neg_peaks
    
    def plot_forceZ(self,start_time, end_time, smooth = 0):
        start = int(start_time * 100)
        end = int(end_time * 100)
        if(start <0 or start > self.bio_data.shape[0]):
            print("start time out of range")
        elif(end <0 or end > self.bio_data.shape[0]):
            print("end time out of range")
        else:
            data_set = self.bio_data.iloc[start:end]
            z_plt, z_peaks,z_neg_peaks =self.plot_in_range(start,end,data_set.FT3.values, 
                                                             self.forceZ_pos_limit,self.forceZ_neg_limit, s = smooth)
            z_plt.xlabel("Time in s")
            z_plt.ylabel("ForceZ in N/m")
            z_plt.savefig("ForceZ in ["+str(start_time)+" to "+str(end_time)+"s]"+self.filename[0:-4])
            return z_peaks,z_neg_peaks
    
    def plot_torqueX(self,start_time, end_time, smooth = 0):
        start = int(start_time * 100)
        end = int(end_time * 100)
        if(start <0 or start > self.bio_data.shape[0]):
            print("start time out of range")
        elif(end <0 or end > self.bio_data.shape[0]):
            print("end time out of range")
        else:
            data_set = self.bio_data.iloc[start:end]
            x_t_plt, x_t_peaks,x_t_neg_peaks =self.plot_in_range(start,end,data_set.FT4.values, 
                                                             self.torqueX_pos_limit,self.torqueX_neg_limit, s = smooth)
            x_t_plt.xlabel("Time in s")
            x_t_plt.ylabel("TorqueX in Nm")
            x_t_plt.savefig("TorqueX in ["+str(start_time)+" to "+str(end_time)+"s]"+self.filename[0:-4])
            return x_t_peaks,x_t_neg_peaks
            
    def plot_torqueY(self,start_time, end_time, smooth = 0):
        start = int(start_time * 100)
        end = int(end_time * 100)
        if(start <0 or start > self.bio_data.shape[0]):
            print("start time out of range")
        elif(end <0 or end > self.bio_data.shape[0]):
            print("end time out of range")
        else:
            data_set = self.bio_data.iloc[start:end]
            y_t_plt, y_t_peaks,y_t_neg_peaks =self.plot_in_range(start,end,data_set.FT5.values, 
                                                             self.torqueY_pos_limit,self.torqueY_neg_limit, s = smooth)
            y_t_plt.xlabel("Time in s")
            y_t_plt.ylabel("TorqueY in Nm")
            y_t_plt.savefig("TorqueY in ["+str(start_time)+" to "+str(end_time)+"s]"+self.filename[0:-4])
            return y_t_peaks,y_t_neg_peaks
    
    def plot_torqueZ(self,start_time, end_time, smooth = 0):
        start = int(start_time * 100)
        end = int(end_time * 100)
        if(start <0 or start > self.bio_data.shape[0]):
            print("start time out of range")
        elif(end <0 or end > self.bio_data.shape[0]):
            print("end time out of range")
        else:
            data_set = self.bio_data.iloc[start:end]
            z_t_plt, z_t_peaks,z_t_neg_peaks =self.plot_in_range(start,end,data_set.FT6.values, 
                                                             self.torqueZ_pos_limit,self.torqueZ_neg_limit, s = smooth)
            z_t_plt.xlabel("Time in s")
            z_t_plt.ylabel("TorqueZ in Nm")
            z_t_plt.savefig("TorqueZ in ["+str(start_time)+" to "+str(end_time)+"s]"+self.filename[0:-4])
            return z_t_peaks,z_t_neg_peaks
    
    def plot_normal_force_and_torque(self,start_time, end_time, smooth = 0):
        start = int(start_time * 100)
        end = int(end_time * 100)
        if(start <0 or start > self.bio_data.shape[0]):
            print("start time out of range")
        elif(end <0 or end > self.bio_data.shape[0]):
            print("end time out of range")
        else:
            two_norm_force, two_norm_torque = self.calculate_2_norm(start, end)
            NF_upper_limit = np.mean(np.array(two_norm_force))+ self.Spread*np.std(np.array(two_norm_force))
            NT_upper_limit = np.mean(two_norm_torque)+self.Spread*np.std(two_norm_torque)
            
            n_f_plt, n_f_peaks,n_f_neg_peaks =self.plot_in_range(start,end,two_norm_force, NF_upper_limit,0,s = smooth)
            n_f_plt.xlabel("Time in s")
            n_f_plt.ylabel("2_norm_force in Nm")
            n_f_plt.savefig("2_norm_force in ["+str(start_time)+" to "+str(end_time)+"s]")
            n_t_plt, n_t_peaks,n_t_neg_peaks =self.plot_in_range(start,end,two_norm_torque,NT_upper_limit,0, s = smooth)
            n_t_plt.xlabel("Time in s")
            n_t_plt.ylabel("2_norm_force in Nm")
            n_t_plt.savefig("2_norm_torque in ["+str(start_time)+" to "+str(end_time)+"s]")
            return n_f_peaks,n_f_neg_peaks,n_t_peaks,n_t_neg_peaks
    
            
    def plot_all_in_range(self,start_time, end_time, smooth = 0):
        x_peaks, x_neg_peaks = self.plot_forceX(start_time, end_time, smooth = smooth)
        y_peaks, y_neg_peaks = self.plot_forceY(start_time, end_time, smooth = smooth)
        z_peaks, z_neg_peaks = self.plot_forceZ(start_time, end_time, smooth = smooth)
        x_t_peaks, x_t_neg_peaks = self.plot_torqueX(start_time, end_time, smooth = smooth)
        y_t_peaks, y_t_neg_peaks = self.plot_torqueY(start_time, end_time, smooth = smooth)
        z_t_peaks, z_t_neg_peaks = self.plot_torqueZ(start_time, end_time, smooth = smooth)
        n_f_peaks,n_f_neg_peaks,n_t_peaks,n_t_neg_peaks = self.plot_normal_force_and_torque(start_time, end_time, smooth = smooth)
        
        #forces average peak values
        meanX = np.mean(np.array(x_peaks))
        meanX_neg = np.mean(np.array(x_neg_peaks)) 
        meanY = np.mean(np.array(y_peaks))
        meanY_neg = np.mean(np.array(y_neg_peaks)) 
        meanZ = np.mean(np.array(z_peaks))
        meanZ_neg = np.mean(np.array(z_neg_peaks)) 
        
        #torques average peak values
        meanX_t = np.mean(np.array(x_t_peaks))
        meanX_neg_t = np.mean(np.array(x_t_neg_peaks)) 
        meanY_t = np.mean(np.array(y_t_peaks))
        meanY_neg_t = np.mean(np.array(y_t_neg_peaks)) 
        meanZ_t = np.mean(np.array(z_t_peaks))
        meanZ_neg_t = np.mean(np.array(z_t_neg_peaks)) 
        
        #2_norm avearge peak values
        mean2F = np.mean(np.array(n_f_peaks))
        mean2T = np.mean(np.array(n_t_peaks)) 
        
        averages, neg_averages = [meanX,meanY,meanZ,meanX_t,meanY_t,meanZ_t,mean2F,mean2T],[meanX_neg,meanY_neg,meanZ_neg,meanX_neg_t,meanY_neg_t,meanZ_neg_t]
                    
        
        return averages, neg_averages
        
    def export_plots_and_table(self,smooth = 0):
        
        current_time, next_time = 0,400
        pos_V = []
        neg_V = []
        row = []
        
          
        while(1):
            if(next_time < self.flen):
                pos_V.append(self.plot_all_in_range(current_time, next_time, smooth = smooth)[0])
                neg_V.append(self.plot_all_in_range(current_time, next_time, smooth = smooth)[1])
                row.append(str(current_time)+ "-" + str(next_time))
                current_time += 400
                next_time += 400
            else:
                next_time = self.flen
                pos_V.append(self.plot_all_in_range(current_time, next_time, smooth = smooth)[0])
                neg_V.append(self.plot_all_in_range(current_time, next_time, smooth = smooth)[1])
                row.append(str(current_time)+ "-" + str(next_time))
                break
                
    
        col1 = ['ForceX','ForceY','ForceZ','TorqueX','TorqueY','TorqueZ','2_norm_Force','2_norm_Torque']
        col2 = ['ForceX','ForceY','ForceZ','TorqueX','TorqueY','TorqueZ']
    
        df_pos = pd.DataFrame(np.array(pos_V), columns = col1,index = np.array(row))
        df_neg = pd.DataFrame(np.array(neg_V), columns = col2,index = np.array(row))
        
        fig_pos, ax1 = plt.subplots(figsize=(15,10)) 
        sns.heatmap(df_pos, annot=True, fmt = ".2f", linewidths=.5, ax=ax1)
        fig_pos.savefig("Positive Average Peak table"+self.filename[0:-4])
        
        plt.clf() 
        fig_neg, ax2 = plt.subplots(figsize=(12,6)) 
        sns.heatmap(df_neg, annot=True, fmt = ".2f", linewidths=.5, ax=ax2, cmap="YlGnBu")
        fig_neg.savefig("Negative Average Peak table"+self.filename[0:-4])
        
        mean_total_pos, mean_total_neg = self.plot_all_in_range(0,self.flen,smooth = smooth)
        mean_total_neg = np.append(mean_total_neg, [0.0,0.0])
        Values = {'Pos Average Peaks':mean_total_pos, 'Neg Average Peaks':mean_total_neg}
        df_total = pd.DataFrame(Values, columns = ['Pos Average Peaks','Neg Average Peaks'],index = col1)
        plt.clf() 
        fig_total, ax3 = plt.subplots(figsize=(6,8)) 
        sns.heatmap(df_total, annot=True, fmt = ".2f", linewidths=.5, ax=ax3, cmap="Greens")
        fig_total.savefig("Total Average Peaks"+self.filename[0:-4])
        

Tester1 = Bio_data_analysis("Procedure 6 - Participant 5.csv",2,250,50)
Tester2 = Bio_data_analysis("Procedure 7 - Participant 5.csv",2,250,50)
Tester3 = Bio_data_analysis("Procedure 8 - Participant 6.csv",2,250,50)
Tester4 = Bio_data_analysis("Procedure 5 - Participant 4.csv",2,250,50)
#Tester1.export_plots_and_table()
#Tester2.export_plots_and_table()
#Tester3.export_plots_and_table()
Tester4.export_plots_and_table()
    
    
    
    
    
    
    
    