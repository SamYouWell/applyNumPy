import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

#Loading the data into a structured array:
mta14_df = np.genfromtxt('Turnstile_Usage_Data__2014.csv', delimiter = ',', dtype=None, names = True, encoding = None, usecols = [1,3,4,5,6,7,8,9,10])

#Converting into a recarray:
rec_mta14 = np.rec.array(mta14_df, dtype=mta14_df.dtype)

#Isolating a station for analysis:
sferry = rec_mta14[rec_mta14.Station == 'SOUTH FERRY']
#What is the covariance of Entries to Exits from South Ferry?:
cov_sferry_e2e = np.cov(sferry.Entries, sferry.Exits)
print(cov_sferry_e2e)
#The cov matrix:    [[6.39379132e+12 4.63837331e+12]    [[x-entry x-entryexit]
#                    [4.63837331e+12 4.46460158e+12]]    [x-exityentry x-exit]]
#What do these numbers really mean? We need to calculate Pearson correlation!
cor_sferry_e2e = np.corrcoef(sferry.Entries, sferry.Exits)
print('South Ferry Entries & Exits P Correlation: ', cor_sferry_e2e)
#South Ferry Entries & Exits P Correlation:    [[1.         0.86815069]
#                                               [0.86815069 1.        ]]
#Calculating the linear regression (i.e. a and b)
lr_sferry_e2e = np.polyfit(sferry.Entries, sferry.Exits, 1)
print(lr_sferry_e2e)
sferry_predict = np.poly1d(lr_sferry_e2e)
print('South Ferry prediction for x = 2,633,000: ', sferry_predict(2633000))    #--> Prediction = 2292338.58; ApproxActual value = 612000
#Calculate the accuracy of model:
print('Accuracy of South Ferry prediction: ', r2_score(sferry.Entries, sferry_predict(sferry.Exits)))   #--> Accuracy = 0.6424802446370217, so only OK...not really usable


#*What is the cov of the overall entry and exit? What is the variance of people maybe jumping the turnstyle?* 
#(below is not an answer to the question because 1) cov is only a lunching point and 2) Entries and Exits are evidently independent of each other):
cov_entry2exit = np.cov(rec_mta14.Entries, rec_mta14.Exits)
#print(cov_entry2exit)
#The cov matrix:   [[1.16102286e+16 8.18445012e+15]    [[x-entry x-entryexit]
#                   [8.18445012e+15 9.35848716e+15]]    [x-exityentry x-exit]]
#There appears to be a lot of variance within just Exits, which makes sense given that people are more likely to have more varied destinations than origins. 
#But we also need to calculate Pearson correlation:
cor_entry2exit = np.corrcoef(rec_mta14.Entries, rec_mta14.Exits)
print('Overall correlation between Entries and Exits: ', cor_entry2exit)
#Overall correlation between Entries and Exits:    [[1.        0.7851747]
#                                                   [0.7851747 1.       ]]
#Calculating the linear regression:
lr_mta14_e2e = np.polyfit(rec_mta14.Entries, rec_mta14.Exits, 1)
print(lr_mta14_e2e)
mta14_predict = np.poly1d(lr_mta14_e2e)
print('Overall mta14 prediction of x = 300,000: ', mta14_predict(300300))   #--> Prediction = 1069257.90; ApproxActual value = 158000
#Calculate the accuracy of model:
print('Accuracy of overall mta prediction model: ', r2_score(rec_mta14.Exits, mta14_predict(rec_mta14.Entries)))    #--> Accuracy = 0.6164993050213795, so only OK, not usable

#^^^^^WE ARE GOING TO HAVE TO CHANGE FINE TUNE PREDICTOR AND TARGET VALUES...maybe what needs to happen is to track things that really more related to each other (as is, these two variables are pretty indepent of each other):
# e.g. tracking starting and terminal places that really track...some guess work and common observations and maybe find surprises.  


#ADDING TO PROJECT CONTINUALLY: next steps!
#--> How can we answer the question on Line 27? Can we answer it with the data?:
#We cannot directly track customer trip, can we? No! But we could track a specific line find (may not all) many popular routes and then track Entries and Exits!....a lot but maybe interest:
#Let's find a simple line that does not have a lot of connections (use a map to help you):
