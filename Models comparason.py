# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 03:12:17 2022

@author: river
"""
import numpy as np
import matplotlib.pyplot as plt
models_evaluation_dictionary = {}

# models_evaluation_dictionary["LR_BTC"] = (324.46, 373.6, 0.9929592674492113, 0.9763780632972102)
# models_evaluation_dictionary["LR_BTC_sentiment"] = (319.66, 364.2,0.9931659715774585, 0.9775534841016149)

# models_evaluation_dictionary["RF_BTC"] = (130.63, 543.3, 0.9988587825043616, 0.9500471116556298)
# models_evaluation_dictionary["RF_BTC_sentiment"]  = (162.95, 458.21, 0.9982241848275566, 0.9644779299560987)

# models_evaluation_dictionary["XGBoost_BTC"] = (311.47, 1754.52, 0.9967454895979643, 0.4791917003451165)
# models_evaluation_dictionary["XGBoost_BTC_sentiment"]  = (217.70, 601.52, 0.9968304022163204, 0.9387842973492698)

# models_evaluation_dictionary["LSTM_BTC"] = (331.83 ,380.81, 0.9926362534209957, 0.975465891616951) 
# models_evaluation_dictionary["LSTM_BTC_sentiment"] = (428.84 ,652.10 ,0.9877010425293345, 0.9280579077975442)

models_evaluation_dictionary["LSTM"] = (115244.37 ,130535.40, 339.48, 358.32) 

models_evaluation_dictionary["LR"] = (134353.62, 135241.35, 366.54, 367.75)

models_evaluation_dictionary["RF"] = (237984.07, 192979.29, 487.83, 439.29)

models_evaluation_dictionary["XGBoost"] = (474237.91, 351862.82, 688.65, 593.18)


keys = models_evaluation_dictionary.keys()
keys = list(keys)

x_axes_labels = keys

x = np.arange(len(x_axes_labels))  # the label locations
  # the width of the bars
MSE_BTC =[]
MSE_BTC_SA = []
RMSE_BTC =[]
RMSE_BTC_SA = []

for key in keys:  
  MSE_BTC.append(models_evaluation_dictionary[key][0])
  MSE_BTC_SA.append(models_evaluation_dictionary[key][1])
  
  
  RMSE_BTC.append(models_evaluation_dictionary[key][2])
  RMSE_BTC_SA.append(models_evaluation_dictionary[key][3])
  
width = 0.3
fig, ax = plt.subplots(figsize=(9,6))
  # best_k, max_silhouette,cohen_kappa_scores,homogeneity,completeness,VM
rects1 = ax.bar(x - width/2, MSE_BTC, width, label='BTC Only')
rects2 = ax.bar(x + width/2, MSE_BTC_SA, width, label='BTC & Sentiment')
#rects4 = ax.bar(x + width,train_r2, width, label='train_r2')
#rects5 = ax.bar(x + 2*width, test_r2, width, label='test_r2')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('MSE')
ax.set_xlabel('modles')
ax.set_title('Models comarison')
ax.set_xticks(x, x_axes_labels)
ax.legend()
ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)
fig.tight_layout()
plt.show()


fig, ax = plt.subplots(figsize=(9,6))
rects4 = ax.bar(x - width/2 ,RMSE_BTC, width, label='BTC Only')
rects5 = ax.bar(x + width/2, RMSE_BTC_SA, width, label='BTC & Sentiment')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('RMSE')
ax.set_xlabel('modles')
ax.set_title('Models comarison')
ax.set_xticks(x, x_axes_labels )
ax.legend()
ax.bar_label(rects4, padding=3)
ax.bar_label(rects5, padding=3)
# ax.set_ylim(30,110)
fig.tight_layout()
plt.show()

#///////////////////////////////////////////////////////////////////////////
# Date Jul 21, 2022 to predict Date Jul 22, 2022

Sentiment_Score = 0.101367
neg = 0.053688
neu = 0.848501
pos = 0.097791
close = 23164.63	;volume = 33631012204
actual = 22714.98
models_prediction_for_day={}
#models_prediction_for_day["Jul 21, test"] = (close)
models_prediction_for_day["Jul 22, Actual"] = (actual)

models_prediction_for_day["LSTM_without_sentiment"] = (21290.996) 
models_prediction_for_day["LSTM_with_sentiment"] = (21222.879)

models_prediction_for_day["LR_without_sentiment"] = (22992.391)
models_prediction_for_day["LR_with_sentiment"] = (22588.716)

models_prediction_for_day["RF_without_sentiment"] = (17776.699)
models_prediction_for_day["RF_with_sentiment"]  = (17372.590)

models_prediction_for_day["AuML_without_sentiment"] = (17421.415)
models_prediction_for_day["AuML_with_sentiment"]  = (20094.344)


keys = models_prediction_for_day.keys()
keys = list(keys)

x_axes_labels = keys


x = np.arange(len(x_axes_labels))  # the label locations

width = 0.4
fig, ax = plt.subplots(figsize=(13,8))
  # best_k, max_silhouette,cohen_kappa_scores,homogeneity,completeness,VM
rects1 = ax.bar(x[0] , list(models_prediction_for_day.values())[0], width, color = "r")
rects2 = ax.bar(x[1:] , list(models_prediction_for_day.values())[1:], width)
#rects4 = ax.bar(x + width,train_r2, width, label='train_r2')
#rects5 = ax.bar(x + 2*width, test_r2, width, label='test_r2')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_title('Predicting the close of Jul 22,2022')
ax.set_ylabel('Close')
ax.set_xlabel('Models')
ax.set_xticks(x, x_axes_labels , rotation=10)
ax.legend()
ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)
fig.tight_layout()
plt.show()




fig, ax = plt.subplots()
b1 = ax.bar(x, models_prediction_for_day.values())

plt.bar_label(b1, fmt='%.2f')


#ax.set_ylim(23000,23400)
ax.set_title('Predicting the close of Jul 22,2022')
ax.set_ylabel('Close')
ax.set_xlabel('Models')
ax.set_xticks(x, x_axes_labels , rotation=45)
ax.legend()
fig.tight_layout()
plt.show()


