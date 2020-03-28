# importing libraries
import matplotlib.pyplot as plt 
import pandas as pd

actual_d1 = pd.read_csv("actual.csv")
predicted_d2 = pd.read_csv("predicted.csv")

plt.style.use('seaborn-darkgrid')
plt.title("Deviation")		#title of the graph
plt.plot(actual_d1, label = "actual", color = "blue", linewidth = 2, marker = "o")	
plt.axis([0, 10, 0, 300])
plt.plot(predicted_d2, label = "predicted", color = "red", linewidth = 1, marker = "o")	
plt.xlabel("x-axis", fontsize = 16)		#labelname on x-axis
plt.ylabel("y-axis", fontsize = 16)		#labelname on y-axis
plt.legend()	#to show the legends on the map
#plt.xlim(0,10)
plt.axis([0, 10, 0, 300])	# FIRST 2 VALUES IS FOR X-AXIS, NEXT 2 FOR Y-AXIS
plt.tick_params(axis = "both", labelsize = 14)
plt.show()		#to show graph on the screen
plt.close()
