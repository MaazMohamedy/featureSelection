import numpy
import cProfile, pstats, io
import time
import profile
import csv

pr = cProfile.Profile()
pr.enable()

def main():

	print("Welcome to Maaz Mohamedy's Feature Selection Algorithm.")
	file  = 1#input("Type in the name of the file to test (1 for CS170_SMALLtestdata__60, "
		#+ "2 for CS170_LARGEtestdata__60: ")

	# data = pd.read_csv('CS170_SMALLtestdata__110.txt',header=None, sep="  ", engine='python')

	# if file == "2":
	# 	data = pd.read_csv('CS170_LARGEtestdata__96.txt',header=None, sep="  ", engine='python')

	algo ="1"# input("Type the number of the algorithm you want to run.\n\n"
		#+ "\t1) Forward Selection\n"
		#+ "\t2) Backward Elimination\n"
		#+ "\t3) Maaz's Special Algorithm\n\n\n")
	
	data = numpy.loadtxt(open("CS170_SMALLtestdata__60.txt"))
	data = data.tolist()

	# for i in range(1,11):
	# 	if ( not (i==6) and not (i==9 )):
	# 		arr = [6,9]
	# 		arr.append(i)
	# 		res = leave_one_out_cross_validation(arr)#data,arr)
	# 		print(res)

	if (algo == "1"): forwardSelection(data)

	pr.disable()
	pr.print_stats()#end def main


def forwardSelection(data):
	print()
	current_set_of_features = []
	current_set_of_features.append([])

	numFeatures = len(data[0])#len(data.columns)
	mostAccurate = 0
	mostAccurateSet = []

	print("Beginning search.\n")

	for i in range(1, numFeatures):
		feature_to_add_at_this_level = 0
		best_so_far_accuracy = 0
		last = len(current_set_of_features)-1

		for j in range(1, numFeatures):
			if j not in current_set_of_features[last]:
				testSet = current_set_of_features[last][:]
				testSet.append(j)

				accuracy = leave_one_out_cross_validation(data, testSet)

				print("\tUsing feature(s) {", end = "") 
				print(*(current_set_of_features[last]), sep = ",", end = "" )
				if (len(current_set_of_features[last]) != 0): print(",", end = "")
				print(str(j) + "} accuracy is " + str(accuracy) + "%")

				if (accuracy > best_so_far_accuracy):
					best_so_far_accuracy = accuracy
					feature_to_add_at_this_level = j

		new_features = current_set_of_features[last][:]
		new_features.append(feature_to_add_at_this_level)
		current_set_of_features.append(new_features)

		if (best_so_far_accuracy > mostAccurate): 
			mostAccurate = best_so_far_accuracy
			mostAccurateSet = new_features[:]

		print("\nFeature set {", end="")
		print(*(current_set_of_features[last]), sep = ",", end = "" )
		if (len(current_set_of_features[last]) != 0): print(",", end = "")
		print(str(feature_to_add_at_this_level), end = "")
		print("} was best, accuracy is " + str(best_so_far_accuracy) + "%\n")

	print("mostAccurate " + str(mostAccurate))
	print("mostAccurateSet " + str(mostAccurateSet))

def leave_one_out_cross_validation(data, testSet):
	# allNeighbors = []
	minDistance = float('inf')
	nearest = -1

	numInstances = len(data)#len(data.index)
	numCorrectClassifications = 0

	for i in range(0, numInstances):
		# allNeighbors.clear()
		nearest = -1
		minDistance = float('inf')
		for j in range(0, numInstances):
			if not(i == j):
				# calculateDistance between i and j using curr set of features
				# throw 'j' into pq -- (distance, class)
				distance = 0
				for k in range(0, len(testSet)):
					a = data[i][testSet[k]]
					b = data[j][testSet[k]]
					x = a-b
					distance += x*x
					# distance += calculateDistance(data,testSet,i,j,k)
					# distance = distance + pow((data[i][testSet[k]]) - ((data[j][testSet[k]])), 2)

				if (distance <= minDistance):
					nearest = data[j][0]#data.iat[j,0]
					minDistance = distance

				# heapq.heappush(allNeighbors, (distance, data[j][0]))#data.iat[j,0]) )


		# if data[i][0] == heapq.heappop(allNeighbors)[1]: numCorrectClassifications += 1

		if data[i][0] == nearest:
			numCorrectClassifications += 1

	return (numCorrectClassifications/numInstances)



if __name__ == "__main__":
	main()
