import pandas as pd
from random import randint
import heapq
import numpy

data = pd.read_csv('CS170_SMALLtestdata__110.txt',header=None, sep="  ", engine='python')

def main():

	print("Welcome to Maaz Mohamedy's Feature Selection Algorithm.")
	file  = 1#input("Type in the name of the file to test (1 for CS170_SMALLtestdata__60, "
		#+ "2 for CS170_LARGEtestdata__60: ")

	# data = pd.read_csv('CS170_SMALLtestdata__110.txt',header=None, sep="  ", engine='python')

	if file == "2":
		data = pd.read_csv('CS170_LARGEtestdata__96.txt',header=None, sep="  ", engine='python')

	# print(data.head())

	algo ="1"# input("Type the number of the algorithm you want to run.\n\n"
		#+ "\t1) Forward Selection\n"
		#+ "\t2) Backward Elimination\n"
		#+ "\t3) Maaz's Special Algorithm\n\n\n")

	# if (algo == "1"): forwardSelection(data)
	if (algo == "1"): forwardSelection()


def forwardSelection():
	print()
	current_set_of_features = []
	current_set_of_features.append([])

	numFeatures = len(data.columns)
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
				accuracy = leave_one_out_cross_validation(testSet)

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

def leave_one_out_cross_validation(current_set_of_features):
	allNeighbors = []
	classOne = 0
	classTwo = 0
	classification = -1

	numInstances = len(data.index)
	numCorrectClassifications = 0

	for i in range(0, numInstances):
		allNeighbors.clear()
		for j in range(0, numInstances):
			if not(i == j):
				# calculateDistance between i and j using curr set of features
				# throw 'j' into pq -- (distance, class)
				distance = calculateDistance(i,j,current_set_of_features)
				heapq.heappush(allNeighbors, (distance, data.loc[j,0]) )

		if data.loc[i,0] == heapq.heappop(allNeighbors)[1]:
			numCorrectClassifications += 1

	# print("numCorrectClassifications" +str(numCorrectClassifications))
	# print("numInstances" +str(numInstances))
	return (numCorrectClassifications/numInstances)


def calculateDistance(i,j,features):
	iVec = []
	jVec = []

	for k in range(0, len(features)):
		iVec.append(data.loc[i,features[k]])
		jVec.append(data.loc[j,features[k]])

	a = numpy.array(iVec)
	b = numpy.array(jVec)

	res = (numpy.linalg.norm(a-b))
	return res

if __name__ == "__main__":
	main()
