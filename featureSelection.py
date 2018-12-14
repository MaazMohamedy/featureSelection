import numpy
import cProfile, pstats, io
import time
import heapq
import random
import profile
import csv
import copy

pr = cProfile.Profile()

def main():

	print("Welcome to Maaz Mohamedy's Feature Selection Algorithm.")
	file  = input("Type in the name of the file to test:  ") # (1 for CS170_SMALLtestdata__60.txt, " #+ "2 for CS170_LARGEtestdata__96: ")

	algo =input("\nType the number of the algorithm you want to run.\n\n"
		+ "\t1) Forward Selection\n"
		+ "\t2) Backward Elimination\n"
		+ "\t3) Maaz's Special Algorithm\n\n\n")
	
	data = numpy.loadtxt(open(file))
	data = data.tolist()
	numFeatures = len(data[0])-1
	numInstances = len(data)
	allFeatures = []
	for i in range(0,numFeatures): allFeatures.append(i+1)

	print("This dataset has " + str(numFeatures) + " features (not including the class attribute), with " + str(numInstances) + " instances.\n")

	acc = leave_one_out_cross_validation(data, allFeatures, 0, False)

	print("Running nearest neighbor with all "+ str(numFeatures) +" features, using “leaving-one-out” evaluation, I get an accuracy of " + str(acc) + "%")

	pr.enable()
	if (algo == "1"): forwardSelection(data, False)
	if (algo == "2"): backwardElimination(data)	
	if (algo == "3"): customSearch(data)		

	pr.disable()
	pr.print_stats()#end def main

def forwardSelection(data, speedUp):
	print()
	current_set_of_features = []
	current_set_of_features.append([])

	numFeatures = len(data[0])
	mostAccurate = 0
	mostAccurateSet = []
	first = True

	print("Beginning search.\n")

	for i in range(1, numFeatures):
		feature_to_add_at_this_level = 0
		best_so_far_accuracy = 0
		last = len(current_set_of_features)-1

		for j in range(1, numFeatures):
			if j not in current_set_of_features[last]:
				testSet = current_set_of_features[last][:]
				testSet.append(j)

				accuracy = leave_one_out_cross_validation(data, testSet, best_so_far_accuracy, speedUp)

				print("\tUsing feature(s) {", end = "") 
				print(*(current_set_of_features[last]), sep = ",", end = "" )
				if (len(current_set_of_features[last]) != 0): print(",", end = "")
				print(str(j) + "} accuracy is " + str("{0:.2f}".format(accuracy)) + "%")


				if (accuracy > best_so_far_accuracy):
					best_so_far_accuracy = accuracy
					feature_to_add_at_this_level = j

		new_features = current_set_of_features[last][:]
		new_features.append(feature_to_add_at_this_level)
		current_set_of_features.append(new_features)

		if (best_so_far_accuracy > mostAccurate): 
			mostAccurate = best_so_far_accuracy
			mostAccurateSet = new_features[:]
		else:
			if first:
				print("\n(Warning, Accuracy has decreased! Continuing search in case of local maxima)")
				first = False

		print("Feature set {", end="")
		print(*(current_set_of_features[last]), sep = ",", end = "" )
		if (len(current_set_of_features[last]) != 0): print(",", end = "")
		print(str(feature_to_add_at_this_level), end = "")
		print("} was best, accuracy is " + str(best_so_far_accuracy) + "%\n")

	print("Finished search!! The best feature subset is {", end = "")
	print(*(mostAccurateSet), sep = ",", end = "" )
	print("}, which has an accuracy of " + str("{0:.2f}".format(mostAccurate*100)) + "%")

	return mostAccurateSet, mostAccurate


def customSearch(data):
	results = []
	accuracies =[]
	ans = forwardSelection(data,True)[0]

	for i in range(1,4):
		modData = []
		modData = copy.deepcopy(data)
		fivePercentOfData = .05 * len(data)
		fivePercentOfData =  int(fivePercentOfData)
		
		for j in range(0,fivePercentOfData):
			rowToBeDeleted = random.randint(0,len(modData)-1)
			del modData[rowToBeDeleted]

		res = forwardSelection(modData, True)
		results.append(res[0])
		accuracies.append(res[1])

	numFeatures = len(data[0])-1
	featureAppearances = [0] * numFeatures

	# go through all results arrays and find the most common feature that does not appear in ans
	for i in range(0,len(results)):
		for j in range(0,len(results[i])-1):
			if results[i][j] not in ans:
				appearances = featureAppearances[results[i][j]-1]
				featureAppearances[results[i][j]-1] = appearances+1

	maxAppearances=0
	feature=-1
	#Find weak feature
	for i in range(0,len(featureAppearances)):
		if featureAppearances[i] > maxAppearances:
			maxAppearances = featureAppearances[i]
			feature = i+1

	if feature != -1:
		ans.append(feature)

	print("\n\nFinished search!! The best feature subset is {", end = "")
	print(*(ans), sep = ",", end = "" )
	x = max(accuracies)
	print("}, which has an accuracy of " + str("{0:.2f}".format(x*100)) + "%")


def backwardElimination(data):
	print()
	numFeatures = len(data[0])
	current_set_of_features = []
	mostAccurate = 0
	for i in range(1,numFeatures): current_set_of_features.append(i)
	mostAccurateSet = []

	best_so_far_accuracy = leave_one_out_cross_validation(data, current_set_of_features)

	print("Beginning search.\n")

	for i in range(1, numFeatures):
		best_so_far_accuracy = 0
		testSet = []
		bestSet = []
		for j in range(1, len(current_set_of_features)+1):
			testSet = current_set_of_features[:]
			testSet.remove(current_set_of_features[j-1])
			accuracy = leave_one_out_cross_validation(data, testSet, best_so_far_accuracy,False)

			print("\tUsing feature(s) {", end = "") 
			print(*(testSet), sep = ",", end = "" )
			print("} accuracy is " + str(accuracy) + "%")

			if (accuracy > best_so_far_accuracy):
				best_so_far_accuracy = accuracy
				bestSet = testSet[:]

		current_set_of_features = bestSet[:]

		if (best_so_far_accuracy > mostAccurate):
			mostAccurate = best_so_far_accuracy
			mostAccurateSet = current_set_of_features
		else:
			print("\n(Warning, Accuracy has decreased! Continuing search in case of local maxima)")


		print("Feature set {", end="")
		print(*(bestSet), sep = ",", end = "")
		print("} was best, accuracy is " + str(best_so_far_accuracy) + "%\n")

	print("Finished search!! The best feature subset is {", end = "")
	print(*(mostAccurateSet), sep = ",", end = "" )
	print("}, which has an accuracy of " + str(mostAccurate*100) + "%")
	return mostAccurateSet

def leave_one_out_cross_validation(data, testSet, best_so_far_accuracy,speedUp):
	minDistance = float('inf')
	nearest = -1
	numWrong = 0

	numInstances = len(data)
	numCorrectClassifications = 0

	for i in range(0, numInstances):
		nearest = -1
		minDistance = float('inf')
		for j in range(0, numInstances):
			if not(i == j):
				# calculateDistance between i and j using curr set of features
				distance = 0
				for k in range(0, len(testSet)):
					a = data[i][testSet[k]]
					b = data[j][testSet[k]]
					x = a-b
					distance += x*x

				if (distance <= minDistance):
					nearest = data[j][0]#data.iat[j,0]
					minDistance = distance

		if data[i][0] == nearest: numCorrectClassifications += 1
		else: numWrong += 1

		if speedUp == True:
			if numWrong > (numInstances-(best_so_far_accuracy*numInstances)): return 0

	return (numCorrectClassifications/numInstances)

if __name__ == "__main__":
	main()