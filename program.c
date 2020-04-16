#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h> 

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include "mpi.h"

#include "program.h"

#define USAGE_MSG "[ERROR] Invalid input arguments. Arguments should be:\n\
	1. dataset name\n\
	2. number of features to sample (-1 for default)\n\
	3. number of trees to create\n\
	4. max decision tree depth\n"

void print_tree(Node *node, int depth){

	char *indentation = (char *) malloc(2 * depth * sizeof(char));
	char *whitespace = " ";

	int i;
	for (i = 0; i < 2 * depth; i++){
		strcpy(&indentation[i], whitespace);
	}

	if (node->isLeaf){
		printf("%s[Leaf] Label: %d\n", indentation, node->label);
		free(indentation);
	}
	else {
		printf("%s[Node] Col: %d | Val: %f\n", indentation, node->colIndex, node->value);
		free(indentation);
		print_tree(node->left, depth + 1);
		print_tree(node->right, depth + 1);
	}
}

void print_dataset_row(int numRows, int numCols, double *dataset, int index){
	if (numRows < index){
		printf("Cannot print row %d when there are %d rows...\n", index, numRows);
	}

	printf("\nRow %d\n", index);
	int i;
	for (i = 0; i < numCols; i++){
		printf("%.4lf,", dataset[index * numCols + i]);
	}
	printf("\n\n");
}

int majority_class(int numRows, int *classLabels, int numClasses){
	int counters[numClasses];
	
	int i;
	for (i = 0; i < numClasses; i++){
		counters[i] = 0;
	} 

	
	for (i = 0; i < numRows; i++){
		if (classLabels[i] > 1){
			printf("Class label %d\n", classLabels[i]);
		}
		counters[classLabels[i]]++;
	}

	int maxCount = -1, maxCountInd;
	for (i = 0; i < numClasses; i++){
		if (counters[i] > maxCount){
			maxCount = counters[i];
			maxCountInd = i;
		}
	}

	return maxCountInd;

}

/* 
 * For an input dataset, select a number of sample features from all of the
 * features present. For that subsample, calculate gini indices for each
 * possible value, for each feature in the subsample.
 * 
 * Za neki skup podataka potrebno je proci kroz odredjeni podskup feature-a
 * i odrediti za koju vrednost bilo kojeg feature-a cemo dobiti najbolju
 * podelu skupa podataka na klase (pod najbolju podelu misli se na 
 * onu vrednost koja najbolje deli dataset)
 *
 * @param dataset The dataset matrix
 * @param numFeatures number of features in the dataset   | The dataset 
 * @param numInstances number of instances in the dataset | matrix dimensions
 * @param numSampleFeatures The number of features to consider in this split
 */
Node * get_split(
		int numRows, 
		int numCols, 
		double *dataset, 
		int *labels,
		int numClasses,
		int numSelectedFeatureColumns,
		int currentDepth){


	if (numRows < 5){
		// Too few instances, return the majority class
		Node *leaf = (Node *) malloc(sizeof(Node));
		int majorityClass;

		majorityClass = majority_class(numRows, labels, numClasses);

		leaf->isLeaf = 1;
		leaf->label = majorityClass;

		return leaf;
	}

	////////////////////////////
	// SELECT RANDOM FEATURES //
	////////////////////////////

	int *selectedColumns = (int *) malloc(numSelectedFeatureColumns * sizeof(int));
	
	// Initialize selected columns to -1
	int cntr;
	for (cntr = 0; cntr < numSelectedFeatureColumns; cntr++){
		selectedColumns[cntr] = -1;
	}

	cntr = 0;
	int j;
	int indexToSelect;
	int alreadySelected;
	// Select random columns to take into consideration
	while (cntr < numSelectedFeatureColumns){
		
		alreadySelected = 0;

		indexToSelect = rand() % numCols;

		for (j = 0; j <= cntr; j++){
			if (selectedColumns[j] == indexToSelect){
				// Column is already selected
				alreadySelected = 1;
				break;
			}
		}

		if (!alreadySelected){
			// If we've picked a new column
			selectedColumns[cntr] = indexToSelect;

			cntr++;
		}

	}


	
	// int *featureColPtr = *selectedFeatureColumns;
	float gini = 10000.0; // Big enough number
	float score;

	int featureIndex, row, innerRow;
	int bestFeatureIndex = -1;
	int feature, nLeft, nRight;
	int *leftLabels;
	int *rightLabels;
	double currSplitValue, bestSplitValue;

	int bestNLeft, bestNRight;
	int *bestLeftLabels = NULL, *bestRightLabels = NULL;

	// We'll use a map to track which rows belong to the
	// left split and which belong to the right
	// if map[index] == 0 -> it belongs to the left split
	// else if       == 1 -> it belongs to the right split 
	int splitMap[numRows];

	for (featureIndex = 0; featureIndex < numSelectedFeatureColumns; featureIndex++){

		feature = selectedColumns[featureIndex];

		// For each row, calculate the gini score
		// of the split with the value at that row
		// dataset[row * numCols + feature]
		for (row = 0; row < numRows; row++){
			currSplitValue = dataset[row * numCols + feature];

			// Test this split:
			nLeft = 0;
			nRight = 0;
			for (innerRow = 0; innerRow < numRows; innerRow++){
				if (dataset[innerRow * numCols + feature] <= currSplitValue){
					nLeft++;
					splitMap[innerRow] = 0;
				}
				else {
					nRight++;
					splitMap[innerRow] = 1;
				}
			}


			// Use the feature map to create arrays for labels
			// placed in the left and right groups

			leftLabels = (int *) calloc(nLeft, sizeof(int));
			rightLabels = (int *) calloc(nRight, sizeof(int));

			int lCount = 0, 
				rCount = 0;

			for (innerRow = 0; innerRow < numRows; innerRow++){
				if (splitMap[innerRow] == 0){
					leftLabels[lCount++] = labels[innerRow];
				}
				else {
					rightLabels[rCount++] = labels[innerRow];
				}
			}

			// Calculate the gini index
			score = gini_index_2(nLeft, leftLabels, nRight, rightLabels, numClasses);

			// If it is less than the last gini index,
			// set the currentSplitValue as the bestSplitValue
			// and update the minimum gini index

			if (score < gini){
				bestFeatureIndex = featureIndex;
				bestSplitValue = currSplitValue;
				gini = score;
				bestNLeft = lCount;
				bestNRight = rCount;

				if (bestLeftLabels != NULL){
					free(bestLeftLabels);
					free(bestRightLabels);
					
				}

				bestLeftLabels = leftLabels;
				bestRightLabels = rightLabels;


				leftLabels = NULL;
				rightLabels = NULL;
			} // END IF
		} // End for (on rows)
	} // End for (on features)

	// After we've found the best feature based on the Gini Impurity 
	// we have to remove that feature from the feature pool
	// featureColPtr[bestFeatureIndex] = -1;

	// If the number of elements on either side of the split
	// is 0, then this node becomes a leaf node

	if (nLeft == 0 || nRight == 0){
		Node *leaf = (Node *) malloc(sizeof(Node));
		int majorityClass;

		if (nLeft){
			majorityClass = majority_class(bestNLeft, bestLeftLabels, numClasses);
		}
		else {
			majorityClass = majority_class(bestNRight, bestRightLabels, numClasses);
		}

		leaf->isLeaf = 1;
		leaf->label = majorityClass;

		return leaf;
	}

	// If we've reached depth 1, then we split for the last time, and
	// the children become terminal nodes
	if (currentDepth == 1) {
		Node *lastParent = (Node *) malloc(sizeof(Node));

		lastParent->colIndex = selectedColumns[bestFeatureIndex];
		lastParent->value = bestSplitValue;
		lastParent->isLeaf = 0;

		Node *leafLeft = (Node *) malloc(sizeof(Node));
		Node *leafRight = (Node *) malloc(sizeof(Node));


		int majorityLeft = majority_class(bestNLeft, bestLeftLabels, numClasses);
		int majorityRight = majority_class(bestNRight, bestRightLabels, numClasses);

		// These 2 leaf nodes could be merged into one... TODO

		leafLeft->isLeaf = 1;
		leafLeft->label = majorityLeft;

		leafRight->isLeaf = 1;
		leafRight->label = majorityRight;

		lastParent->left = leafLeft;
		lastParent->right = leafRight;

		return lastParent;
	}

	// If we are not at max depth and we are not a leaf:

	Node *current = (Node *) malloc(sizeof(Node));

	current->colIndex = selectedColumns[bestFeatureIndex];
	current->value = bestSplitValue;
	current->isLeaf = 0;

	///////////////////////////////////////////
	// SPLIT THE DATASET INTO LEFT AND RIGHT //
	///////////////////////////////////////////

	// *Could be done earlier, in the big for loop

	double *leftSplitDataset;
	double *rightSplitDataset;
	int *leftSplitLabels;
	int *rightSplitLabels;
	int numInLeftDataset = 0;
	int numInRightDataset = 0;

	// 0 = goes left
	// 1 = goes right
	int leftRightMap[numRows];
	int i;
	for (i = 0; i < numRows; i++){
		if (dataset[i * numCols + current->colIndex] <= current->value){
			// Goes left...
			leftRightMap[i] = 0;
			numInLeftDataset++;
		}
		else {
			// Goes right...
			leftRightMap[i] = 1;
			numInRightDataset++;
		}
	}

	leftSplitDataset = (double *) malloc(numInLeftDataset * numCols * sizeof(double));
	rightSplitDataset = (double *) malloc(numInRightDataset * numCols * sizeof(double));
	leftSplitLabels = (int *) malloc(numInLeftDataset * sizeof(int));
	rightSplitLabels = (int *) malloc(numInRightDataset * sizeof(int));


	int leftFillCount = 0;
	int rightFillCount = 0;
	for (i = 0; i < numRows; i++){
		int valInd;
		if (leftRightMap[i] == 0){
			for (valInd = 0; valInd < numCols; valInd++){
				leftSplitDataset[leftFillCount * numCols + valInd] = 
					dataset[i * numCols + valInd];
			}
			// TODO: Are labels ok?
			leftSplitLabels[leftFillCount] = labels[i];
			leftFillCount++;
		}
		else {
			for (valInd = 0; valInd < numCols; valInd++){
				rightSplitDataset[rightFillCount * numCols + valInd] =
					dataset[i * numCols + valInd];
			}
			rightSplitLabels[rightFillCount] = labels[i];
			rightFillCount++;
		}
	}

	Node *leftChild = get_split(
		numInLeftDataset,
		numCols,
		leftSplitDataset,
		leftSplitLabels,
		numClasses,
		numSelectedFeatureColumns,
		currentDepth - 1);

	Node *rightChild = get_split(
		numInRightDataset,
		numCols,
		rightSplitDataset,
		rightSplitLabels,
		numClasses,
		numSelectedFeatureColumns,
		currentDepth - 1);

	////////////////////////
	// WITH SPLITTING END //
	////////////////////////

	current->left = leftChild;
	current->right = rightChild;

	free(leftSplitDataset);
	free(leftSplitLabels);

	free(rightSplitDataset);
	free(rightSplitLabels);

	return current;


}

Node * create_tree(
		int numRows,
		int numCols,
		double *dataset,
		int *labels,
		int numClasses,
		int maxDepth,
		int numFeatures){

	/////////////////
	// CREATE TREE //
	/////////////////


	Node *root = get_split(
		numRows,
		numCols,
		dataset,
		labels,
		numClasses,
		numFeatures,
		maxDepth);


	return root;

}

/*
 * Read a csv file, return the number of rows.
 * TODO: Check if all rows have the same number of columns
 */
int read_csv(
		char *fname, 
		double **dataset, 
		int *numFeatures, 
		int *numLabels,
		LabelMap **labelMap,
		int **intLabelsIn){
	FILE *f;
	f = fopen(fname, "r");

	
	// If the file exists
	if (f){
		char line[2048];

		// Read num of instances
		fgets(line, sizeof(line), f);

		int numInstances = (int) strtol(line, NULL, 10);

		printf("Num instances %d\n", numInstances);

		// Read num of features
		fgets(line, sizeof(line), f);

		*numFeatures = (int) strtol(line, NULL, 10);

		printf("Num of features %d\n", *numFeatures);

		// Read num of labels, and initialize LabelMap
		fgets(line, sizeof(line), f);

		*numLabels = (int) strtol(line, NULL, 10);

		printf("Num of labels %d\n", *numLabels);

		LabelMap lblMap[*numLabels];

		*labelMap = lblMap;

		// int intLabels[numInstances];

		// *intLabelsIn = intLabels;

		int *intLabels = (int *) malloc(numInstances * sizeof(int));

		*intLabelsIn = intLabels;

		// Initialize the label map
		int labelCnt;
		for (labelCnt = 0; labelCnt < *numLabels; labelCnt++){
			strcpy(lblMap[labelCnt].name, "\0");
			lblMap[labelCnt].value = labelCnt;
		}

		///////////////////////////////////
		// INITIALIZE THE DATASET MATRIX //
		///////////////////////////////////

		(*dataset) = (double *) calloc(*numFeatures * numInstances, sizeof(double));

		// Pretpostavka da je fajl dobro formatiran,
		// tj da je tacan broj redova
		int row;
		for (row = 0; row < numInstances; row++){

			/////////////////////////
			// EXTRACT DATASET ROW //
			/////////////////////////

			// Read next line
			fgets(line, sizeof(line), f);

			// Init columnCount, currVal == row[index], token
			int columnCount = 0;
			double currVal;
			char *token;

			token = strtok(line, ",");
			while (token != NULL && columnCount < *numFeatures){
				sscanf(token, "%lf", &currVal);
				(*dataset)[row * (*numFeatures) + columnCount] = currVal;

				token = strtok(NULL, ",\n");
				columnCount++;
			}

			///////////////////////
			// EXTRACT ROW LABEL //
			///////////////////////

			// Read the label, and insert it into the label map,
			// the token is ALREADY read in the last iteration of
			// the previous while loop

			int currentRowIntLabel;

			int i;
			for (i = 0; i < *numLabels; i++){
				char *labelToken = lblMap[i].name;

				if (strcmp(labelToken, "\0") == 0){
					// Insert new label
					strcpy(lblMap[i].name, token);
					currentRowIntLabel = lblMap[i].value;
					break;
				}
				else if (strcmp(token, labelToken) == 0){
					currentRowIntLabel = lblMap[i].value;
					break;
				}
			}

			intLabels[row] = currentRowIntLabel;

		}

		// Close the file
		fclose(f);

		return numInstances;
	}
	else 
		return -1;
}

/*
 * Returns round(numRows * ratio) number of subsamples from a dataset
 */
int get_dataset_sample(
		int numRows, 
		int numColumns, 
		double *dataset,
		int *labels,
		float ratio, 
		double **datasetSample,
		int **sampleLabels){

	int numRetEls = round((numRows * ratio));

	*datasetSample = (double *) calloc(numRetEls * numColumns, sizeof(double));
	*sampleLabels = (int *) calloc(numRetEls, sizeof(int));

	int count;
	for (count = 0; count < numRetEls; count++){
		// Copy numRetEls into the dataset sample array
		int columnIndex;

		int randElementIndex = rand() % numRows;

		//////////////////
		// COPY THE ROW //
		//////////////////

		for (columnIndex = 0; columnIndex < numColumns; columnIndex++){
			// Copy all the columns for each row
			(*datasetSample)[count * numColumns + columnIndex] = dataset[randElementIndex * numColumns + columnIndex];
		}

		////////////////////
		// COPY THE LABEL //
		////////////////////

		(*sampleLabels)[count] = labels[randElementIndex];

	}

	return numRetEls;

}

void shuffle_dataset(int numInstances, long int numCols, double **dataset, int **labels){

	int *lblPointer = *labels;


	int tempInt;
	int i;
	double temp[numCols];
	for (i = 0; i < numInstances; i++){
		
		int randElementIndex = rand() % numInstances;

		int colIndex;

		for (colIndex = 0; colIndex < numCols; colIndex++){
			temp[colIndex] = (*dataset)[i * numCols + colIndex];
		}

		for (colIndex = 0; colIndex < numCols; colIndex++){
			(*dataset)[i * numCols + colIndex] = (*dataset)[randElementIndex * numCols + colIndex];
		}

		for (colIndex = 0; colIndex < numCols; colIndex++){
			(*dataset)[randElementIndex * numCols + colIndex] = temp[colIndex];
		}

		// Shuffle labels the same way
		tempInt = lblPointer[i];

		lblPointer[i] = lblPointer[randElementIndex];

		lblPointer[randElementIndex] = tempInt;

	}

}

/*
 * Method: Gini Impurity 
 * Calculates the "purity" of the split. A node splits creating 2 possible paths
 * in the decision tree. 
 *
 * @param nLeft Number of elements in the left side of the split
 * @param leftLabels Labels of the elements in the left split
 * @param nRight Number of elements in the right side of the split
 * @param rightLabels Labels of the elements in the right split
 * @param numClasses Number of classes
 */
float gini_index_2(int nLeft, int *leftLabels, int nRight, int *rightLabels, int numClasses){

	int totalNumInstances = nLeft + nRight;

	float gini = 0.0;

	int i, labelMarker;

	int maxIter;

	if (nRight > nLeft){
		maxIter = nRight;
	}
	else {
		maxIter = nLeft;
	}
	
	float score = 0.0;
	float p = 0.0;
	int count = 0;

	int leftLblCount[2] = {0, 0};
	int rightLblCount[2] = {0, 0};

	for (i = 0; i < maxIter; i++){

		if (i < nLeft){
			leftLblCount[leftLabels[i]]++;
		}

		if (i < nRight){
			rightLblCount[rightLabels[i]]++;
		}

	}

	float scoreLeft = 0.0, scoreRight = 0.0;

	if (nLeft){
		p = (float) leftLblCount[0] / nLeft;
		scoreLeft += (p * p);
		p = (float) leftLblCount[1] / nLeft;
		scoreLeft += (p * p);
	}

	if (nRight){
		p = (float) rightLblCount[0] / nRight;
		scoreRight += (p * p);
		p = (float) rightLblCount[1] / nRight;
		scoreRight += (p * p);
	}

	gini += ((1.0 - scoreLeft) * ((float) nLeft / totalNumInstances));
	gini += ((1.0 - scoreRight) * ((float) nRight / totalNumInstances));


	return gini;
}

int split_dataset(
		int numRows, 
		int numCols, 
		double *dataset, 
		int *labels, 
		float ratio,
		double **train_data,
		int **train_labels,
		double **validation_data,
		int **validation_labels){

	if (ratio < 0.0 || ratio > 1.0){
		printf("Ratio must be between 0.0 and 1.0...\n");
	}

	int splitIndex = round(numRows * ratio);

	*train_data = (double *) calloc(splitIndex * numCols, sizeof(double));
	*train_labels = (int * ) calloc(splitIndex, sizeof(int));

	*validation_data = (double *) calloc((numRows - splitIndex) * numCols, sizeof(double));
	*validation_labels = (int *) calloc(numRows - splitIndex, sizeof(int));


	int row, col;
	for (row = 0; row < numRows; row++){
		if (row < splitIndex){

			for (col = 0; col < numCols; col++){
				(*train_data)[row * numCols + col] = dataset[row * numCols + col];
			}

			// (*train_data)[row] = dataset[row];
			(*train_labels)[row] = labels[row];
		}
		else {

			for (col = 0; col < numCols; col++){
				(*validation_data)[(row - splitIndex) * numCols + col] = dataset[row * numCols + col];
			}

			// (*validation_data)[row - splitIndex] = dataset[row];
			(*validation_labels)[row - splitIndex] = labels[row];
		}
	}

	return splitIndex;

}


float validate_tree(
	int numRows,
	int numColumns,
	double *validationSet,
	int *validationLabels,
	Node *root){

	int numCorrect = 0;

	Node *currNode = root;

	double observedValue;
	int predictedLabel;

	int i;
	for (i = 0; i < numRows; i++){

		while (currNode->isLeaf == 0){
			// Find a leaf

			observedValue = validationSet[i * numColumns + currNode->colIndex];

			if (observedValue <= currNode->value){
				// Go left in the tree
				currNode = currNode->left;
			}
			else {
				// Go right in the tree
				currNode = currNode->right;
			}

		}

		predictedLabel = currNode->label;

		if (predictedLabel == validationLabels[i]){
			numCorrect++;
		}

	}

	float accuracy = ((float) numCorrect) / ((float) numRows);

	return accuracy;

}


/*
 *	Gives predictions to every instance in the dataset.
 *	Returns an array of predictions.
 */
int * predict(
	int numRows, 
	int numCols, 
	double *dataset, 
	Node *root){

	int *predictions;

	predictions = (int *) malloc(numRows * sizeof(int));

	Node *currNode;

	double observedValue;

	int i;
	for (i = 0; i < numRows; i++){

		currNode = root;

		while (currNode->isLeaf == 0){

			observedValue = dataset[i * numCols + currNode->colIndex];

			if (observedValue <= currNode->value){
				// Go left in the tree
				currNode = currNode->left;
			}
			else {
				// Go right in the tree
				currNode = currNode->right;
			}

		}

		predictions[i] = currNode->label;

	}

	return predictions;

}

/*
 *	Label the instance based on the
 *	majority vote.
 */
int * get_majority_vote(
	int numTrees,
	int numInstances,
	int *matrix,
	int numClasses){

	int *voteResults = (int *) calloc(numInstances, sizeof(int));

	int vote;
	int maxVoteIndex;
	int votes[numClasses];


	int treeInd;
	int instance;
	for (instance = 0; instance < numInstances; instance++){

		
		int i;
		for (i = 0; i < numClasses; i++){
			votes[i] = 0;
		} 
		
		for (treeInd = 0; treeInd < numTrees; treeInd++){
			vote = matrix[treeInd * numInstances + instance];
			if (vote >= numClasses){
				printf("Bad label: %d", vote);
				fflush(stdout);
			}
			votes[vote]++;
		}

		maxVoteIndex = 0;
		for (i = 0; i < numClasses; i++){
			if (votes[i] > votes[maxVoteIndex]){
				maxVoteIndex = i;
			}
		}

		voteResults[instance] = maxVoteIndex;

	}

	return voteResults;

}

float get_accuracy(int n, int *labels, int *predictions){

	int numCorrect = 0;

	int i;
	for (i = 0; i < n; i++){

		printf("%d == %d!\n", labels[i], predictions[i]);
		fflush(stdout);
		if (labels[i] == predictions[i]){
			numCorrect++;
		}
	}

	return (((float) numCorrect) / ((float) n));

}

int check_input(int argc, char *argv[]){

	if (argc != 5){
		printf(USAGE_MSG);
		return 1;
	}

	char *fileName = argv[1];
	int numFeaturesToSample = atoi(argv[2]);
	int numTrees = atoi(argv[3]);
	int maxDepth = atoi(argv[4]);

	if (numFeaturesToSample <= 0
		|| numTrees <= 0
		|| maxDepth <= 0){
		printf(USAGE_MSG);
		return 1;
	}

	// Check if file exists
	if (access(fileName, F_OK) == -1){
		printf("[ERROR] \"%s\" does not exist.\n", fileName);
		return 1;
	}

	return 0;

}

void log_results(float accuracy, double time, int numTrees, int numWorkers){

	FILE *f;
	char *fname = "./log/log.txt";

	struct stat st = {0};

	if (stat("./log/", &st) == -1) {
	    mkdir("./log/", 0700);
	    fflush(stdout);
	}


	if (access(fname, F_OK) == -1){
		// Doesn't exist
		f = fopen(fname, "a");

		fprintf(f, "Num trees: %d\n\n", numTrees);

		fprintf(f, "[Workers: %d | Accuracy: %f] Time: %f\n", numWorkers, accuracy, time);
	}
	else {
		f = fopen(fname, "a");
		
		fprintf(f, "[Workers: %d | Accuracy: %f] Time: %f\n", numWorkers, accuracy, time);

	}
	
	fclose(f);

}


/*
 *	Arguments:
 *		1. dataset name
 *		2. number of features to sample (-1 for default)
 *		3. number of trees to create
 *		4. max decision tree depth
 */
int main(int argc, char *argv[]) {

	//////////////////////////
	// VARIABLE DECLARATION //
	//////////////////////////

	int myRank;
	int numWorkers;

	int numTrees;
	int maxTreeDepth;

	int numTreesPerWorker;
	int numTreesForMaster;

	int numFeaturesToSample; // Number of random features to select when training

	char *fileName;

	int numRows;
	int numCols;
	int numClasses;
	LabelMap *labelMap;

	double *dataset;
	int *labels;

	//////////////
	// MPI INIT //
	//////////////

	MPI_Init(NULL, NULL);

	MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
	MPI_Comm_size(MPI_COMM_WORLD, &numWorkers);

	if (myRank == 0){
	
		if (check_input(argc, argv) != 0){
			MPI_Finalize();
		}	


		char *fileName = argv[1];
		numFeaturesToSample = atoi(argv[2]);
		numTrees = atoi(argv[3]);
		maxTreeDepth = atoi(argv[4]);

		// Calculate number of trees per node
		numTreesPerWorker = numTrees / numWorkers;

		numTreesForMaster = numTrees - (numTreesPerWorker * (numWorkers - 1));

		printf("Number of trees per worker %d, and for the master %d\n", numTreesPerWorker, numTreesForMaster);

		// Read dataset
		numRows = read_csv(
			fileName, 
			&dataset, 
			&numCols,
			&numClasses,
			&labelMap,
			&labels
		);

		shuffle_dataset(
			numRows,
			numCols,
			&dataset,
			&labels
		);


	}

	/*
	 *	The master node should:
	 *		1. Validate input
	 *		2. Read the dataset
	 *		3. Send the dataset dimensions
	 *		4. Send the dataset to the workers
	 *		5. Send the number of trees to create
	 *		6. Send the max depth of the trees
	 *		7. Send the number of features to sample
	 */

	/*
	 *	The workers should:
	 *		1. Accept dataset dims
	 *		2. Accept the incoming dataset
	 *		3. Split into train and validation sets
	 *		4. Accept the rest of the variables...
	 *		5. For each tree:
	 *			- subample the train set
	 *			- create tree
	 */

	////////////////////////
	// BROADCAST THE DATA //
	////////////////////////

	// Measure time AFTER loading the dataset:
	clock_t begin = clock();

	MPI_Bcast(
		&numClasses,		// buffer
		1,					// length
		MPI_INT,			// data type
		0,					// root/sender 
		MPI_COMM_WORLD		// comm port
	);	// Send number of classes

	MPI_Bcast(&numRows, 1, MPI_INT, 0, MPI_COMM_WORLD); // Num rows
	MPI_Bcast(&numCols, 1, MPI_INT, 0, MPI_COMM_WORLD); // Num cols

	if (myRank != 0){
		
		dataset = (double *) calloc(numRows * numCols, sizeof(double));
		labels = (int *) calloc(numRows, sizeof(int));
	}


	MPI_Bcast(
		dataset, 
		(numRows * numCols),
		MPI_DOUBLE,
		0,
		MPI_COMM_WORLD); // Send the dataset
	MPI_Bcast(
		labels,
		numRows,
		MPI_INT,
		0,
		MPI_COMM_WORLD); // Send labels
	MPI_Bcast(&numFeaturesToSample, 1, MPI_INT, 0, MPI_COMM_WORLD); // Num features
	MPI_Bcast(&numTreesPerWorker, 1, MPI_INT, 0, MPI_COMM_WORLD); // Num trees per worker
	MPI_Bcast(&maxTreeDepth, 1, MPI_INT, 0, MPI_COMM_WORLD); // Max depth

	///////////////////
	// SPLIT DATASET //
	///////////////////

	double *trainSet;
	int *trainLabels;
	double *validationSet;
	int *validationLabels;

	int numTrain = split_dataset(
		numRows,
		numCols,
		dataset,
		labels,
		0.8,
		&trainSet,
		&trainLabels,
		&validationSet,
		&validationLabels
	);


	int numValidation = numRows - numTrain;

	///////////////////////
	// SUBSAMPLE DATASET //
	///////////////////////

	srand(time(NULL));

	double *datasetSample;
	int *sampleLabels;

	int sampleSize = get_dataset_sample(
		numTrain, 
		numCols, 
		trainSet, 
		trainLabels, 
		1, 
		&datasetSample, 
		&sampleLabels
	);



	//////////////////
	// CREATE TREES //
	//////////////////


	int treesToTrain;
	if (myRank == 0){
		treesToTrain = numTreesForMaster;
	}
	else {
		treesToTrain = numTreesPerWorker;
	}

	// The master will have at most this many trees,
	// if not less
	Node *trees[treesToTrain];

	int treeInd;
	for (treeInd = 0; treeInd < treesToTrain; treeInd++){

		trees[treeInd] = create_tree(
			sampleSize,
			numCols,
			datasetSample,
			sampleLabels,
			numClasses, // Number of classes -  HARD CODED
			maxTreeDepth, // Max depth
			numFeaturesToSample
		);
		// printf("\n\n>>>Rank %d trained tree #%d\n\n", myRank, treeInd);
		// print_tree(trees[treeInd], 0);
		// fflush(stdout);
	}

	/////////////////////
	// GET PREDICTIONS //
	/////////////////////

	int *localPredictionMatrix = (int *) malloc(treesToTrain * numValidation * sizeof(int));

	int datapointIndex;
	for (treeInd = 0; treeInd < treesToTrain; treeInd++){
		int *preds = predict(
			numValidation,
			numCols,
			validationSet,
			trees[treeInd]);

		for (datapointIndex = 0; datapointIndex < numValidation; datapointIndex++){
			localPredictionMatrix[treeInd * numValidation + datapointIndex] = 
				preds[datapointIndex];
		}
	}


	// /////////////////////
	// // ACCURACY REPORT //
	// /////////////////////

	if (myRank == 0){
		// Create prediction matrix (numTrees * numValidation)

		int *predictionMatrix = (int *) malloc(numTrees * numValidation * sizeof(int));

		// Add my predictions to the matrix

		for (treeInd = 0; treeInd < treesToTrain; treeInd++){
			for (datapointIndex = 0; datapointIndex < numValidation; datapointIndex++){
				// Possible error?
				predictionMatrix[treeInd * numValidation + datapointIndex] =
					localPredictionMatrix[treeInd * numValidation + datapointIndex];
			}
		}

		// Accept predictions of workers

		int *incomingPredictions = (int *) malloc(numTreesPerWorker * numValidation * sizeof(int));

		int sizeOfIncomingPrediction = numTreesPerWorker * numValidation;

		// The offset from the begining
		int offset = numTreesForMaster * numValidation;


		int workerId;
		for (workerId = 1; workerId < numWorkers; workerId++){
			MPI_Recv(
				incomingPredictions,		// buffer
				sizeOfIncomingPrediction,	// size of buffer
				MPI_INT,					// data type
				workerId,					// the source
				0,							// TAG
				MPI_COMM_WORLD,				// Communication
				MPI_STATUS_IGNORE);			// Status - IGNORE

			// Add predictions to main matrix


			for (treeInd = 0; treeInd < numTreesPerWorker; treeInd++){
				for (datapointIndex = 0; datapointIndex < numValidation; datapointIndex++){
					

					int currWorkerOffset = 
							offset + 
							(workerId - 1) * numTreesPerWorker * numValidation + 
							treeInd * numValidation + datapointIndex;

					predictionMatrix[currWorkerOffset] =
						incomingPredictions[treeInd * numValidation + datapointIndex];

				}
			}

		}

		// Get the ending time AFTER receiving all the data
		clock_t end = clock();

		double time_spent = (double) (end - begin) / CLOCKS_PER_SEC;

		// Calculate accuracy

		int *finalVotePredictions = get_majority_vote(
			numTrees,
			numValidation,
			predictionMatrix,
			numClasses);


		float accuracy = get_accuracy(numValidation, validationLabels, finalVotePredictions);

		printf("\n\nAccuracy: %.2f\nElapsed time: %f\n\n", accuracy * 100, time_spent);
		fflush(stdout);

		log_results(accuracy, time_spent, numTrees, numWorkers);

		free(finalVotePredictions);
		free(incomingPredictions);
		free(predictionMatrix);

	}
	else {
		// Send predictions to the master

		MPI_Send(
			localPredictionMatrix,
			treesToTrain * numValidation,
			MPI_INT,
			0,								// Destination - master
			0,
			MPI_COMM_WORLD);

	}
	

	free(localPredictionMatrix);

	free(dataset);
	free(labels);

	free(datasetSample);
	free(sampleLabels);
	
	free(trainSet);
	free(trainLabels);

	free(validationSet);
	free(validationLabels);

	MPI_Finalize();

	return 0;
}