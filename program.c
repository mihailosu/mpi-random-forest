#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h> 

typedef struct Node {
	int colIndex;
	double value;

	int isLeaf;
	int label;

	struct Node *left;
	struct Node *right;
} Node;

typedef struct LabelMap {
	char name[10];
	int value;
} LabelMap;


void print_tree(Node *node, int depth);
int majority_class(int numRows, int *classLabels, int numClasses);
void print_dataset_row(int numRows, int numCols, double *dataset, int index);
Node * get_split(int numRows, int numCols, double *dataset, int *labels, int numClasses, int numSelectedFeatureColumns, int **selectedFeatureColumns, int currentDepth);
Node * create_tree(int numRows, int numCols, double *dataset, int *labels, int numClasses, int maxDepth, int numFeatures);
int read_csv(char *fname, double **dataset, long int *numFeatures, long int *numLabels, LabelMap **labelMap, int **intLabelsIn);
int get_dataset_sample(int numRows, int numColumns, double *dataset, int *labels, float ratio, double **datasetSample, int **sampleLabels);
void shuffle_dataset(int numInstances, long int numCols, double **dataset, int **labels);
float gini_index(int nLeft, int *leftLabels, int nRight, int *rightLabels, int numClasses);
float gini_index_2(int nLeft, int *leftLabels, int nRight, int *rightLabels, int numClasses);
int split_dataset(int numRows, int numCols, double *dataset, int *labels, float ratio, double **train_data, int **train_labels, double **validation_data, int **validation_labels);


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
		int **selectedFeatureColumns,
		int currentDepth){

	printf("Entered depth %d\n", currentDepth);
	
	int *featureColPtr = *selectedFeatureColumns;
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

		printf("\tSelecting feature index %d\n", featureIndex);

		// If the feature is unavailable
		if (featureColPtr[featureIndex] == -1){
			continue;
		}

		feature = featureColPtr[featureIndex];

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
					if (labels[innerRow] > 1){
						printf("Label going left: %d\n", leftLabels[lCount - 1]);
						
					}
				}
				else {
					rightLabels[rCount++] = labels[innerRow];
					if (labels[innerRow] > 1){
						printf("Label going right: %d\n", rightLabels[rCount - 1]);
						
					}
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

	printf("Best feature index at depth %d is: %d\n", currentDepth, bestFeatureIndex);

	// After we've found the best feature based on the Gini Impurity 
	// we have to remove that feature from the feature pool
	featureColPtr[bestFeatureIndex] = -1;

	// If the number of elements on either side of the split
	// is 0, then this node becomes a leaf node

	if (nLeft == 0 || nRight == 0){
		// printf("First if\n");
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
		// printf("Second if\n");
		Node *lastParent = (Node *) malloc(sizeof(Node));

		lastParent->colIndex = bestFeatureIndex;
		lastParent->value = bestSplitValue;
		lastParent->isLeaf = 0;

		Node *leafLeft = (Node *) malloc(sizeof(Node));
		Node *leafRight = (Node *) malloc(sizeof(Node));


		int majorityLeft = majority_class(bestNLeft, bestLeftLabels, numClasses);
		int majorityRIght = majority_class(bestNRight, bestRightLabels, numClasses);

		// These 2 leaf nodes could be merged into one... TODO

		leafLeft->isLeaf = 1;
		leafLeft->label = majorityLeft;

		leafRight->isLeaf = 1;
		leafRight->label = majorityRIght;

		lastParent->left = leafLeft;
		lastParent->right = leafRight;

		return lastParent;
	}

	// printf("Else\n");
	// If we are not at max depth and we are not a leaf:

	Node *current = (Node *) malloc(sizeof(Node));

	current->colIndex = bestFeatureIndex;
	current->value = bestSplitValue;
	current->isLeaf = 0;

	Node *leftChild = get_split(
		numRows,
		numCols,
		dataset,
		labels,
		numClasses,
		numSelectedFeatureColumns,
		selectedFeatureColumns,
		currentDepth - 1);

	Node *rightChild = get_split(
		numRows,
		numCols,
		dataset,
		labels,
		numClasses,
		numSelectedFeatureColumns,
		selectedFeatureColumns,
		currentDepth - 1);

	current->left = leftChild;
	current->right = rightChild;

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

	int *selectedColumns = (int *) malloc(numFeatures * sizeof(int));
	
	// Initialize selected columns to -1
	int i;
	for (i = 0; i < numFeatures; i++){
		selectedColumns[i] = -1;
	}

	i = 0;
	int j;
	int indexToSelect;
	int alreadySelected;
	// Select random columns to take into consideration
	while (i < numFeatures){
		
		alreadySelected = 0;

		indexToSelect = rand() % numFeatures;

		for (j = 0; j <= i; j++){
			if (selectedColumns[j] == indexToSelect){
				// Column is already selected
				alreadySelected = 1;
				break;
			}
		}

		if (!alreadySelected){
			// If we've picked a new column
			selectedColumns[i] = indexToSelect;

			i++;
		}

	}

	/////////////////
	// CREATE TREE //
	/////////////////

	printf("Creating root for dataset with %d rows\n", numRows);

	Node *root = get_split(
		numRows,
		numCols,
		dataset,
		labels,
		numClasses,
		numFeatures,
		&selectedColumns,
		maxDepth);

	free(selectedColumns);

	return root;

}

/*
 * Read a csv file, return the number of rows.
 * TODO: Check if all rows have the same number of columns
 */
int read_csv(
		char *fname, 
		double **dataset, 
		long int *numFeatures, 
		long int *numLabels,
		LabelMap **labelMap,
		int **intLabelsIn){
	FILE *f;
	f = fopen(fname, "r");

	
	// If the file exists
	if (f){
		char line[2048];

		// Read num of instances
		fgets(line, sizeof(line), f);

		long int numInstances = strtol(line, NULL, 10);

		printf("Num instances %ld\n", numInstances);

		// Read num of features
		fgets(line, sizeof(line), f);

		*numFeatures = strtol(line, NULL, 10);

		printf("Num of features %ld\n", *numFeatures);

		// Read num of labels, and initialize LabelMap
		fgets(line, sizeof(line), f);

		*numLabels = strtol(line, NULL, 10);

		printf("Num of labels %ld\n", *numLabels);

		LabelMap lblMap[*numLabels];

		*labelMap = lblMap;

		int intLabels[numInstances];

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
				// printf("%lf ", currVal);
				(*dataset)[row * (*numFeatures) + columnCount] = currVal;

				token = strtok(NULL, ",\n");
				// printf("%d", columnCount);
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
					// printf("Inicijalizujemo %s\n", token);
					strcpy(lblMap[i].name, token);
					currentRowIntLabel = lblMap[i].value;
					// printf("Inicijalizujemo %s\n", token);
					break;
				}
				else if (strcmp(token, labelToken) == 0){
					// printf("Uzimamo vrednost na %d\n", i);
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
		
		// printf("%d\n", randElementIndex);

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

	printf("Number of rows %d\n", numInstances);
	printf("Number of columns %ld\n", numCols);

	int *lblPointer = *labels;


	int tempInt;
	int i;
	double temp[numCols];
	for (i = 0; i < numInstances; i++){
		
		int randElementIndex = rand() % numInstances;

		int colIndex;

		// printf("Before\n");
		// print_dataset_row(numInstances, numCols, *dataset, i);
		// print_dataset_row(numInstances, numCols, *dataset, randElementIndex);

		// temp = dataset[i]
		for (colIndex = 0; colIndex < numCols; colIndex++){
			// printf("%d\n", colIndex);
			temp[colIndex] = (*dataset)[i * numCols + colIndex];
		}

			// printf("%s\n", "Done");
		// dataset[i] = dataset[randIndex]
		for (colIndex = 0; colIndex < numCols; colIndex++){
			// printf("%s\n", "B");
			(*dataset)[i * numCols + colIndex] = (*dataset)[randElementIndex * numCols + colIndex];
		}

		// dataset[randIndex] = temp
		for (colIndex = 0; colIndex < numCols; colIndex++){
			// printf("%s\n", "C");
			(*dataset)[randElementIndex * numCols + colIndex] = temp[colIndex];
		}


		// printf("After\n");
		// print_dataset_row(numInstances, numCols, *dataset, i);
		// print_dataset_row(numInstances, numCols, *dataset, randElementIndex);

		// // Shuffle labels the same way
		tempInt = lblPointer[i];
		// tempInt = (*labels)[i];

		// printf("Switching label #%d and #%d, values: %d and %d\n", 
		// 	i, randElementIndex,
		// 	(lblPointer[i]), (lblPointer[randElementIndex]));

		lblPointer[i] = lblPointer[randElementIndex];
		// ((*labels)[i]) = ((*labels)[randElementIndex]);

		lblPointer[randElementIndex] = tempInt;
		// ((*labels)[randElementIndex]) = tempInt;

		// tempInt = (*labels)[i];

		// printf("%d) %d\n", i, (*labels)[i]);

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
float gini_index(int nLeft, int *leftLabels, int nRight, int *rightLabels, int numClasses){

	int totalNumInstances = nLeft + nRight;

	float gini = 0.0;

	int i, labelMarker;
	
	float score = 0.0;
	float p = 0;
	int count = 0;
	// Calculate score for the left group;
	if (nLeft){
		// If there are instances in the left group
		// For every class...
		for (labelMarker = 0; labelMarker < numClasses; labelMarker++){
			// ... count the number of occurences of that class
			// in the left group
			for (i = 0; i < nLeft; i++){
				if (leftLabels[i] == labelMarker){
					count++;
				}
			}
			// Calculate the probability of an instance belonging
			// to the class
			p = count / nLeft;
			score += (p * p);
		}
		printf("\t\tScore LEFT: %f\n", score);
		// Update the gini index for the left side of the split
		gini += ((1.0 - score) * (nLeft / totalNumInstances));
	}

	count = 0;
	p = 0;
	score = 0.0;

	// Calculate score for the right group

	if (nRight){
		// If there are instances in the left group
		// For every class...
		for (labelMarker = 0; labelMarker < numClasses; labelMarker++){
			// ... count the number of occurences of that class
			// in the left group
			for (i = 0; i < nRight; i++){
				if (rightLabels[i] == labelMarker){
					count++;
				}
			}
			// Calculate the probability of an instance belonging
			// to the class
			p = count / nRight;
			score += (p * p);
		}
		printf("\t\tScore RIGHT: %f\n", score);
		// Update the gini index for the right side of the split
		gini += ((1.0 - score) * (nRight / totalNumInstances));
	}

	return gini;

}

/*
 *
 *
 *
 *
 * ZANIMLJIVOST: Potrebno je naglasiti ako nije celobrojno deljenje...
 */
float gini_index_2(int nLeft, int *leftLabels, int nRight, int *rightLabels, int numClasses){

	int totalNumInstances = nLeft + nRight;

	// printf("Calculating gini for %d rows\n", totalNumInstances);

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

	// printf("\t\tTotal: %d\n", totalNumInstances);
	// printf("\t\tLeft [0]: %d\n", leftLblCount[0]);/
	// printf("\t\tLeft [1]: %d\n", leftLblCount[1]);
	// printf("\t\tRight [0]: %d\n", rightLblCount[0]);
	// printf("\t\tRight [1]: %d\n", rightLblCount[1]);

	if (nLeft){
		p = (float) leftLblCount[0] / nLeft;
		// printf("\t\tLeft p 0: %f\n", p);
		scoreLeft += (p * p);
		p = (float) leftLblCount[1] / nLeft;
		// printf("\t\tLeft p 1: %f\n", p);
		scoreLeft += (p * p);
		// printf("\t\tScoreLeft: %f\n", scoreLeft);
	}

	if (nRight){
		p = (float) rightLblCount[0] / nRight;
		// printf("\t\tRight p 0: %f\n", p);
		scoreRight += (p * p);
		p = (float) rightLblCount[1] / nRight;
		// printf("\t\tRight p 1: %f\n", p);
		scoreRight += (p * p);
		// printf("\t\tScoreRight: %f\n", scoreRight);
	}

	gini += ((1.0 - scoreLeft) * ((float) nLeft / totalNumInstances));
	gini += ((1.0 - scoreRight) * ((float) nRight / totalNumInstances));

	// printf("\t\tGini score is %f\n", gini);

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



/*
 * Steps to take when paralelizin:
 *   1. Determin how much trees each node should train
 *   	- each node will probably train > 1
 *   	- node 0 should train the leftover number of nodes
 *   2. Send the dataset to each node (include the dimensions of the matrix)
 *   3. Send 
 *
 */
int main(int argc, char *argv[]){

	long int numFeatures;
	long int numLabels;

	double *dataset;
	// Label *labels; // Label strings of length 1


	LabelMap *labelMap;

	int *labels;

	// The file is structured as follows:
	// number of datapoints
	// number of features
	// number of classes
	// data...
	long int numInstances = read_csv(
		"sonar.all-data", 
		&dataset, 
		&numFeatures,
		&numLabels,
		&labelMap,
		&labels
	);

	// Print the first row of the dataset:

	// int i;
	// for (i = 0; i < numFeatures; i++){
	// 	printf("%lf ", dataset[i]);
	// }
	// printf("\n");

	// Print the label map

	// int j;
	// for (j = 0; j < numLabels; j++){
	// 	printf("%s\n", (*(labelMap + j)).name);
	// }

	// int k;
	// for (k = 0; k < numInstances; k++){
	// 	printf("%d\n", labels[k]);
	// }


	///////////////////////
	// RUN RANDOM FOREST //
	///////////////////////

	// Seed random values with time
	srand(time(0));

	/////////////////////////
	// SHUFFLE THE DATASET //
	/////////////////////////

	printf("Shuffle the dataset...\n");
	printf("Number of features %ld\n", numFeatures);
	shuffle_dataset(
		numInstances,
		numFeatures,
		&dataset,
		&labels
	);


	////////////////////////////////////.
	// SPLIT INTO TRAIN AND VALIDATION //
	/////////////////////////////////////

	double *trainSet;
	int *trainLabels;
	double *validationSet;
	int *validationLabels;

	int numTrain = split_dataset(
		numInstances,
		numFeatures,
		dataset,
		labels,
		0.8,
		&trainSet,
		&trainLabels,
		&validationSet,
		&validationLabels
	);



	int numValidation = numInstances - numTrain;

	printf("Dataset split --- train: %d | validation: %d", numTrain, numValidation);

	int i;
	for (i = 0; i < 5; i++){
		print_dataset_row(numTrain, numFeatures, trainSet, i);
		printf("%d\n", trainLabels[i]);
	}


	///////////////////////////
	// GET DATASET SUBSAMPLE //
	///////////////////////////

	double *datasetSample;
	int *sampleLabels;

	int sampleSize = get_dataset_sample(
		numTrain, 
		numFeatures, 
		trainSet, 
		trainLabels, 
		0.8, 
		&datasetSample, 
		&sampleLabels
	);

	/////////////////
	// CREATE TREE //
	/////////////////

	int numFeaturesToSample = round(sqrt(numFeatures));

	printf("Feature pool size is %d\n", numFeaturesToSample);

	Node *root = create_tree(
		sampleSize,
		numFeatures,
		datasetSample,
		sampleLabels,
		2, // Number of classes -  HARD CODED
		3, // Max depth
		numFeaturesToSample
	);

	print_tree(root, 0);

	// Free memory

	free(dataset);
	return 0;
}