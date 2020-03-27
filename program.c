#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h> 

typedef struct Node {
	int featureIndex;
	double value;

	int isLeaf;
	int label;

	Node *left;
	Node *right;
} Node;

typedef struct LabelMap {
	char name[10];
	int value;
} LabelMap;

void print_dataset_row(int numRows, int numCols, double *dataset, int index){
	if (numRows < index){
		printf("Cannot print row %d when there are %d rows...\n", index, numRows);
	}

	printf("\nRow %d\n", index);
	int i;
	for (i = 0; i < numCols; i++){
		printf("%lf ", dataset[index * numCols + i]);
	}
	printf("\n\n");
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
void get_split(double *dataset, int numFeatures, int numInstances, int numSampleFeatures){
	
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
 */
float gini_index(int nLeft, int *leftLabels, int nRight, int *rightLabels, int numLabels){

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
		for (labelMarker = 0; labelMarker < numLabels; labelMarker++){
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
		for (labelMarker = 0; labelMarker < numLabels; labelMarker++){
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
		// Update the gini index for the right side of the split
		gini += ((1.0 - score) * (nRight / totalNumInstances));
	}

	return gini;

}

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

	// printf("Labels after shuffle\n");
	// int hg;
	// for (hg = 0; hg < numInstances; hg++){
	// 	printf("%d\n", labels[hg]);
	// }

	int k;
	for (k = 0; k < 3; k++){
		print_dataset_row(numInstances, numFeatures, dataset, k);
		printf("%d\n", labels[k]);
		labels[k];
	}

	double *datasetSample;
	int *sampleLabels;

	int sampleSize = get_dataset_sample(
		numInstances, 
		numFeatures, 
		dataset, 
		labels, 
		0.1, 
		&datasetSample, 
		&sampleLabels
	);

	int i;
	for (i = 0; i < 3; i++){
		print_dataset_row(numInstances, numFeatures, datasetSample, i);
		printf("%d\n", sampleLabels[i]);
	}

	// Free memory

	free(dataset);
	return 0;
}