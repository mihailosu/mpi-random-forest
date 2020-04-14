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
// int read_csv(char *fname, double **dataset, long int *numFeatures, long int *numLabels, LabelMap **labelMap, int **intLabelsIn);
int read_csv(char *fname, double **dataset, int *numFeatures, int *numLabels, LabelMap **labelMap, int **intLabelsIn);
int get_dataset_sample(int numRows, int numColumns, double *dataset, int *labels, float ratio, double **datasetSample, int **sampleLabels);
void shuffle_dataset(int numInstances, long int numCols, double **dataset, int **labels);
float gini_index(int nLeft, int *leftLabels, int nRight, int *rightLabels, int numClasses);
float gini_index_2(int nLeft, int *leftLabels, int nRight, int *rightLabels, int numClasses);
int split_dataset(int numRows, int numCols, double *dataset, int *labels, float ratio, double **train_data, int **train_labels, double **validation_data, int **validation_labels);
float validate_tree(int numRows, int numColumns, double *validationSet, int *validationLabels, Node *root);
int algorithm_example(int argc, char *argv[]);
int check_input(int argc, char *argv[]);
int * predict(int numRows, int numCols, double *dataset, Node *root);
int * get_majority_vote(int numTrees, int numInstances, int *matrix, int numClasses);
float get_accuracy(int n, int *labels, int *predictions);