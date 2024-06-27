// Simple network with 1 input layer, 1 hidden layer, and 1 output node

struct Neuron
{
	double *weights;
	unsigned weights_len;
};

struct Layer
{
	struct Neuron *neurons;
	unsigned size;
	double (*activation)(double);
};

struct SimpleSequential
{
	struct Layer *layers;
	unsigned num_layers;
};

double sigmoid(double x);

Layer *layer_new(unsigned size, double (*activation)(double));
