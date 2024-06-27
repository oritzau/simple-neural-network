#include <math.h>
#include "simple_sequential.h"

// Simple network with 1 input layer, 1 hidden layer, and 1 output node

double sigmoid(double x)
{
	return 1 / ( 1 + exp((-x)) );
}
