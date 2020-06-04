# Reference Data

## Why?
A major part of the NEMF framework is its capability to inverse model or fit a
designed model output to data.
These data sets might for example be measurements take in the field.
They could also be some sort of expected behavior by the modeler.
I.e. something like:
"if my assumptions are correct,
 I expect this to converge to twice the size of the other compartment"
However they were created, the user needs to provide them somehow to the 
framework.
The framework expects the data in a standardized form.
This ensures that the data is interpreted correctly while avoiding tedious
data import configurations by the user,

## Format
The format required is sometimes referred to as the "tidy-data" format.
We expect rows to be 'observations' of the system.
The columns represent the compartments in the model.

## Example
Assume that A \& B are names of two compartments in the model.
Further, assume that t_i is the time at which the i'th observation took place.
A(t_i) represents the value of the compartment A at the specific moment in time.

A data set containing several observations may then look
like the following table:

| time 	| A		| B 	|
|-------|-------|-------|
| t_1 	| A(t_1)| B(t_1)|
| t_2 	| A(t_2)| B(t_2)|
| t_3 	| A(t_3)| B(t_3)|

An other potential outcome is that the desired state of the model is a 
steady-state. Meaning, that the described compartment is expected to converge
to a certain value.
In that case the we expect that the compartment will reach the steady-state 
value at t=infinity.
Such situation is expected to be described by

| time 	| A		| B 	|
|-------|-------|-------|
| inf 	| A(inf)| B(inf)|

Note, that the framework does not specifically enforce a convergence as this
behaviors is governed by the model. 
Enforcing to only consider steady-states in the optimization process is 
currently no possible in the framework.
This means that the model might reach the desired values A(inf) & B(inf) at the 
end of the time evolution while the model has not (yet) converged.
To avoid such an output it is possible to specify the the steady-state
solution repeatably as in the first table.
However, the values for A and B remain the same for all time steps.
This suppress non-steady-states solutions in the model optimization.

## File format
Currently we only implemented parsers for plain text files, like *csv* or *tsv* 
files. Parsers for i.e. excel files are in development.

An example of such a file can be found in the [GitHub Repository](https://github.com/465b/nemf/blob/master/example_files/NPZD_oscillation_on_1990.csv)