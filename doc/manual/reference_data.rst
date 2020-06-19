Reference Data
==============

Why?
----

A major part of the NEMF framework is its capability to inverse model or
fit a designed model output to data. These data sets might for example
be measurements take in the field. They could also be some sort of
expected behavior by the modeler. I.e. something like: “if my
assumptions are correct, I expect this to converge to twice the size of
the other compartment” However they were created, the user needs to
provide them somehow to the framework. The framework expects the data in
a standardized form. This ensures that the data is interpreted correctly
while avoiding tedious data import configurations by the user,

Format
------

The format required is sometimes referred to as the “tidy-data” format.
We expect rows to be ‘observations’ of the system. The columns represent
the compartments in the model.

Example
-------

Assume that A & B are names of two compartments in the model. Further,
assume that t_i is the time at which the i’th observation took place.
A(t_i) represents the value of the compartment A at the specific moment
in time.

A data set containing several observations may then look like the
following table:

==== ====== ======
time A      B
==== ====== ======
t_1  A(t_1) B(t_1)
t_2  A(t_2) B(t_2)
t_3  A(t_3) B(t_3)
==== ====== ======

An other potential outcome is that the desired state of the model is a
steady-state. Meaning, that the described compartment is expected to
converge to a certain value. In that case the we expect that the
compartment will reach the steady-state value at t=infinity. Such
situation is expected to be described by

==== ====== ======
time A      B
==== ====== ======
inf  A(inf) B(inf)
==== ====== ======

The framework does not specifically enforce a convergence as
this behaviors is governed by the model. Enforcing to only consider
steady-states in the optimization process is currently no possible in
the framework. This means that the model might reach the desired values
A(inf) & B(inf) at the end of the time evolution while the model has not
(yet) converged. To avoid such an output it is possible to specify the
the steady-state solution repeatably as in the first table. However, the
values for A and B remain the same for all time steps. This suppress
non-steady-states solutions in the model optimization.

.. note:: It is not possible to mix the 'inf' timestamp with date-time values 
          in the reference data sets. If both is present the optimizer will 
          default to the non-inf type fitting if the 'inf' timestamp is not 
          set in the very row of the data set.


Time format
-----------

[placeholder]

* posix

  * integer counting seconds since 1.1.1970 UTC

  * also negative possible, seconds before 1970

* Datetime

  * which datetime formats

  * how in excel/csv


File format
-----------

Their are two data types that are supported:

	* **plain text files**, (typical file extensions are '.txt' or '.csv')
	* **excel files** (.xlsx)

The plain text files are required to be in a specific standard to be correctly 
interpreted by the framework.
Besides the general structure introduced in the previous section we require the 
headers to be named exactly the same as the compartments in the model.
This is necessary to match them without any potential misinterpretation.
Any columns which do not have a exact counterpart in the model will be ignored.

Plain text files
~~~~~~~~~~~~~~~~

Every data sets needs to contain a header defining the names of the columns 
which are compared to the model compartments.
The names of the compartment are required to be separated by the chosen 
delimiter (default ```','```)

An example of such a file can be found in the `GitHub
Repository <https://github.com/465b/nemf/blob/master/example_files/NPZD_oscillation_on_1990.csv>`__

The headers has some flexibility how it is defined.
The line may start with a comment symbol (default: "#')

.. code-block:: csv

   # Datetime, A, B, [...]

Leading and trailing white spaces in a name are ignored.
Hence,

.. code-block:: csv

   Datetime,A,B = Datetime, A, B


It is also possible to write the name inside of quotation marks. This is 
typically used to mark character strings.
Hence,

.. code-block:: csv

   "Datetime","A","B" = 'Datetime','A','B' = Datetime, A, B

.. note:: If quotation marks are used, the names are parsed literally.
   As a result of that, leading and trailing whitespaces are no longer ignored.


Generally, it is assumed that the names of the column are defined in the in the 
very first row of the document.

.. code-block:: csv

   1 Datetime, A, B
   2 t0,A(t0),B(t0)

If this is not the case, i.e. the file is formatted something like this,

.. code-block:: csv

   Some additional information about the file and its origin
   before the data column headers are parsed
   [Data]

the framework might not know how to interpret it.
There are two option how to deal with this.
We recommend using the following format to avoid this problem.

* Use the comment mark (#) for the non-header lines of the data while *NO* 
  comment mark is used for the column headers. I.e:
  
  This will work,
  
  .. code-block:: python
  
     # Some additional information about the file and its origin
     # before the data column headers are parsed
     Datetime, A, B
     [...]
  
  this will *NOT* work
  
  .. code-block:: python
  
     # Some additional information about the file and its origin
     # before the data column headers are parsed
     # Datetime, A, B
     [...]
  
  while THIS will also work.
  
  .. code-block:: python
     
     # Datetime, A, B
     [...]
  
* Alternatively, the *load_ref_data()* method has the option to 
  *"skip_header= "*.
  However, this requires a manual re-import of the data and is not recommended.
  
  .. code-block:: python
  
     model = nemf.load_model('path/to/model.yml')
     model.load_ref_data('path/to/reference/data.csv',skip_header=5)


Excel files
~~~~~~~~~~~


