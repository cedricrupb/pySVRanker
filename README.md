# Algorithm Selection for Software Validation
The field of Software Validation offers a plethora of techniques to verify the correctness of software. Choosing the right tool for this purpose often requires
expert knowledge. In other words, an given approach might handle recursive
programs but fails in presence of loops.

In this project we tackle the selection of a fitting algorithm
by machine learning. More precisely, we predict a ranking over possible
algorithms using graph kernels.

## Requirements
The general requirements to run experiments for Algorithm Selection:
+ Java 8
+ Python 3.6
+ [PeSCo](https://github.com/cedricrupb/cpachecker)

In addition, some packages have to be installed via pip:
+ NumPy 1.15.4
+ tqdm 4.18.0
+ SciPy 0.19.1
+ murmurhash3 2.3.5
+ Scikit-Learn 0.19.1
+ lxml 4.0.0
+ [pyTasks](https://github.com/cedricrupb/pyTasks)


## Build and Execute

### Build
To build the project, you have to install the requirements above.
Till now, pyTasks is not available on public repositories. Hence,
you need to clone the project from github:
```cmdline
git clone <REPOSITORY_URL>
PYTHONPATH="<PATH_TO_REPRO>:$PYTHONPATH"
export PYTHONPATH
```
Now you should be to use pyTasks during execution.

### Execute
To perform experiments on the software verification competition [SV-Comp](https://sv-comp.sosy-lab.org) benchmark
set, the following steps are necessary:
1. Generate Software Validation Graphs from C Code using PeSCo
2. Collect subtrees from each graph
3. Label instances using results of SV-Comp
4. Start experiments on the benchmark

#### Software Verification Graphs
For generating graphs, please refer to [PeSCo](https://github.com/cedricrupb/cpachecker). A list of programs used in our experiments can be found in doc folder.

#### Subtrees
To collect subtrees, we require a folder src/ containing all graphs in .dfs format
(.dfs is the output format of PeSCo). The process can be started by using generate_bag.py in scripts/ folder.
```cmdline
scripts/generate_bag.py [-b] src/ <iteration_bound> <depth_bound> subtrees/
```
The script requires a single file as input. If you want to specify a whole folder
for processing, you can use -b as parameter. The iteration bound specifies how deep
our subtrees will be. The depth bound defines the depth of AST trees in our
verification graph. For more detailled information, please refer to our paper.
In general, an AST depth of 5 and an iteration bound of 2 is good choice.

Till this point, we created many files individually for each C program.
Now it is time to summarize our results.
```cmdline
scripts/summarize_bags.py subtrees/ bags/
```
You might recognize subtrees/ from the command before. This is the folder
containg all gathered subtree representation. In bags/ we collect all
representation in files (sorted after iteration bound and AST depth).
This step is important for our labelling procedure.

#### Labelling
Since we want to learn on our graph representation, we firstly have to label our
instances.
To start with, you firstly have to download raw data obtained in SV-Comp.
For SV-Comp 2018, you can find the data [here](https://sv-comp.sosy-lab.org/2018/results/results-verified/All-Raw.zip).
After extracting the Zip File into a directory, you can run:
```cmdline
scripts/index_svcomp.py [-p] [-c] <ZIP_DIR> <OUTPUT_FILE>
```
The parameter -p allows you to define a prefix that is removed from a path name.
As an example, "../sv-benchmarks/c/" could be a valid prefix. If the SV-Comp results
are encoded as CSV files, the -c parameter will change the parsing behaviour.

After this step, you created a file that indexes C file names to matching labels.
Now it is necessary to bind the labels to graph instances.
```cmdline
scripts/map_label.py <INDEX_FILE> <BAG_DIR> <OUT_DIR>
```
In this script, we can merge our index with the subtree files generated in the
previous steps. The output directory will contain our training data ready for our
experiments.

#### Running experiments
The task framework used for experiments requires a working directory.
Therefore we have to create a directory (e.g. work_dir/) and move our
training inside (e.g. work_dir/training/).
To run our experiment, we can use:
```cmdline
scripts/run_experiments.py -c <CONFIG_FILE> -p <PREFIX> -o <EXECUTION_GRAPH>
```
Howerver, we need to specify experiments parameter before we can start
our experiments. Firstly, we have to create a config file. An example can be found
in the scripts/ folder. It is important to set the pattern config under
"BagLoadingTask". Other config options can remain. More importantly, you can modify
some running options (e.g. AST depth) in run_experiments.py itself.
PREFIX defines the working directory. In our example, this is work_dir.
The execution graph is there to safe partial process. It is important to
notice that the execution graph is saved if exit the code. Therefore, you have
to delete the old graph or define another file path if you start new experiments.
