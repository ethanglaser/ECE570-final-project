# ECE570-final-project

## Running the Experiments:
Install the necessary requirements with *pip install -r requirements.txt*

Download the .mat file with the Amazon dataset [here](https://github.com/v-mipeng/TargetSpecificDomainAdaptation)

Train the base model by running the co_training file with two command line arguments based on which product types are desired for the source and target domains (0 (books), 1 (DVDs), 2 (electronics), 3 (kitchen) options available) - note that any combination will produce a model, so it doesn't really matter which are chosen. Example:

*python co_training.py 0 1*

The model can also be trained using a GPU in Google Colaboratory by downloading the dataset to Google Drive and running *co_training_reimplementation.ipynb*. This option is preferred due to the long run-time for creation of the base model (~12 hours if just using CPU).

The output of the co_training scripts is the base model that will be used during the transfer learning process of the project.

On the CPU, *plot_experiments.py* can be run to create the transfer models for the domains specified in the loops in main (this can be modified to create and test models for any combination of the 4 domains). Additionally the parameters in the CoTrain class can be tweaked proportionally to determine which breakdown of train/test/validate/tuning sizes are optimal for the model creations.

More experimentation can be done by implementing *plot_experiments.ipynb* which does the full SourceOnly and TransferClassifier object creation. The current experiment is set up to test various tune sizes to determine whether a size different than 50 labelled data from the target domain is optimal.

Note: *plot_experiments.py* and *plot_experiments.ipynb* were used for experimenting different things, they aren't just the same code in different formats

Running either of the plot_experiment scripts will create a transferable model across the specified domains, outputting plots and printing loss and accuracy values to gauge whether or not the models are successful and provide visual evidence.


## Description of the files

Note: any copied files are from the repo [here](https://github.com/v-mipeng/TargetSpecificDomainAdaptation), which is from the paper [here](https://www.aclweb.org/anthology/P18-1233.pdf)

*co_training.py*: copied from the repo and modified for TensorFlow v2 compatibility

*co_training_reimplementation.ipynb*: modified co_training and reimplemented in Google Colab for GPU use, drive compatibility, file location modifications, imports, etc.

*utils_amazon.py*: copied from the repo and modified for TensorFlow v2 compatibility

*plot_experiments.py*: modified code from the repo to run experiments to determine the optimal corpus size and test/train/valid/tune combinations were optimal for model performance and efficiency

*plot_experiment.ipynb*: modified code from the repo and reimplemented in Google Colab to determine whether a smaller labelled subset of the target domain could produce comparable results or whether a larger subset could produce significantly better accuracy

*plot_tf2.py*: Do not use. Attempted reimplementation of plot file with sessions, graphs, and other TF v1 features completely removed. Included to show progress, but it is not a runnable file.

## Dataset Descriptions

The primary dataset that was used was the Amazon sentiment analysis dataset. This dataset includes approximately 30,000 labelled sentiment polarity data points, across four different domains (books, DVDs, electronics, and kitchen products). This dataset was ideal for this project, since the focus is on creating a model from a source domain and transferring it to a target domain. The dataset is available in the format necessary for the models purposes in the original repo (.mat file). Further implementation and future work would have compatibility with more datasets with lots of labelled data (IMDB, Yelp) as well as the true intended domains of the model - those with minimal labelled data.

