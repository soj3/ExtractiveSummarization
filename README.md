# Project Overview

This project is an exploration of different extractive summarization techniques in NLP. In simple terms, extractive summarization is the task of selecting sentences from an input document that best represent the document's information in order to create a summary. More in-depth information on the function of extractive summarization tasks can be found in the Writeup file. For this project, I implemented the SumBasic algorithm, along with an extension of SumBasic which I called SumBasicExtended. I also implemented a simple clustering algorithm called CIDR. Explanations of the workings of these algorithms and the reasoning behind them can also be found in the Writeup. 

In order to run this project, first run the install.py script in order to download the necessary datasets. I would recommend including the -t flag. This will create datasets of 1000 examples, rather than using the dataset as a whole. You can omit this flag, but then the datasets get so large that Out-of-memory errors can occur when running the algorithm. Run the file on the file dataset at your own risk. 

Once the dataset has been installed, you can run either of the SumBasic models as follows:

```
main.py -d dataset -m M -s S -de
```

Where dataset is the dataset you wish to run the algorithm on, M is the model (1 for SumBasic, 2 for SumBasicExtended), and S is the summary length. in terms of number of sentences. -de is the debug flag to make sure that only 1000 examples are used. Again, run without this flag at your own risk. This will output the ROUGE scores of the summaries generated by the models as compared to summaries generated by humans. A discussion of the results of my algorithms can be found in Chapter 4 of the writeup. 

In order to run the CIDR model, simply run

```
CIDR.py
```

This will output the confusion matrix of CIDR as compared to the clustering IDs used in the multi-news dataset, as well as a metric known as purity. Again, a further explanation of the metrics used and what they represent can be found in Chapter 4 of the writeup. 

Credit to github.com/dblincoe and github.com/kclejuene for their contributions to the writeup as well as writing the dataset utilities. 