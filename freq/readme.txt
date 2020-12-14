In order to run the frequency models, do 

main.py -m M -d D -s S -de

Where M is the model number, D is the dataset, and S is the summary length. 
-de is the Debug flag, which is necessary at the moment because running the entire dataset is so large
that I get Out of Memory errors and my computer crashes. So remove that flag at your own risk. 

Choices for model are 1 for Sumbasic and 2 for SumBasicExtended
Choices for dataset are reddit_tifu/long, multi_news, cnn_dailymail, and billsum
Summary length are any integer, however if you make the summary length longer than the length of the document this may cause errors.

In order to run CIDR and compare it to the multi_news clusters, simply run

CIDR.py