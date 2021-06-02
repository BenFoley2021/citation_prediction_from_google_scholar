Scripts for reading the output data from the scrappers, cleaning, and selecting a subset of the data.
Subsets focusing on one topic (like "solar cells") are used as the total data is too large to be used
given the computing power available to me.

the actual data isn't uploaded to github.

get_data_from_author_crawl.py retreives the data for each paper from the pickle files output by author_list.py

select_sub_set.py performs a similar operation, but only gets papers with certain keywords.

one_hot_encode_transform_fit.py encodes the metadata and fits a model, currently using xgboost

residual analysis.py contains some ad-hoc functions for examining the quality of the fit

