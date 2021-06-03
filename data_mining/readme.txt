Descrption of overall data mining scheme:

Papers are found by giving the paperGraph script a starting point, it then crawls through the citations 
of that paper and all papers citing it in a breadth first manner. The script reports the title of the papers
and the authors (identified by google scholar id) to the sql table. Other data are saved as pickle files.

The authorList script reads the authors from the sql table, goes to the google scholar page for that author,
and records the data for each paper on that authors page. The author pages have much more detailed meta data,
and so this dataset is what's used for ML.

Both scripts are necessary to explore the avialable data and find papers on a topic, and get the detailed
information needed to predict inmportance. Multiple instances of each script can be run in parallel, and 
syncing to the sql table prevents these scripts rescraping the same data.

