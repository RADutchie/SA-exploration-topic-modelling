South Australian exploration envelope topic analysis 
==============================

It is a requirement of legislation that exploration companies operating in South Australia submit reports and data on their exploration activities. The South Australian Government has ~ 8000 digital exploration 'envelopes' or reports on mineral and petroleum exploration and activities dating back to the 1950's.

These data are all freely available via the [SARIG](https://map.sarig.sa.gov.au/) web portal. In addition to the raw reports, the Geological Survey of South Australia (GSSA) has indexed these reports and provide a csv dataset which includes the envelope number, the tenement number associated with the report, a broad subject and a short summary or abstract of the report (see the raw data file). 

There are 5894 abstracts provided in this data set. In these notebooks I present a way to utilise NLP (natural language processing) techniques to clean up the datasets and apply Latent Dirichlet Allocation (LDA) topic modelling to identify the main 'topics' discussed in the exploration report summaries. Once the main topics were identified I utilised the associated tenement numbers and their spatial boundaries (available as geodatabase or shape files from SARIG) to display a spatial distribution of the topics across South Australia. 

You can find links to the two blog posts that work through and discuss the results of these two notebooks below:
* [What are explorers looking for in S.A.? Part 1](https://geodataanalytics.net/what-are-explorers-looking-for/)
* [What are explorers looking for in S.A.? Part 2](https://geodataanalytics.net/what-are-explorers-looking-for-in-south-australia/)

The results suggest the states exploration record can be defined by 8 major topics: 
* IOCG exploration
* Gold, Copper and base metal exploration
* Uranium and coal exploration
* Heavy minerals, extractives and environmental
* Mines operations and development
* Diamond exploration
* Oil and Gas exploration, and
* Geothermal exploration

The distribution of these topics across the state demonstrate which regions are prospective for different types of commodities or where exploration has been concentrated because of other factors like infrastructure or 'safe' brown-fields exploration targets. 

While these results may not be all that surprising. This demonstrates some of the potential information stored in unstructured text based company data and some of the potential ways to begin to unlock that knowledge.

![Spatial kdr plot of exploration topics across South Australia](Notebooks/Abstract%20LDA/Figures/kde_topic_dist_plot_3a.png)
![Spatial kdr plot of exploration topics across South Australia cont](Notebooks/Abstract%20LDA/Figures/kde_topic_dist_plot_3b.png)


To run the notebooks locally
------------
Requires 2 environments, one for the NLP topic modelling and one for the geospatial data analysis
* Setup a virtualenv using conda running Python 3.7
* For the *topic_modelling* notebooks
* `conda env create --file NLP-env.yml` 
* For the *topic_analysis* notebook
* `conda env create --file spatial-env.yml`
* Clone the notebooks folder

Project Organization
------------

    ├── README.md
    ├── requirements_nlp.txt
    ├── requirements_spatial.txt
    ├── notebooks
    │   ├── Abstract LDA
    │   │   ├── Figures
    │   │   ├── Model
    │   │   ├── Abstract_LDA_topic_analysis.ipynb
    │   │   └── Abstract_LDA_topic_modelling.ipynb
    │   ├── helper                      <--- helper functions
    │   ├── create_env_dataset.ipynb    <--- creating the dataset  
    │   └── text_preprocessing.ipynb    <--- text preprocessing experiments
    └── data              
        ├── Processed                   <--- processed modeled topics
        └── Raw                         <--- input abstracts and metadata
           
    
--------
