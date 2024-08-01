# Tracking Southern Alaskan Forest Health 

Deep learning algorithms to map damaged trees in forests.

## Data Creation and Locations

The data used for training these models is contained in [the following huggingface directory.](https://huggingface.co/datasets/saking3/alaska_dead_trees) 

The procedure for segmenting data was as follows: 
- Use k-means augmented with Gray Level Cooccurrence Matrix (GLCM) to cluster image into 30-60 classes.
- Convert k-means classed image into binary mask by using QGIS to reclass each class as either "dead tree" (1) or "non-dead tree" (0)

For reference, failed segmentation attempts include: 
- the segment-geospatial python package
- automatic segmentation with the SAM model.  

## Original Modeling Methodology

## Environment Setup

## References
