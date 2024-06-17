## Virality Ranking

Here I have computed the weighted sum of different traditional ML classifiers to generate the virality score for the video. This implementation will be edited later when working with a dataset for the trending video game content.

Alternatively, we can use the Video Vision Transformer to predict the virality score.
The base code for this implementation is availabe in this [link](https://github.com/harbarex/tiktok-virality-prediction/tree/main)



### Multimodal virality

The Multimodal Virality Model is designed to predict the virality of social media content by integrating both metadata and video features. This model leverages metadata such as descriptions, likes, and shares, alongside deep video features extracted using a pre-trained ResNet50 model. By combining these data modalities, the model aims to provide a robust analysis and prediction of content virality.


In further editions of the model, I will be integrating audio features as well, to allow for a more dynamic and accurate prediction. This was generated as, in the final virality computation we won't have access to metadata features like likes, shares etc and would have to compute the virality score on the basis of the visual and audio features alone (Similar to the OpusClip computation). 


### Files

**tiktok_scraper.py**: You can scrape for tiktok videos and save the metadata to a csv file. Currently the automatic download feature is not working so had to separately download the videos.

**video_processing.py**: Extracts the video features to serve as an input for the multimodal network.

**multimodal_virality.py**: Uses the combined visual and metadata based input to identify the chances of a video going viral. The model is trained on a small dataset to compute this virality score.

### Dependencies

You need to install TikTokApi
```sh
pip install TikTokApi
python -m playwright install
```


### References

Largely inspired from the logic presented in the paper, [Slapping Cats, Bopping Heads, and Oreo Shakes: Understanding Indicators of Virality in TikTok Short Videos](https://arxiv.org/pdf/2111.02452).