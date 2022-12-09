### CoralNet Crawler

The CoralNet Crawler is a tool that can be used to crawl the CoralNet website 
and download all images and annotations from a public source. The tool is 
written in Python and uses the requests library to download the images and 
annotations from the CoralNet website.

To use the CoralNet Crawler, you will need to provide the `SOURCE_ID` of the 
source you want to download. The `SOURCE_ID` is a unique identifier for the 
source on the CoralNet website, and can be found in the URL of the source's 
page on the website. For example, for this source's ID is `2733`:
```
https://coralnet.ucsd.edu/source/2733/
```

Once you have the SOURCE_ID, you can use the CoralNet Crawler to download all 
images and annotations from the source. The Crawler will download the images 
and annotations to your local computer, and you can use them for further 
analysis or processing.

The CoralNet Crawler is available as a Google Colab notebook, which allows you
to run the code online without having to install any additional software on your 
computer. To access the CoralNet Crawler on Google Colab, 
click [this link](https://colab.research.google.com/drive/1A-KGTOlfG7M4392-suQOiwEYmHsKf8c-?usp=sharing).
