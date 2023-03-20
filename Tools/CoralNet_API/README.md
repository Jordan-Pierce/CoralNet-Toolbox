# CoralNet_API
This repository provides a Python library for accessing the CoralNet API, which allows users to programmatically interact with CoralNet and perform tasks such as uploading and downloading data, annotating images, and managing user accounts.

The code in this repository is originally based on Scott Miller's work and is meant to serve as a walk-through for using the CoralNet API.

### Features
CoralNet recently opened up their API to the public, allowing researchers to use their own, or any other publicly available model to provide additional sparse labels to any image of interest. With the CoralNet API, users can request that the model label specific pixels in the image instead of just randomly placing points.

The CoralNet API library included in this repository provides the following features:

- Authorization and authentication with the CoralNet API
- Uploading images to CoralNet
- Annotating images with point labels or pixel-wise labels
- Downloading images and annotations from CoralNet
- Managing user accounts and API keys

#### Getting started
To get started, users can refer to the Making_Requests script included in this repository. This script provides a step-by-step guide for setting up a Dropbox account, obtaining authorization, and making requests to CoralNet using the CoralNet API.

Users can also refer to the API documentation provided by CoralNet for more information on the available API endpoints and their usage.

### Conclusion
The CoralNet API library provided in this repository makes it easier for researchers to programmatically interact with CoralNet and perform various tasks. With the ability to request specific pixel-wise labels for an image, the CoralNet API provides a powerful tool for studying coral reefs and marine ecosystems.

![](Figures/Workflow.png)
