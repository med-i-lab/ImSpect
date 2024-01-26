### ImSpect: Image-driven Self-supervised Learning for Surgical Margin Evaluation with Mass Spectrometry

Laura Connolly[^1], Fahimeh Fooladgar[^2], Amoon Jamzad[^1], Martin Kaufmann, ...., Purang Abolmaesumi[^2], Parvin Mousavi[^1]

[^1]: Queen's University
[^2]: University of British Columbia

![Block Diagram](./images/Overview.png)

#### Purpose: 
Real-time assessment of surgical margins is critical for favourable outcomes in cancer patients. The iKnife is a mass spectrometry device that has demonstrated potential for margin detection in cancer surgery. Previous studies have shown that using deep learning on iKnife data can facilitate real-time tissue characterization. However, none of the existing literature on the iKnife facilitate the use of publicly available, state-of-the-art pre-trained networks or datasets that have been used in computer vision and other domains. 
#### Methods: 
In a new framework we call ImSpect, we convert 1D iKnife data, captured during basal cell carcinoma (BCC) surgery, into 2D images in order to capitalize on state-of-the-art image classification networks. We also use self-supervision to leverage large amounts of unlabelled, intraoperative data to accommodate the data requirements of these networks. 
#### Results: 
Through extensive ablation studies we show that we can surpass previous benchmarks of margin evaluation in BCC surgery using iKnife data, achieving an area under the receiver operating characteristic curve (AUC) of 81.6\%. We also depict the attention maps of the developed DL models to evaluate the biological relevance of the embedding space
of the models. 
#### Conclusions: 
We propose a new method for characterizing tissue at the surgical margins, using mass spectrometry data from cancer surgery.


---

<!-- 
  >> Codes will be uploaded soon ...-->

## Running the code

### 1. Image Generation
The `spec2img.py` script provides a comprehensive approach to convert 1D spectra into 2D images using various techniques like Markov Transition Fields (MTF), Gramian Angular Field Summation (GAFS), and Gramian Angular Field Difference (GAMD).

---

### 2. Self-Supervised Learning: 
Begin by utilizing the SIMCLR code available in this repository. This repository contains the necessary scripts and modules for implementing self-supervised learning model. Proceed to train your model using the self-supervised learning approach outlined in the repository. This step involves learning representations without using labeled data.

#### Saving the Trained Model: After the training process is complete, save the trained model. This will preserve the learned weights, which are crucial for the next step.

---
### 3. Supervised Learning:
Proceed the supervised training by adding a Linear Classifiera on the top of the last layer of the self-supervised model. 
We need to initialize the encoder part of your model with the weights from the saved SSL trained model. These pre-trained weights will serve as the starting point for further supervised learning. You can run the following command to run this step completely:
```
python train_linearCL.py --exp-dir <experiment-dir>   --ckpt <path-of-pretrained-ssl-model> --learning_rate 0.001 --epochs 300 --model resnet18 --cosine
```

---

## Cite ImSpect
<pre>
</pre>
