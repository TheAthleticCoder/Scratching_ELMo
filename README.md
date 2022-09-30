# **ELMo: Deep Contextualized Word Representations And Sentence Classification**

[![forthebadge](https://forthebadge.com/images/badges/made-with-python.svg)](https://forthebadge.com)
[![forthebadge](https://forthebadge.com/images/badges/fixed-bugs.svg)](https://forthebadge.com)

-----
## ***Objectives:***

This repository contains the codes for:
1. ELMo architecture from scratch using PyTorch 
2. Stacked Bi-LSTM - each of which gives the
embedding for a word in a sentence, and a trainable parameter for weighing the word embeddings
obtained at each layer of the ELMo network. 
3. Training the ELMo architecture on a 5-way classification task using the Yelp reviews
dataset.
4. Using the above to build a pipeline for training and evaluating your model on the classification task.

ELMo explores the construction of a layered representation of a word using stacked Bi-LSTM layers, weighing syntactic and semantic representations independently.

In ELMo, a stacked Bi-LSTM is utilised to capture natural language intricacies in a layered fashion, with lower layers focusing on syntactic features and higher layers focused on more complicated, semantic characteristics. Furthermore, by utilising a Bi-LSTM, we can capture the context for a given word in a sentence based on all of the surrounding words in that sentence. This bidirectional context of increasing complexity is created at each layer by simply concatenating the representations obtained from the forward and backward LSTMs at each layer. Finally, we total together the representations from each layer of the model (including the non-contextual input embeddings) to get our ELMo architecture's final contextual word representation.

-----

## ***File Structure:***

1. `ELMo_SenClass_Model.py` single file cleaned and commented code which contains dataset handling, preprocessing, ELMo model pretraining, finetuning based on Downstream Task and multi-class sentence classification. 
2. `Report.pdf` contains visualizations and analysis of the results of our model. Mentions the hyperparameters used as well as required metrics specified for analysis.
3. `Model_Paths.pdf` redirects you to an Outlook One Drive link containing ZIPs of all the models. The models are those keys which gave the best value (least loss) on the validation data. It also contains the word embeddings obtained after pretraining.
4. `extra` folder contains dump of `images` and `ipynb` versions for the code above.
-----

## ***Execution:***
The code can be executed by:
```c++
python3 <filename>.py
```
Ofcourse, you can only run those files ending with `.py`. 

When the model is run, the code snippet:
```c++
torch.save(model.state_dict(), 'model.pt')
```
stores the best parameters of the model which gives the lowest validation loss.
The code by itself calls the model back for testing purposes using:
```c++
model.load_state_dict(torch.load('model.pt'))
```
If the model has been successfully loaded, it returns
`<All keys matched successfully>`

-----
