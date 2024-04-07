# leash_bio
This repo gathers the work I completed during a kaggle competition (https://www.kaggle.com/competitions/leash-BELKA). The goal of the competition was to design a model which determine the probability of link between a molecule and somes proteins.

The molecules were described as SMILE. I decided to turn the SMILE into graphs and then apply GCN on the molecule to find the probability. 

One of the main difficulty of this competition is the size of the data. The training set was about 3M molecules. 

Another difficulty is that the dataset's target is mostly made of 0, around 99,995% of it. Hence, we had to reweighted the dataset by only taking some 0 during training.