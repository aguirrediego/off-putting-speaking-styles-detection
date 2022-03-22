# Models for Detecting Off-Putting Speaking Styles

Paper Title: Comparison of Models for Detecting Off-Putting Speaking Styles

Authors: Diego Aguirre$, Nigel G. Ward, Jonathan E. Avila, and Heike Lehnert-LeHouillier

Abstract: In human-human interaction, speaking styles variation is pervasive. Modeling such variation has seen increasing interest, but there has been relatively little work on how best to discriminate among styles, and apparently none on how to exploit pretrained models for this. Moreover, little computational work has addressed questions of how styles are perceived, although this is often the most important aspect in terms of social and interpersonal relevance.  Here we develop models of whether an utterance is likely to be perceived as off-putting.  We explore different ways to leverage state-of-the-art pretrained representations, namely those for TRILL, COLA, and TRILLsson.  We obtain reasonably good performance in detecting off-putting styles, and find that architectures and learned representations designed to capture multi-second temporal information perform better.

In this repository, you will find our initialization code (Keras). If you would like to add our initialization technique
to your project, see 'main/simple_example.py' to learn how to do so. It's a fairly simple process.

Files in this repository:

extract_embeddings.py: Script that extracts TRILL19, TRILLsson5, and COLA embeddings. To extract low-level features, 
we used the MidLevel Prosodic Features Toolkit (https://github.com/nigelgward/midlevel) 

train_models.py: Code we used to train and evaluate all linear and recurrent models presented in the paper