# Generating tweets using Machine Learning

This project creates new tweets using all previous tweets made by @realDonaldTrump. Inspired by his characteristic syntax and abuse of the platform, I wanted to see if my laptop could make tweets that could've been sent by the president himself. 

Using TensorFlow and Keras, I implemented a Recurrent Neural Network (RNN) to train off of a file of all tweets mined using the Twitter API and cleaned using the file [txtCleaner.py](./txtCleaner.py). Using Google Colab, the model can train in roughly 30 minutes and output semantically correct messages that closely mirror the user's syntax and diction. 

You can follow the project on Twitter [@MLTrumpTweets](https://twitter.com/MLTrumpTweets)!
