# Dogs-vs-Cats-Image-classifier-with-keras-and-saved-trained-models
5 keras models trained on dogs vs cats dataset from kaggle with script to use any of those models to predict some image.

The code for any of the trained model saves model1.h5, model2.h5, model3.h5, model4.h5 and model5.h5 is in the respective model{n}.py file which uses tensorflow version 2.1.
In order to make a prediction on any .jpg image you need to use the predict.py file like this

```python predict.py "image file" "saved model file"```

So in order to make a prediction on the image "cat.jpg" using "model1.h5" you should run the following

```python predict.py "cat.jpg" "model1.h5"```

Here are the accuracies on the training data set of all models (if you just want to pick one you should take Model 5 as it is probably not too overfitted because of a dropout layer):

Model 1 --> 96%

Model 2 --> 88%

Model 3 --> 87%

Model 4 --> 88.19%

Model 5 --> 96.26%

(Note that model4.h5 is not present as it is too big with more than 25MB)
