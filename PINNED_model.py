import pandas as pd
import time
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from tensorflow import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from random import randint, shuffle
import time
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, recall_score, precision_score, 
confusion_matrix, ConfusionMatrixDisplay, matthews_corrcoef, f1_score)

def display_metrics(probs, targets, history, threshold=0.5):
    """
    Displays model performance metrics, confusion matrix, and a histogram of 
    predicted scores of a binary TensorFlow classifier model.
        
    Args:
        probs (tf.Tensor): druggability probabilities assigned by the model
        targets (pd.Series or other list-like): true labels
        history (tf.History): model training history
        threshold (float): druggability score required to designate a protein as 
            druggable

    Returns:
        None
    
    """
    
    # Displays plots of the model's loss and AUC over the training epochs for 
    # the training and validation sets
    
    print("Neural network performance")

    plt.plot(history.history['loss'], label='Train loss')
    plt.plot(history.history['val_loss'], label='validation loss')
    plt.xlabel('Epochs'), plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.plot(history.history['auc'], label='Train AUC')
    plt.plot(history.history['val_auc'], label='validation AUC')
    plt.xlabel('Epochs'), plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    
    # preds are predicted class labels obtained from the class probabilities
    
    preds = np.array([1 if x >= threshold else 0 for x in probs])
    
    
    # Computes and prints performance metrics for the model
        
    true_neg = np.sum((targets == 0) & (preds == 0))
    false_pos = np.sum((targets == 0) & (preds == 1))

    sensitivity = 100 * recall_score(y_true=targets, y_pred=preds)
    specificity = 100 * true_neg / (true_neg + false_pos)
    accuracy = 100 * sum(preds == targets) / len(targets)
    auc = 100 * roc_auc_score(y_true=targets, y_score=probs)

    print("This model:")
    print("SN:",
              round(sensitivity, 2),
              '\\t',
              "SP:",
              round(specificity, 2),
              '\\t',
              "ACC:",
              round(accuracy, 2),
              '\\t',
              "AUC:",
              round(auc, 2),
             )
    
    
    # Generates and displays the confusion matrix

    cm = confusion_matrix(targets, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                                  display_labels=["No drug", "Drug"])
    disp.plot()
    plt.show()
    
    probs = np.array(probs)
    
    
    # Displays a histogram of class probabilities from 0 to 1, with positives 
    # and negatives separated
            
    pos_probs = np.array([probs[x] for x in range(len(probs)) 
                          if targets.iloc[x] == 1]).flatten()
    neg_probs = np.array([probs[x] for x in range(len(probs)) 
                          if targets.iloc[x] == 0]).flatten()
        
    plt.hist([pos_probs, neg_probs], 
             alpha=0.5, 
             bins=48, 
             density=True, 
             histtype = 'stepfilled', 
             label=['Drugged proteins', 'Undrugged proteins'])
    plt.xlabel('Druggability score')
    plt.legend()
    plt.show()
            
    return None
                                   
    
    
def fit_multiscore_nn(train_inputs, 
                      train_labels, 
                      validation_inputs, 
                      validation_labels):
    """
    Creates and trains the neural network classifier model and returns the model
    along with its training history.
    
    Args:
        train_inputs (list of DataFrames): List of the inputs to each of the 4 
            networks as DataFrames for the training set
        train_labels (pd.Series or other list-like): Labels for the training 
            data
        validation_inputs (list of DataFrames): List of the inputs to each of 
            the 4 networks as DataFrames for the validation set
        validation_labels (pd.Series or other list-like): Labels for the 
            validation data

    Returns:
        model (tf.keras.Model): Trained model
        history (tf.History): Model training history
        
    """
    
    # Generates the subscore for the sequence and structure subnetwork
        
    seq_and_struc_input = keras.Input(shape=(122,))
    seq_and_struc = layers.Dense(units=64, 
                                 activation='relu', 
                                 kernel_regularizer=l2)(seq_and_struc_input)
    seq_and_struc = layers.Dense(units=1, kernel_regularizer=l2)(seq_and_struc)
    
    
    
    # Generates the subscore for the localization subnetwork
    
    localization_input = keras.Input(shape=(558,))
    localization = layers.Dense(units=512, 
                                activation='relu',
                                kernel_regularizer=l2)(localization_input)
    localization = layers.Dense(units=1, kernel_regularizer=l2)(localization)
    
    
    
    # Generates the subscore for the biological functions subnetwork
    
    bio_func_input = keras.Input(shape=(3464,))
    bio_func = layers.Dense(units=2048, 
                            activation='relu', 
                            kernel_regularizer=l2)(bio_func_input)
    k = 5
    split_sets = k_split(train_validation_set, k)

    start_time = time.time()
    verbose = True
    aucs = []
    final_validation_losses = []
    n_trials = 5
    oversamples = vanilla_oversample(split_sets)
    all_false_negs = []

    for i in range(n_trials):
        train_sets = [oversamples[x] for x in range(k) if x != i]
        train_set = pd.concat(train_sets)
        validation_set = split_sets[i]
        (train_names, train_seq_and_struc, train_localization, train_bio_func, train_network_info, train_labels) = process_set(train_set)
        (validation_names, validation_seq_and_struc, validation_localization, validation_bio_func, validation_network_info, validation_labels) = process_set(validation_set)
        train_inputs = [train_seq_and_struc, train_localization, train_bio_func, train_network_info]
        validation_inputs = [validation_seq_and_struc, validation_localization, validation_bio_func, validation_network_info]
        print("Split #: ", i + 1)
        model, history = fit_multiscore_nn(train_inputs, train_labels, validation_inputs, validation_labels)
        probs = tf.keras.activations.sigmoid(model.predict(validation_inputs))
        display_metrics(probs, validation_labels, history)
        aucs.append(100 * roc_auc_score(y_true=validation_labels, y_score=probs))
        score = model.evaluate(validation_inputs, validation_labels, verbose=False)
        final_validation_losses.append(score[0])
        false_negs = get_false_negatives(probs, validation_labels, validation_names)
        all_false_negs += false_negs

    print(aucs)
    elapsed_time = time.time() - start_time
    print("Time elapsed (s): ", elapsed_time)

    def get_subscores(features, model):
        subscores_model = keras.Model(inputs=model.input, outputs=model.get_layer('subscores').output)
        model, history = fit_multiscore_nn(train_inputs, train_labels, validation_inputs, validation_labels)
        probs = tf.keras.activations.sigmoid(model.predict(validation_inputs))
        display_metrics(probs, validation_labels, history)
        aucs.append(100 * roc_auc_score(y_true=validation_labels, y_score=probs))
        score = model.evaluate(validation_inputs, validation_labels, verbose=False)
        final_validation_losses.append(score[0])
        false_negs = get_false_negatives(probs, validation_labels, validation_names)
        all_false_negs += false_negs

        print(aucs)
        elapsed_time = time.time() - start_time
        print("Time elapsed (s): ", elapsed_time)

    def get_subscores(features, model):
        subscores_model = keras.Model(inputs=model.input, outputs=model.get_layer('subscores').output)
        subscores_output = subscores_model.predict(features, verbose=False)
        total_scores = tf.math.reduce_sum(subscores_output, axis=1)
        drug_probs = tf.keras.activations.sigmoid(tf.reshape(total_scores, shape=(len(total_scores),1))) * 100
        subscores_output = tf.concat([drug_probs, subscores_output], axis=1)
        return subscores_output
