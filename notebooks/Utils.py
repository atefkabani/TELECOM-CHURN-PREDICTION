import sys
import os
 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from Utilss import Utils as ut
import importlib
 
importlib.reload(ut)
def plot_confusion_matrix(y_test, y_pred , classes):    
    ut.plot_confusion_matrix(y_test, y_pred , classes)

def get_classification_report(y_test, y_pred):
    ut.get_classification_report(y_test, y_pred)

def get_accuracy(y_test, y_pred):
    ut.get_accuracy(y_test, y_pred)
    
def show_all_metrics(y_test, y_pred , classes):
    ut.show_all_metrics(y_test, y_pred , classes)