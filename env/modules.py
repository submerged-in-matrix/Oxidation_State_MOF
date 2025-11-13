import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer

from hyperopt import fmin, tpe, rand, anneal, hp, Trials, STATUS_OK
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.patheffects as pe
import seaborn as sns
import joblib
import pickle
import shap

import warnings
warnings.filterwarnings('ignore')
import sys
import requests

from pymatgen.ext.matproj import MPRester
from pymatgen.core.periodic_table import Element
from pymatgen.core import Composition
from pymatgen.core import Lattice, Structure, Element
from pymatgen.analysis.local_env import VoronoiNN
from matminer.featurizers.site import CrystalNNFingerprint, GaussianSymmFunc
from matminer.utils.data import MagpieData
import ast

from itertools import product