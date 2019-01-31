# Load libraries
import numpy as np
import pickle
import matplotlib.pyplot as plt

def plotROC(FPR, TPR, AUC, estimatorName, title):
    plt.figure()
    for i in range(0, len(estimatorName)):
        plt.plot(FPR[i], TPR[i], label=estimatorName[i])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves For Models Using ' + title + ' Features')
    plt.legend(loc="lower right")
    plt.show()

# ------------------------------------------------------------------------------
with open("../rocResults/24hr_AllFeatures_Estimators.txt", "rb") as fp:   # Unpickling
    AllFeatures_Estimators = pickle.load(fp)
with open("../rocResults/24hr_AllFeatures_AUC.txt", "rb") as fp:   # Unpickling
    AllFeatures_AUC = pickle.load(fp)
with open("../rocResults/24hr_AllFeatures_FPR.txt", "rb") as fp:   # Unpickling
    AllFeatures_FPR = pickle.load(fp)
with open("../rocResults/24hr_AllFeatures_TPR.txt", "rb") as fp:   # Unpickling
    AllFeatures_TPR = pickle.load(fp)

with open("../rocResults/24hr_BestFeatures_Estimators.txt", "rb") as fp:   # Unpickling
    BestFeatures_Estimators = pickle.load(fp)
with open("../rocResults/24hr_BestFeatures_AUC.txt", "rb") as fp:   # Unpickling
    BestFeatures_AUC = pickle.load(fp)
with open("../rocResults/24hr_BestFeatures_FPR.txt", "rb") as fp:   # Unpickling
    BestFeatures_FPR = pickle.load(fp)
with open("../rocResults/24hr_BestFeatures_TPR.txt", "rb") as fp:   # Unpickling
    BestFeatures_TPR = pickle.load(fp)

with open("../rocResults/24hr_BestEqualFeatures_Estimators.txt", "rb") as fp:   # Unpickling
    BestEqualFeatures_Estimators = pickle.load(fp)
with open("../rocResults/24hr_BestEqualFeatures_AUC.txt", "rb") as fp:   # Unpickling
    BestEqualFeatures_AUC = pickle.load(fp)
with open("../rocResults/24hr_BestEqualFeatures_FPR.txt", "rb") as fp:   # Unpickling
    BestEqualFeatures_FPR = pickle.load(fp)
with open("../rocResults/24hr_BestEqualFeatures_TPR.txt", "rb") as fp:   # Unpickling
    BestEqualFeatures_TPR = pickle.load(fp)

# Make plot for the 5 models and all feature sets
plotROC(AllFeatures_FPR[0:5], AllFeatures_TPR[0:5], AllFeatures_AUC[0:5], AllFeatures_Estimators[0:5], "All")

# Make plot for the 3 average models and all feature sets
plotROC(AllFeatures_FPR[5:8], AllFeatures_TPR[5:8], AllFeatures_AUC[5:8], AllFeatures_Estimators[5:8], "All")

# Make plot for the 5 models and best feature sets
plotROC(BestFeatures_FPR[0:5], BestFeatures_TPR[0:5], BestFeatures_AUC[0:5], BestFeatures_Estimators[0:5], "Best")

# Make plot for the 3 average models and best feature sets
plotROC(BestFeatures_FPR[5:8], BestFeatures_TPR[5:8], BestFeatures_AUC[5:8], BestFeatures_Estimators[5:8], "Best")

# Make plot for the 5 models and BestEqual feature sets
plotROC(BestEqualFeatures_FPR[0:5], BestEqualFeatures_TPR[0:5], BestEqualFeatures_AUC[0:5], BestEqualFeatures_Estimators[0:5], "Best + Equal")

# Make plot for the 3 average models and BestEqual feature sets
plotROC(BestEqualFeatures_FPR[5:8], BestEqualFeatures_TPR[5:8], BestEqualFeatures_AUC[5:8], BestEqualFeatures_Estimators[5:8], "Best + Equal")
