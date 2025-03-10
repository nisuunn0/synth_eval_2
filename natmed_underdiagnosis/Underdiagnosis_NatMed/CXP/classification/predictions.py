from classification.dataset import CheXpert
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import sklearn.metrics as sklm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


def make_pred_multilabel(model, test_df, val_df, PATH_TO_IMAGES, device):
    
    """
        This function gives predictions for test fold and calculates AUCs using previously trained model.
        
        Arguments:
        model: densenet-121 from torchvision previously fine tuned to training data
        val_df : validation dataframe -- to evaluate the threshold for binary classificaion
        test_df : test dataframe 
        PATH_TO_IMAGES: Path to the image directory on the server
        device: Device on which to run computation
        
        Returns:
        pred_df.csv: dataframe containing individual predictions for each test image
        Threshold.csv: the thereshold we used for binary prediction based on maximizing the F1 score over all labels on the validation set
        bipred.csv: dataframe containing individual binary predictions for each test image
        True.csv: dataframe containing true labels
        TestEval.csv: dataframe containing AUCs per label
        
    """

    BATCH_SIZE = 128 #32
    WORKERS = 16 #12
    print("First 5 elements of test_df:\n", test_df.head())
    #test_df = test_df.head(4000) # ONLY USE IF YOU HAVE LESS TESTING IMAGES THAN DECLARED BY THE TEST DF
    print("First 5 elements of test_df:\n", test_df.head())
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    dataset_test = CheXpert(test_df, PATH_TO_IMAGES=PATH_TO_IMAGES, transform=transforms.Compose([
        #transforms.Scale(256), # OLD DEPRECIATED
        transforms.Resize(256), # NEW ATTEMPT FIX
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        normalize]))
    test_loader = torch.utils.data.DataLoader(dataset_test, BATCH_SIZE, shuffle=True, num_workers=WORKERS, pin_memory=True)
    #test_loader = torch.utils.data.DataLoader(dataset_test, BATCH_SIZE, shuffle=False, num_workers=WORKERS, pin_memory=True)
    

    dataset_val = CheXpert(val_df, PATH_TO_IMAGES=PATH_TO_IMAGES, transform=transforms.Compose([ # OG REAL LIFE DATA (not necessarily real life data actually)
    #dataset_val = CheXpert(val_df, PATH_TO_IMAGES="/workspace/persistent/code/MEDFAIR/chexpert/chexpertchestxrays-u20210408/", transform=transforms.Compose([ # ROENTGEN TEST VALID AND TEST IN DIFFERENT DIRS
        #transforms.Scale(256), # OLD DEPRECIATED
        transforms.Resize(256), # NEW ATTEMPT FIX
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        normalize]))
    val_loader = torch.utils.data.DataLoader(dataset_val, BATCH_SIZE, shuffle=True, num_workers=WORKERS, pin_memory=True)


    size = len(test_df)
    print("Test _df size :", size)
    size = len(val_df)
    print("val_df size :", size)



    model = model.to(device)
    # to find this thresold, first we get the precision and recall withoit this, from there we calculate f1 score, using f1score, we found this theresold which has best precsision and recall.  Then this threshold activation are used to calculate our binary output.
    

    # ORIGINAL
    PRED_LABEL = ['No Finding',
                  'Enlarged Cardiomediastinum', 
                  'Cardiomegaly',
                  'Lung Opacity',
                  'Lung Lesion',
                  'Edema',
                  'Consolidation',
                  'Pneumonia',
                  'Atelectasis',
                  'Pneumothorax',
                  'Pleural Effusion',
                  'Pleural Other',     
                  'Fracture', 
                  'Support Devices']
    
    """
    # ATTRIBUTE CLASSIFER
    PRED_LABEL = [ 
                'Sex bin',
                'frontlat bin',   
                'AP',
                'PA',
                'Age Group 0-20',
                'Age Group 20-40',
                'Age Group 80+',
                ]
    """


    for mode in ["Threshold", "test"]: # ORIGINAL, ASSUMING YOU DONT HAVE THRESHOLD?
    #for mode in ["test"]: # ASSUMING YOU ALREADY HAVE COMPUTED THRESHOLD.csv
        # create empty dfs
        pred_df = pd.DataFrame(columns=["Path"])
        bi_pred_df = pd.DataFrame(columns=["Path"])
        true_df = pd.DataFrame(columns=["Path"])

        if mode == "Threshold":
            loader = val_loader
            Eval_df = pd.DataFrame(columns=["label",'bestthr'])    
            thrs = []            
        
        if mode == "test":
            loader = test_loader
            TestEval_df = pd.DataFrame(columns=["label", 'auc', "auprc"])
            
            
            # ORIGINAL
            Eval = pd.read_csv("./results/Threshold.csv")
            thrs = [Eval["bestthr"][Eval[Eval["label"]=="No Finding"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"]=="Enlarged Cardiomediastinum"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"]=="Cardiomegaly"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"]== "Lung Opacity"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"]=="Lung Lesion"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"]=="Edema"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"]=="Consolidation"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"]=="Pneumonia"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"]=="Atelectasis"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"]=="Pneumothorax"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"]=="Pleural Effusion"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"]=="Pleural Other"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"]=="Fracture"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"]=="Support Devices"].index[0]]]
            
            
            """
            # ATTRIBUTE CLASSIFER
            Eval = pd.read_csv("./results/Threshold.csv")
            thrs = [Eval["bestthr"][Eval[Eval["label"]=="Sex bin"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"]=="frontlat bin"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"]=="AP"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"]== "PA"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"]=="Age Group 0-20"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"]=="Age Group 20-40"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"]=="Age Group 80+"].index[0]],]
            """


        for i, data in enumerate(loader):
            inputs, labels, item = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            true_labels = labels.cpu().data.numpy()

            batch_size = true_labels.shape

            model.eval()
            with torch.no_grad():
                outputs = model(inputs)
                probs = outputs.cpu().data.numpy()

            # get predictions and true values for each item in batch
            for j in range(0, batch_size[0]):
                thisrow = {}
                bi_thisrow = {}
                truerow = {}

                truerow["Path"] = item[j]
                thisrow["Path"] = item[j]
                if mode == "test":
                    bi_thisrow["Path"] = item[j]

                # iterate over each entry in prediction vector; each corresponds to
                # individual label
                for k in range(len(PRED_LABEL)):
                    thisrow["prob_" + PRED_LABEL[k]] = probs[j, k]
                    truerow[PRED_LABEL[k]] = true_labels[j, k]

                    if mode == "test":
                        bi_thisrow["bi_" + PRED_LABEL[k]] = probs[j, k] >= thrs[k]

                #pred_df = pred_df.append(thisrow, ignore_index=True)
                pred_df = pd.concat([pred_df, pd.DataFrame([thisrow])], ignore_index=True)
                #true_df = true_df.append(truerow, ignore_index=True)
                true_df = pd.concat([true_df, pd.DataFrame([truerow])], ignore_index=True)
                if mode == "test":
                    #bi_pred_df = bi_pred_df.append(bi_thisrow, ignore_index=True)
                    bi_pred_df = pd.concat([bi_pred_df, pd.DataFrame([bi_thisrow])], ignore_index=True)
           
            if (i % 200 == 0):
                print(str(i * BATCH_SIZE))


 
        for column in true_df:
            if column not in PRED_LABEL:
                    continue
            actual = true_df[column]
            pred = pred_df["prob_" + column]
            
            thisrow = {}
            thisrow['label'] = column
            if mode == "test":
                bi_pred = bi_pred_df["bi_" + column]
                thisrow['auc'] = np.nan
                thisrow['auprc'] = np.nan
            else:

                thisrow['bestthr'] = np.nan

            try:


                if mode == "test":
                    thisrow['auc'] = sklm.roc_auc_score(
                        #actual.as_matrix().astype(int), pred.as_matrix())
                        actual.to_numpy().astype(int), pred.to_numpy())

                    thisrow['auprc'] = sklm.average_precision_score(
                        #actual.as_matrix().astype(int), pred.as_matrix())
                        actual.to_numpy().astype(int), pred.to_numpy())
                else:

                    #p, r, t = sklm.precision_recall_curve(actual.as_matrix().astype(int), pred.as_matrix())
                    p, r, t = sklm.precision_recall_curve(actual.to_numpy().astype(int), pred.to_numpy())
                    # Choose the best threshold based on the highest F1 measure
                    f1 = np.multiply(2, np.divide(np.multiply(p, r), np.add(r, p)))
                    bestthr = t[np.where(f1 == max(f1))]

                    thrs.append(bestthr)
                    thisrow['bestthr'] = bestthr[0]


            except BaseException:
                print("can't calculate auc for " + str(column))

            if mode == "Threshold":
                #Eval_df = Eval_df.append(thisrow, ignore_index=True)
                Eval_df = pd.concat([Eval_df, pd.DataFrame([thisrow])], ignore_index=True)



            if mode == "test":
                #TestEval_df = TestEval_df.append(thisrow, ignore_index=True)
                TestEval_df = pd.concat([TestEval_df, pd.DataFrame([thisrow])], ignore_index=True)


        pred_df.to_csv("results/preds.csv", index=False)
        true_df.to_csv("results/True.csv", index=False)

        if mode == "Threshold":
            Eval_df.to_csv("results/Threshold.csv", index=False)

        if mode == "test":
            TestEval_df.to_csv("results/TestEval.csv", index=False)
            bi_pred_df.to_csv("results/bipred.csv", index=False)

    # default is 14, attribute classifier is 7 here
    print("AUC ave:", TestEval_df['auc'].sum() / 14.0) # HERE FOR ATTRIBUTE CLASSIFER USE 7 OR WHATEVER YOUR NUMBER IS

    print("done")

    #return pred_df, Eval_df, bi_pred_df , TestEval_df # , bi_pred_df , Eval_bi_df # ORIGINAL
    return pred_df, bi_pred_df , TestEval_df # , bi_pred_df , Eval_bi_df # NO THRESHOLD COMPUTATION
    

