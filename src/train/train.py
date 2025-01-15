import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam as torchAdam
from torch.optim import AdamW as torchAdamW
import keras
import pandas as pd
import argparse
import json
import os
import pickle
import copy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA 
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, precision_recall_fscore_support, matthews_corrcoef, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import RandomOverSampler
from keras.models import Model
from keras.optimizers import Adam, AdamW
from keras.models import load_model
from keras.metrics import AUC
from keras.callbacks import ModelCheckpoint
from eipy.ei import EnsembleIntegration

from src.models.models import DistilBERTSeqClassifier, BERTClassifier, W2V2Classifier, B2AIW2V2Classifier, GauderClassifier, Autoencoder, EIPredictors, FastEIPredictors
from src.models.models import Autoencoder, EIPredictors, FastEIPredictors, BasePredictors, KUModel, DA4ADClassifier, MISAClassifier, MFNClassifier, SigNLTKPredictors
from src.models.models import BCLSTMClassifier, BCLSTMUnimodalClassifier, BCLSTMTextCNN 
import src.models.models as models
from src.data.dataloaders import StandardDataGenerator, AutoencoderDataGenerator, DA4ADDataset, MISADataset, MFNDataset, BCLSTMDataset
import src.data.dataloaders as dataloaders
from src.utils import utils 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process configuration file for script.")
    parser.add_argument("--dataset_config", type=str, help="Path to the JSON configuration file.")
    parser.add_argument("--training_config", type=str, help="Which embeddings to train on")
    parser.add_argument("--n_folds", type=int, default=20, help="Number of train/test splits to generate")
    parser.add_argument("--model_name", type=str, default="model", help="N")
    parser.add_argument("--offset", type=int, default=0, help="Offset for fold random seed")
    parser.add_argument('--make_val_set', action='store_true')

    args = parser.parse_args()

    with open(args.dataset_config) as f:
        dataset_config = json.load(f)
    with open(args.training_config) as f:
        training_config = json.load(f)

    print('Configs parsed')

    train_dataset = [i for i in dataset_config['datasets'] if i['setname']=='training'][0]
    groundtruth_df, data_df = utils.prepare_dataframes(train_dataset, training_config, cohort=dataset_config['cohort'])
    breakpoint()


    print('Groundtruth df generated')

    all_scores = []
    offset = args.offset
    make_val = False
    # my_auc = AUC()
    for ii in range(offset,(args.n_folds+offset)):
        print(f'Fold {ii+1} of {args.n_folds}')
        mci_preds = []
        mmse_scores = []
        test_size = 0.25

        if not args.make_val_set:
            train_df, test_df = train_test_split(groundtruth_df, test_size=test_size, stratify=groundtruth_df['dx'], random_state=ii)

            resamp_train_df = utils.random_oversample_dataframe(train_df)

            # Get the resampled subjects
            resamp_train_subjects = resamp_train_df['Subject']
            # Create a dataframe to count occurrences of each subject
            subject_counts = resamp_train_subjects.value_counts().reset_index()
            subject_counts.columns = ['Subject', 'Count']
            # Merge the subject counts with data_df to get the repeated instances
            train_data_df = data_df.merge(subject_counts, on='Subject', how='inner')
            # Repeat each row according to its count
            train_data_df = train_data_df.loc[train_data_df.index.repeat(train_data_df['Count'])].reset_index(drop=True)
            # Drop the count column as it's no longer needed
            train_data_df = train_data_df.drop(columns=['Count'])

            mean_age = train_data_df['age'].mean()
            std_age = train_data_df['age'].std() + 1e-7
            train_data_df['age'] = (train_data_df['age'] - mean_age) / (std_age)
            train_tkdnames = train_data_df['tkdname'].drop_duplicates()

            test_subjects = test_df['Subject']
            test_data_df = data_df[data_df['Subject'].isin(test_subjects)]
            test_data_df['age'] = (test_data_df['age'] - mean_age) / (std_age)
            test_tkdnames = test_data_df['tkdname'].drop_duplicates()
        else:
            print('Making Validation Data')
            temp_train_df, test_df = train_test_split(groundtruth_df, test_size=test_size, stratify=groundtruth_df['dx'], random_state=ii)
            train_df, val_df = train_test_split(temp_train_df, test_size=test_size, stratify=temp_train_df['dx'], random_state=ii)

            resamp_train_df = utils.random_oversample_dataframe(train_df)
            # Get the resampled subjects
            resamp_train_subjects = resamp_train_df['Subject']
            # Create a dataframe to count occurrences of each subject
            subject_counts = resamp_train_subjects.value_counts().reset_index()
            subject_counts.columns = ['Subject', 'Count']
            # Merge the subject counts with data_df to get the repeated instances
            train_data_df = data_df.merge(subject_counts, on='Subject', how='inner')
            # Repeat each row according to its count
            train_data_df = train_data_df.loc[train_data_df.index.repeat(train_data_df['Count'])].reset_index(drop=True)
            # Drop the count column as it's no longer needed
            train_data_df = train_data_df.drop(columns=['Count'])

            mean_age = train_data_df['age'].mean()
            std_age = train_data_df['age'].std() + 1e-7
            train_data_df['age'] = (train_data_df['age'] - mean_age) / (std_age)
            train_tkdnames = train_data_df['tkdname'].drop_duplicates()

            val_subjects = val_df['Subject']
            val_data_df = data_df[data_df['Subject'].isin(val_subjects)]
            val_data_df['age'] = (val_data_df['age'] - mean_age) / (std_age)
            val_tkdnames = val_data_df['tkdname'].drop_duplicates()

            test_subjects = test_df['Subject']
            test_data_df = data_df[data_df['Subject'].isin(test_subjects)]
            test_data_df['age'] = (test_data_df['age'] - mean_age) / (std_age)
            test_tkdnames = test_data_df['tkdname'].drop_duplicates()

            print('done making validation data')

        if training_config['architecture']=='SSL':
            if training_config['target'] == "MCI":
                train_generator = StandardDataGenerator(
                    dataframe=train_data_df, 
                    batch_size=training_config['batch_size'], 
                    shuffle=True, 
                    label=training_config['target']
                )
                test_generator = StandardDataGenerator(
                    dataframe=test_data_df, 
                    batch_size=training_config['batch_size'], 
                    shuffle=True, 
                    label=training_config['target']
                )
                loss = 'binary_crossentropy'
            if training_config['modality']=='BERT':
                model = BERTClassifier(
                    input_size=data_input_size, 
                    dense_units=training_config['dense_units'], 
                    output_head=training_config['target'], 
                    dropout_rate=training_config['dropout_rate'], 
                    activation=training_config['activation'], 
                    use_demographics=False,
                    num_mmse_groups=15) 
            elif training_config['modality']=='W2V2':
                model = W2V2Classifier(
                    input_size=data_input_size, 
                    dense_units=training_config['dense_units'], 
                    output_head=training_config['target'], 
                    dropout_rate=training_config['dropout_rate'], 
                    activation=training_config['activation'], 
                    use_demographics=False,
                    num_mmse_groups=15) 

            class_weights = compute_class_weight(
                class_weight='balanced', 
                classes=train_df['dx'].unique(), 
                y=train_df['dx'].values)
            weights_dict = {index: value for index, value in enumerate(class_weights)}
            optimizer = Adam(learning_rate=training_config['learning_rate'])
            my_auc = AUC()
            model.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
            
            model.fit(
                train_generator,
                validation_data=test_generator,
                batch_size=training_config['batch_size'],
                epochs=training_config['n_epochs'],
                class_weight=weights_dict,
                verbose=2
            )

            train_scores = utils.get_scores(train_tkdnames, train_data_df, model)
            test_scores = utils.get_scores(test_tkdnames, test_data_df, model)

            temp_df = {}
            temp_df['Outer Fold'] = ii
            temp_df['Model'] = args.model_name
            for key, val in test_scores.items():
                temp_df[key] = val
            all_scores.append(temp_df)

        elif training_config['architecture']=='B2AI':
            if training_config['target'] == "MCI":
                train_generator = StandardDataGenerator(
                    dataframe=train_data_df, 
                    batch_size=training_config['batch_size'], 
                    shuffle=True, 
                    label=training_config['target']
                )
                test_generator = StandardDataGenerator(
                    dataframe=test_data_df, 
                    batch_size=training_config['batch_size'], 
                    shuffle=True, 
                    label=training_config['target']
                )
                loss = 'binary_crossentropy'
            model = B2AIW2V2Classifier(
                    input_size=data_input_size, 
                    dense_units=training_config['dense_units'], 
                    output_head=training_config['target'], 
                    dropout_rate=training_config['dropout_rate'], 
                    activation=training_config['activation'], 
                    use_demographics=False,
                    num_mmse_groups=15) 

            class_weights = compute_class_weight(
                class_weight='balanced', 
                classes=train_df['dx'].unique(), 
                y=train_df['dx'].values)
            weights_dict = {index: value for index, value in enumerate(class_weights)}
            optimizer = Adam(learning_rate=training_config['learning_rate'])
            my_auc = AUC()
            model.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
            
            model.fit(
                train_generator,
                validation_data=test_generator,
                batch_size=training_config['batch_size'],
                epochs=training_config['n_epochs'],
                class_weight=weights_dict,
                verbose=2
            )

            train_scores = utils.get_scores(train_tkdnames, train_data_df, model)
            test_scores = utils.get_scores(test_tkdnames, test_data_df, model)

            temp_df = {}
            temp_df['Outer Fold'] = ii
            temp_df['Model'] = args.model_name
            for key, val in test_scores.items():
                temp_df[key] = val
            all_scores.append(temp_df)

        elif training_config['architecture']=='GAUDER':
            if training_config['target'] == "MCI":
                train_generator = StandardDataGenerator(
                    dataframe=train_data_df, 
                    batch_size=training_config['batch_size'], 
                    shuffle=True, 
                    label=training_config['target']
                )
                test_generator = StandardDataGenerator(
                    dataframe=test_data_df, 
                    batch_size=training_config['batch_size'], 
                    shuffle=True, 
                    label=training_config['target']
                )
                loss = 'binary_crossentropy'
            model = GauderClassifier(
                    input_size=data_input_size, 
                    filter_sizes=training_config['filter_sizes'],
                    kernel_sizes=training_config['kernel_sizes'],
                    strides=training_config['strides'], 
                    output_head=training_config['target'], 
                    activation=training_config['activation'], 
                    use_demographics=False,
                    num_mmse_groups=15) 

            class_weights = compute_class_weight(
                class_weight='balanced', 
                classes=train_df['dx'].unique(), 
                y=train_df['dx'].values)
            weights_dict = {index: value for index, value in enumerate(class_weights)}
            optimizer = Adam(learning_rate=training_config['learning_rate'])
            my_auc = AUC()
            model.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
            
            model.fit(
                train_generator,
                validation_data=test_generator,
                batch_size=training_config['batch_size'],
                epochs=training_config['n_epochs'],
                class_weight=weights_dict,
                verbose=2
            )

            train_scores = utils.get_scores(train_tkdnames, train_data_df, model)
            test_scores = utils.get_scores(test_tkdnames, test_data_df, model)

            temp_df = {}
            temp_df['Outer Fold'] = ii
            temp_df['Model'] = args.model_name
            for key, val in test_scores.items():
                temp_df[key] = val
            all_scores.append(temp_df)

        elif training_config['architecture']=='LEUVEN':
            if training_config['target'] == "MCI":

                # train_df, val_df = train_test_split(train_df, test_size=0.25, stratify=train_df['dx'], random_state=ii)
                # train_subjects = train_df['Subject']
                # train_data_df = data_df[data_df['Subject'].isin(train_subjects)]
                # mean_age = train_data_df['age'].mean()
                # std_age = train_data_df['age'].std() + 1e-7
                # train_data_df['age'] = (train_data_df['age'] - mean_age) / (std_age)
                # train_tkdnames = train_data_df['tkdname'].drop_duplicates()

                # val_subjects = val_df['Subject']
                # val_data_df = data_df[data_df['Subject'].isin(val_subjects)]
                # val_data_df['age'] = (val_data_df['age'] - mean_age) / (std_age)
                # val_tkdnames = val_data_df['tkdname'].drop_duplicates()

                train_generator = StandardDataGenerator(
                    dataframe=train_data_df, 
                    batch_size=training_config['batch_size'], 
                    shuffle=True, 
                    label=training_config['target']
                )
                # val_generator = StandardDataGenerator(
                #     dataframe=val_data_df, 
                #     batch_size=val_data_df.shape[0], 
                #     shuffle=True, 
                #     label=training_config['target']
                # )
                test_generator = StandardDataGenerator(
                    dataframe=test_data_df, 
                    batch_size=training_config['batch_size'], 
                    shuffle=True, 
                    label=training_config['target']
                )
                loss = 'binary_crossentropy'
            
            model = KUModel(
                    hidden_dim=training_config['hidden_dim'],  
                    dropout_rate=training_config['dropout_rate'], 
                    activation=training_config['activation'], 
                    use_demographics=False,
                    num_mmse_groups=15
            )

            class_weights = compute_class_weight(
                class_weight='balanced', 
                classes=train_df['dx'].unique(), 
                y=train_df['dx'].values)
            weights_dict = {index: value for index, value in enumerate(class_weights)}
            # optimizer = AdamW(
            #                 learning_rate=training_config['learning_rate'],
            #                 weight_decay=1e-2)
            optimizer = Adam(learning_rate=training_config['learning_rate'])
            
            model.compile(optimizer=optimizer, loss=loss, metrics=my_auc)
            lr_scheduler = utils.LinearWarmUpScheduler(
                            training_config['learning_rate'], 
                            training_config['warmup_steps'])

            # checkpoint = ModelCheckpoint('checkpoint.weights.h5', 
            #         monitor="val_auc", mode="max", save_weights_only=True, 
            #         save_best_only=True, verbose=1)
            
            model.fit(
                train_generator,
                # validation_data=val_generator,
                # validation_steps = 1,
                epochs=training_config['n_epochs'],
                class_weight=weights_dict,
                # callbacks=[checkpoint],
                verbose=2
            )

            train_scores = utils.get_scores(train_tkdnames, train_data_df, model)
            test_scores = utils.get_scores(test_tkdnames, test_data_df, model)
            temp_df = {}
            temp_df['Outer Fold'] = ii
            temp_df['Model'] = args.model_name
            for key, val in test_scores.items():
                temp_df[key] = val
            all_scores.append(temp_df) 

            # model.load_weights('checkpoint.weights.h5')

            # train_scores = utils.get_scores(train_tkdnames, train_data_df, model)
            # test_scores = utils.get_scores(test_tkdnames, test_data_df, model)
            # temp_df = {}
            # temp_df['Outer Fold'] = ii
            # temp_df['Model'] = args.model_name+' Best'
            # for key, val in test_scores.items():
            #     temp_df[key] = val
            # all_scores.append(temp_df) 

            class_balance_thresh = (1-(train_data_df['dx'].values.sum()/train_data_df['dx'].values.shape[0]))
            rand_df = []
            for jj in range(100):
                y_pred = np.random.uniform(size=test_data_df['dx'].values.shape)
                pred_score = utils.scores(test_data_df['dx'].values.astype(int), y_pred)
                pred_score['Accuracy'] = accuracy_score(
                    test_data_df['dx'].values.astype(int),
                    (y_pred>class_balance_thresh).astype(int))
                rand_df.append(pred_score)
            cat_rand_df = pd.DataFrame(rand_df)
            pred_score = cat_rand_df.mean()

            temp_df = {}
            temp_df['Outer Fold'] = ii
            temp_df['Model'] = 'Random'
            for key, val in pred_score.items():
                temp_df[key] = val
            all_scores.append(temp_df)

        elif training_config['architecture']=='SeqCls':
            train_df, val_df = train_test_split(train_df, test_size=0.25, stratify=train_df['dx'], random_state=ii)
            train_subjects = train_df['Subject']
            train_data_df = data_df[data_df['Subject'].isin(train_subjects)]
            mean_age = train_data_df['age'].mean()
            std_age = train_data_df['age'].std() + 1e-7
            train_data_df['age'] = (train_data_df['age'] - mean_age) / (std_age)
            train_tkdnames = train_data_df['tkdname'].drop_duplicates()

            val_subjects = val_df['Subject']
            val_data_df = data_df[data_df['Subject'].isin(val_subjects)]
            val_data_df['age'] = (val_data_df['age'] - mean_age) / (std_age)
            val_tkdnames = val_data_df['tkdname'].drop_duplicates()

            train_generator = StandardDataGenerator(
                dataframe=train_data_df, 
                batch_size=training_config['batch_size'], 
                shuffle=True, 
                label=training_config['target']
            )
            val_generator = StandardDataGenerator(
                dataframe=val_data_df, 
                batch_size=val_data_df.shape[0], 
                shuffle=True, 
                label=training_config['target']
            )
            test_generator = StandardDataGenerator(
                dataframe=test_data_df, 
                batch_size=training_config['batch_size'], 
                shuffle=True, 
                label=training_config['target']
            )
            loss = 'binary_crossentropy'
            if training_config['modality']=='DistilBERT':
                model = DistilBERTSeqClassifier(
                    distilbert_model=dataset_config['distilbert_model'])
            

            class_weights = compute_class_weight(
                class_weight='balanced', 
                classes=train_df['dx'].unique(), 
                y=train_df['dx'].values)
            weights_dict = {index: value for index, value in enumerate(class_weights)}
            optimizer = Adam(learning_rate=training_config['learning_rate'])
            model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy',my_auc])

            checkpoint = ModelCheckpoint('checkpoint.weights.h5', 
                    monitor="val_auc", mode="max", save_weights_only=True, 
                    save_best_only=True, verbose=1)
            
            model.fit(
                train_generator,
                validation_data=val_generator,
                batch_size=training_config['batch_size'],
                epochs=training_config['n_epochs'],
                class_weight=weights_dict,
                callbacks=[checkpoint],
                verbose=2
            )

            train_scores = utils.get_scores(train_tkdnames, train_data_df, model)
            test_scores = utils.get_scores(test_tkdnames, test_data_df, model)
            temp_df = {}
            temp_df['Outer Fold'] = ii
            temp_df['Model'] = args.model_name
            for key, val in test_scores.items():
                temp_df[key] = val
            all_scores.append(temp_df) 

            model.load_weights('checkpoint.weights.h5')

            train_scores = utils.get_scores(train_tkdnames, train_data_df, model)
            test_scores = utils.get_scores(test_tkdnames, test_data_df, model)
            temp_df = {}
            temp_df['Outer Fold'] = ii
            temp_df['Model'] = args.model_name+' Best'
            for key, val in test_scores.items():
                temp_df[key] = val
            all_scores.append(temp_df) 

            class_balance_thresh = (1-(train_data_df['dx'].values.sum()/train_data_df['dx'].values.shape[0]))
            rand_df = []
            for jj in range(100):
                y_pred = np.random.uniform(size=test_data_df['dx'].values.shape)
                pred_score = utils.scores(test_data_df['dx'].values, y_pred)
                pred_score['Accuracy'] = accuracy_score(
                    test_data_df['dx'].values.astype(int),
                    (y_pred>class_balance_thresh).astype(int))
                rand_df.append(pred_score)
            cat_rand_df = pd.DataFrame(rand_df)
            pred_score = cat_rand_df.mean()

            temp_df = {}
            temp_df['Outer Fold'] = ii
            temp_df['Model'] = 'Random'
            for key, val in pred_score.items():
                temp_df[key] = val
            all_scores.append(temp_df)

            del model

        elif training_config['architecture']=='AE':

            if training_config['target'] == "MCI":
                train_generator = AutoencoderDataGenerator(
                    dataframe=train_data_df, 
                    batch_size=training_config['batch_size'], 
                    shuffle=True, 
                    modality = training_config['modality']
                )
                test_generator = AutoencoderDataGenerator(
                    dataframe=test_data_df, 
                    batch_size=training_config['batch_size'], 
                    shuffle=True, 
                    modality = training_config['modality']
                )
                loss = 'binary_crossentropy'
            
            autoencoder = Autoencoder(
                input_size=data_input_size, 
                encoder_hidden_units=training_config['encoder_hidden_units'],
                decoder_hidden_units=training_config['decoder_hidden_units'], 
                dropout_rate=0.0, 
                activation='tanh',
                use_demographics=False,
                modality=training_config['modality']
            )

            optimizer = Adam(learning_rate=training_config['learning_rate'])
            loss = 'mean_squared_error'
            autoencoder.compile(optimizer=optimizer, loss=loss)
            autoencoder.fit(
                train_generator,
                validation_data=test_generator,
                batch_size=training_config['batch_size'],
                epochs=training_config['n_epochs'],
                verbose=2
            )
            
            if training_config['modality'] == 'BERT':
                train_ae_embeddings, train_y_true = utils.get_bert_ae_embeddings_per_tkdname(train_subjects, train_data_df, autoencoder.encoder)
                test_ae_embeddings, test_y_true = utils.get_bert_ae_embeddings_per_tkdname(test_subjects, test_data_df, autoencoder.encoder)
            elif training_config['modality'] == 'W2V2':
                train_ae_embeddings, train_y_true = utils.get_w2v2_ae_embeddings_per_tkdname(train_subjects, train_data_df, autoencoder.encoder)
                test_ae_embeddings, test_y_true = utils.get_w2v2_ae_embeddings_per_tkdname(test_subjects, test_data_df, autoencoder.encoder)

            train_ae_embeddings, test_ae_embeddings = np.squeeze(train_ae_embeddings), np.squeeze(test_ae_embeddings)
            
            data_train = {training_config['modality']: train_ae_embeddings}
            data_test = {training_config['modality']: test_ae_embeddings}

            ei_predictors = FastEIPredictors()
            EI = EnsembleIntegration(
                            base_predictors=ei_predictors.base_predictors,
                            k_outer=5,
                            k_inner=5,
                            n_samples=1,
                            sampling_strategy='oversampling',
                            sampling_aggregation=None,
                            n_jobs=-1,
                            random_state=ii,
                            project_name="NLP_CI",
                            model_building=True,
                            )

            for name, modality_data in data_train.items():
                EI.fit_base(modality_data, train_y_true, modality_name=name)

            base_models_list = copy.deepcopy(EI.final_models["base models"][training_config['modality']])  # loads all base models for a particular modality
            model_dicts = [dictionary for dictionary in base_models_list]

            for base in model_dicts:
                model = pickle.loads(base["pickled model"])
                temp_df = {}
                temp_df['Outer Fold'] = ii
                temp_df['Ensemble Model'] = base["model name"]
                y_pred = model.predict_proba(data_test[training_config['modality']])
                y_pred = y_pred[:,1]
                pred_score = utils.scores(test_y_true, y_pred)
                for key, val in pred_score.items():
                    temp_df[key] = val
                all_scores.append(temp_df)
            
            EI.fit_ensemble(ensemble_predictors=ei_predictors.ensemble_predictors)

            for ensemble, _ in ei_predictors.ensemble_predictors.items():
                temp_df = {}
                temp_df['Outer Fold'] = ii
                temp_df['Ensemble Model'] = ensemble
                y_pred = EI.predict(X_dict=data_test, ensemble_model_key=ensemble)
                pred_score = utils.scores(test_y_true, y_pred)
                for key, val in pred_score.items():
                    temp_df[key] = val
                all_scores.append(temp_df)

            temp_df = {}
            temp_df['Outer Fold'] = ii
            temp_df['Ensemble Model'] = 'Random'
            y_pred = np.random.uniform(size=test_data_df['dx'].values.shape)
            pred_score = utils.scores(test_data_df['dx'].values, y_pred)
            for key, val in pred_score.items():
                temp_df[key] = val
            all_scores.append(temp_df)

        elif training_config['architecture']=='EI':
            if training_config['target'] == "MCI":
                lang_train_embeddings = np.vstack(
                    [np.load(filepath) for filepath in train_data_df['Language_Filename']]
                    )
                aud_train_embeddings = np.vstack(
                    [np.load(filepath) for filepath in train_data_df['Audio_Filename']]
                    )

                


                train_embeddings = np.concatenate((lang_train_embeddings,aud_train_embeddings), axis=-1)
                
                lang_train_mean = lang_train_embeddings.mean(axis=0)
                lang_train_std = lang_train_embeddings.std(axis=0)
                lang_train_embeddings = (lang_train_embeddings - lang_train_mean) / (lang_train_std + 1e-7)
                aud_train_mean = aud_train_embeddings.mean(axis=0)
                aud_train_std = aud_train_embeddings.std(axis=0)
                aud_train_embeddings = (aud_train_embeddings - aud_train_mean) / (aud_train_std + 1e-7)


                train_y_true = train_data_df['dx'].values.astype(int)
                train_y_true_eval = train_data_df[['tkdname','dx']].drop_duplicates()['dx'].values.astype(int)
                
                lang_test_embeddings = np.vstack(
                    [np.load(filepath) for filepath in test_data_df['Language_Filename']]
                    )
                aud_test_embeddings = np.vstack(
                    [np.load(filepath) for filepath in test_data_df['Audio_Filename']]
                    )
                test_embeddings = np.concatenate((lang_test_embeddings,aud_test_embeddings), axis=-1)

                lang_test_embeddings = (lang_test_embeddings - lang_train_mean) / (lang_train_std + 1e-7)
                aud_test_embeddings = (aud_test_embeddings - aud_train_mean) / (aud_train_std + 1e-7)

                test_y_true = test_data_df['dx'].values.astype(int)
                test_y_true_eval = test_data_df[['tkdname','dx']].drop_duplicates()['dx'].values.astype(int)
            
            data_train = {
                'Linguistic': lang_train_embeddings,
                'Acoustic': aud_train_embeddings
                }
            data_test = {
                'Linguistic': lang_test_embeddings,
                'Acoustic': aud_test_embeddings
                }

            ei_predictors = EIPredictors()
            EI = EnsembleIntegration(
                            base_predictors=ei_predictors.base_predictors,
                            k_outer=5,
                            k_inner=5,
                            n_samples=1,
                            sampling_strategy='oversampling',
                            sampling_aggregation=None,
                            n_jobs=-1,
                            random_state=ii,
                            project_name="NLP_CI",
                            model_building=True,
                            )


            for name, modality_data in data_train.items():
                EI.fit_base(modality_data, train_y_true, modality_name=name)
            
            EI.fit_ensemble(ensemble_predictors=ei_predictors.ensemble_predictors)

            class_balance_thresh = (1-(train_data_df['dx'].values.sum()/train_data_df['dx'].values.shape[0]))
            for ensemble, _ in ei_predictors.ensemble_predictors.items():
                temp_df = {}
                temp_df['Outer Fold'] = ii
                temp_df['Ensemble Model'] = ensemble
                outputs = EI.predict(X_dict=data_test, ensemble_model_key=ensemble)
                y_pred = outputs
                pred_score = utils.scores(test_y_true_eval, y_pred)
                for key, val in pred_score.items():
                    temp_df[key] = val
                temp_df['Accuracy'] = accuracy_score(test_y_true_eval,(y_pred>class_balance_thresh).astype(int))
                all_scores.append(temp_df)

            temp_df = {}
            temp_df['Outer Fold'] = ii
            temp_df['Ensemble Model'] = 'Random'
            rand_df = []
            for jj in range(100):
                y_pred = np.random.uniform(size=test_data_df['dx'].values.shape)
                pred_score = utils.scores(test_data_df['dx'].values.astype(int), y_pred)
                pred_score['Accuracy'] = accuracy_score(
                    test_data_df['dx'].values.astype(int),
                    (y_pred>class_balance_thresh).astype(int))
                rand_df.append(pred_score)
            cat_rand_df = pd.DataFrame(rand_df)
            pred_score = cat_rand_df.mean()
            for key, val in pred_score.items():
                temp_df[key] = val
            all_scores.append(temp_df)

        elif training_config['architecture']=='BP':
            if training_config['target'] == "MCI":
                if "Audio_Filename" in train_data_df.columns:
                    modality = 'MULTIMODAL'
                    train_lang_embeddings = np.vstack(
                        [np.load(filepath) for filepath in train_data_df['Language_Filename']]
                        )
                    test_lang_embeddings = np.vstack(
                        [np.load(filepath) for filepath in test_data_df['Language_Filename']]
                        )
                    train_aud_embeddings = np.vstack(
                        [np.load(filepath) for filepath in train_data_df['Audio_Filename']]
                        )
                    test_aud_embeddings = np.vstack(
                        [np.load(filepath) for filepath in test_data_df['Audio_Filename']]
                        )
                    train_embeddings = np.concatenate((train_lang_embeddings,train_aud_embeddings), axis=-1)
                    test_embeddings = np.concatenate((test_lang_embeddings,test_aud_embeddings), axis=-1)
                else: 
                    modality = None   
                    train_embeddings = np.vstack(
                        [np.load(filepath) for filepath in train_data_df['Filename']]
                        )
                    test_embeddings = np.vstack(
                        [np.load(filepath) for filepath in test_data_df['Filename']]
                        )

                train_mean = train_embeddings.mean(axis=0)
                train_std = train_embeddings.std(axis=0)
                train_embeddings = (train_embeddings - train_mean) / (train_std + 1e-7)
                train_y_true = train_data_df['dx'].values.astype(int)
                train_y_true_eval = train_data_df[['tkdname','dx']].drop_duplicates()['dx'].values.astype(int)
                
                test_embeddings = (test_embeddings - train_mean) / (train_std + 1e-7)
                test_y_true = test_data_df['dx'].values.astype(int)
                test_y_true_eval = test_data_df[['tkdname','dx']].drop_duplicates()['dx'].values.astype(int)
        
            data_train = {training_config['modality']: train_embeddings}
            data_test = {training_config['modality']: test_embeddings}

            # base_predictors = BasePredictors()
            base_predictors = models.FastBasePredictors()
            # base_predictors = SigNLTKPredictors()
            all_preds = []
            for predictor_name, base_predictor in base_predictors.base_predictors.items():
                print(f'Fitting {predictor_name}')
                base_predictor.fit(train_embeddings,train_y_true)
                eval_df = utils.get_predictions(
                    test_tkdnames, 
                    test_data_df, 
                    base_predictor, 
                    EI=True, 
                    modality=modality,
                    data_mean=train_mean, 
                    data_std=train_std
                )
                eval_df['Ensemble Model'] = predictor_name
                all_preds.append(eval_df)
                
                pred_score = utils.get_scores(
                    test_tkdnames, 
                    test_data_df, 
                    base_predictor, 
                    EI=True, 
                    modality=modality,
                    data_mean=train_mean, 
                    data_std=train_std)

                temp_df = {}
                temp_df['Outer Fold'] = ii
                temp_df['Ensemble Model'] = predictor_name
                for key, val in pred_score.items():
                    temp_df[key] = val
                all_scores.append(temp_df)

            breakpoint()
            pred_df = pd.concat(all_preds)
            pred_path = training_config['scores_outdir'].replace('reports','predictions')
            os.makedirs(pred_path, exist_ok=True)
            preds_outpath = os.path.join(pred_path,args.model_name+'.csv')
            pred_df.to_csv(preds_outpath)
            breakpoint()

            _, tkdname_indices = utils.average_ei_model_output_per_tkdname(
                test_tkdnames, 
                test_data_df, 
                base_predictor, 
                modality=modality)

            
            temp_df = {}
            temp_df['Outer Fold'] = ii
            temp_df['Ensemble Model'] = 'Random'
            rand_df = []
            class_balance_thresh = (1-(train_data_df['dx'].values.sum()/train_data_df['dx'].values.shape[0]))
            for jj in range(100):
                y_pred = np.random.uniform(size=test_data_df['dx'].values.shape)
                pred_score = utils.scores(test_data_df['dx'].values.astype(int), y_pred)
                pred_score['Accuracy'] = accuracy_score(
                    test_data_df['dx'].values.astype(int),
                    (y_pred>class_balance_thresh).astype(int))
                rand_df.append(pred_score)
            cat_rand_df = pd.DataFrame(rand_df)
            pred_score = cat_rand_df.mean()
            for key, val in pred_score.items():
                temp_df[key] = val
            all_scores.append(temp_df)

        elif training_config['architecture']=='DA4AD':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if training_config['target'] == "MCI":
                train_dataset = DA4ADDataset(dataframe=train_data_df)
                train_loader = DataLoader(
                    train_dataset, 
                    batch_size=32, 
                    shuffle=True,
                    drop_last=True)
                test_dataset = DA4ADDataset(dataframe=test_data_df)
                test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

                
                # Define the model
                model = DA4ADClassifier()
                
                # Define the loss functions and optimizer
                class_weights = compute_class_weight(
                    class_weight='balanced', 
                    classes=np.unique(train_data_df['dx']), 
                    y=train_data_df['dx'].values
                )
                weights_dict = {index: value for index, value in enumerate(class_weights)}
                
                optimizer = torchAdam(model.parameters(), lr=training_config['learning_rate'])
                criterion_mci = nn.BCEWithLogitsLoss()
                criterion_gender = nn.BCEWithLogitsLoss()
                
                # Training loop
                for epoch in range(training_config['n_epochs']):
                    model.train()
                    epoch_loss = 0
                    for inputs, targets in train_loader:
                        # Move data to the appropriate device (e.g., GPU if available)
                        inputs = [x.to(device) for x in inputs]
                        mci_targets, gender_targets = [t.to(device) for t in targets]

                        optimizer.zero_grad()
                        outputs = model(inputs)

                        # Compute loss
                        loss_mci = criterion_mci(outputs[0].squeeze(), mci_targets.float())
                        loss_gender = criterion_gender(outputs[1].squeeze(), gender_targets.float().squeeze())
                        loss = 0.9*loss_mci + 0.1*loss_gender

                        loss.backward()
                        optimizer.step()
                        
                        epoch_loss += loss.item()
                    
                    print(f"Epoch {epoch+1}/{training_config['n_epochs']}, Loss: {epoch_loss/len(train_loader)}")
                    

            model.to('cpu')
            train_scores = utils.get_scores(train_tkdnames, train_data_df, model, pytorch=True)
            test_scores = utils.get_scores(test_tkdnames, test_data_df, model, pytorch=True)

            temp_df = {}
            temp_df['Outer Fold'] = ii
            temp_df['Ensemble Model'] = args.model_name
            for key, val in test_scores.items():
                temp_df[key] = val
            all_scores.append(temp_df)

            temp_df = {}
            temp_df['Outer Fold'] = ii
            temp_df['Ensemble Model'] = 'Random'
            rand_df = []
            class_balance_thresh = (1-(train_data_df['dx'].values.sum()/train_data_df['dx'].values.shape[0]))
            for jj in range(100):
                y_pred = np.random.uniform(size=test_data_df['dx'].values.shape)
                pred_score = utils.scores(test_data_df['dx'].values.astype(int), y_pred)
                pred_score['Accuracy'] = accuracy_score(
                    test_data_df['dx'].values.astype(int),
                    (y_pred>class_balance_thresh).astype(int))
                rand_df.append(pred_score)
            cat_rand_df = pd.DataFrame(rand_df)
            pred_score = cat_rand_df.mean()
            for key, val in pred_score.items():
                temp_df[key] = val
            all_scores.append(temp_df)

            del model 

        elif training_config['architecture']=='SARAWGI':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            train_embeddings = np.vstack(
                    [np.load(filepath) for filepath in train_data_df['Filename']]
                    )
            train_mean = train_embeddings.mean(axis=0)
            train_std = train_embeddings.std(axis=0)
            train_embeddings = (train_embeddings - train_mean) / (train_std + 1e-7) 

            train_pca = PCA(n_components=21)
            train_embeddings = train_pca.fit_transform(train_embeddings)
            train_y_true = train_data_df['dx'].values.astype(int)
            
            val_embeddings = np.vstack(
                [np.load(filepath) for filepath in val_data_df['Filename']]
                )
            val_embeddings = (val_embeddings - train_mean) / (train_std + 1e-7)
            val_embeddings = train_pca.transform(val_embeddings) 
            val_y_true = val_data_df['dx'].values.astype(int)
            
            test_embeddings = np.vstack(
                [np.load(filepath) for filepath in test_data_df['Filename']]
                )
            test_embeddings = (test_embeddings - train_mean) / (train_std + 1e-7)
            test_embeddings = train_pca.transform(test_embeddings) 
            test_y_true = test_data_df['dx'].values.astype(int)

            train_dataset = dataloaders.SarawgiDataset(
                train_embeddings, 
                train_y_true
            )
            train_loader = DataLoader(
                train_dataset, 
                batch_size=8, 
                shuffle=True)
            val_dataset = dataloaders.SarawgiDataset(
                val_embeddings, 
                val_y_true
            )
            val_loader = DataLoader(
                val_dataset, 
                batch_size=len(val_dataset), 
                shuffle=True)
            test_dataset = dataloaders.SarawgiDataset(
                test_embeddings, 
                test_y_true
            )
            test_loader = DataLoader(
                test_dataset, 
                batch_size=len(test_dataset), 
                shuffle=True)

            if training_config['target'] == "MCI":              
                # Define the loss functions and optimizer
                class_weights = compute_class_weight(
                    class_weight='balanced', 
                    classes=np.unique(train_data_df['dx']), 
                    y=train_data_df['dx'].values
                )
                weights_dict = {index: value for index, value in enumerate(class_weights)}
                model = models.SarawgiClassifier()
                optimizer = torchAdam(model.parameters(), lr=1e-2)
                criterion_mci = nn.CrossEntropyLoss()

                # Training loop
                patience = 2001  # Number of epochs to wait for improvement
                best_val_loss = float('inf')
                epochs_without_improvement = 0
                for epoch in range(2000):
                    model.train()
                    epoch_loss = 0
                    for inputs, targets in train_loader:
                        # Move data to the appropriate device (e.g., GPU if available)
                        inputs = torch.stack([x.to(device) for x in inputs]).to(device)
                        mci_targets = torch.tensor([t.to(device) for t in targets]).to(device)
                        activations = torch.nn.functional.relu(model.fc1(inputs))

                        optimizer.zero_grad()
                        outputs = model(inputs)

                        # Compute loss
                        loss_mci = criterion_mci(outputs, mci_targets)
                        l2_penalty = sum(torch.sum(param ** 2) for param in model.parameters())
                        l1_penalty = torch.sum(torch.abs(activations)) + torch.sum(torch.abs(outputs))
                        loss = loss_mci + 1e-2*l2_penalty + 1e-2*l1_penalty

                        loss.backward()
                        optimizer.step()
                        
                        epoch_loss += loss.item()
                    
                    print(f"Epoch {epoch+1}/{training_config['n_epochs']}, Loss: {epoch_loss/len(train_loader)}")

                    model.eval()
                    val_loss = 0
                    with torch.no_grad():
                        for val_inputs, val_targets in val_loader:
                            val_inputs = torch.stack([x.to(device) for x in val_inputs]).to(device)
                            val_mci_targets = torch.tensor([t.to(device) for t in val_targets]).to(device)
                            
                            val_outputs = model(val_inputs)
                            val_loss_mci = criterion_mci(val_outputs, val_mci_targets)
                            val_l2_penalty = sum(torch.sum(param ** 2) for param in model.parameters())
                            val_l1_penalty = torch.sum(torch.abs(activations)) + torch.sum(torch.abs(outputs))
                            val_loss_sum = val_loss_mci + 1e-2*val_l2_penalty + 1e-2*val_l1_penalty
                            
                            val_loss += val_loss_sum.item()

                    # Calculate average validation loss
                    avg_val_loss = val_loss / len(val_loader)
                    print(f"Validation Loss: {avg_val_loss:.4f}")

                    # Early stopping check
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        epochs_without_improvement = 0
                        best_model_state = model.state_dict()
                        print("Validation loss improved, model saved.")
                    else:
                        epochs_without_improvement += 1
                        print(f"No improvement for {epochs_without_improvement} epochs.")

                    # Check if early stopping criterion is met
                    if epochs_without_improvement >= patience:
                        print("Early stopping triggered.")
                        break
                    
            model.load_state_dict(best_model_state)
            model.to('cpu')
            model.eval()
            for test_inputs, test_targets in test_loader:
                test_inputs = torch.stack([x.to('cpu') for x in test_inputs]).to('cpu')
                test_mci_targets = torch.tensor([t.to('cpu') for t in test_targets]).to('cpu')
                
                test_outputs = model(test_inputs).cpu().detach().numpy()
                
            y_pred = test_outputs[:,1]
            test_y_true = test_mci_targets.cpu().detach().numpy()
            test_scores = utils.scores(test_y_true, y_pred)
            

            temp_df = {}
            temp_df['Outer Fold'] = ii
            temp_df['Ensemble Model'] = args.model_name
            for key, val in test_scores.items():
                temp_df[key] = val
            all_scores.append(temp_df)

            temp_df = {}
            temp_df['Outer Fold'] = ii
            temp_df['Ensemble Model'] = 'Random'
            rand_df = []
            class_balance_thresh = (1-(train_data_df['dx'].values.sum()/train_data_df['dx'].values.shape[0]))
            for jj in range(100):
                y_pred = np.random.uniform(size=test_data_df['dx'].values.shape)
                pred_score = utils.scores(test_data_df['dx'].values.astype(int), y_pred)
                pred_score['Accuracy'] = accuracy_score(
                    test_data_df['dx'].values.astype(int),
                    (y_pred>class_balance_thresh).astype(int))
                rand_df.append(pred_score)
            cat_rand_df = pd.DataFrame(rand_df)
            pred_score = cat_rand_df.mean()
            for key, val in pred_score.items():
                temp_df[key] = val
            all_scores.append(temp_df)

            del model

        elif training_config['architecture']=='Ying':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if training_config['target'] == "MCI":
                train_iso_embeddings = np.vstack(
                    [np.load(filepath) for filepath in train_data_df['Iso_Filename']]
                    )
                test_iso_embeddings = np.vstack(
                    [np.load(filepath) for filepath in test_data_df['Iso_Filename']]
                    )
                train_dataset = dataloaders.YingDataset(dataframe=train_data_df)
                bert_train_loader = DataLoader(
                    train_dataset, 
                    batch_size=16, #should be 16 
                    shuffle=True,
                    collate_fn=utils.ying_collate_fn
                    )
                w2v2_train_loader = DataLoader(
                    train_dataset, 
                    batch_size=8, #should be 8 
                    shuffle=True,
                    collate_fn=utils.ying_collate_fn
                    )
                full_train_dataloader = DataLoader(
                    train_dataset, 
                    batch_size=len(train_dataset),  
                    shuffle=False,
                    collate_fn=utils.ying_collate_fn
                    )
                test_dataset = dataloaders.YingDataset(dataframe=test_data_df)
                full_test_dataloader = DataLoader(
                    test_dataset, 
                    batch_size=len(test_dataset), 
                    shuffle=False,
                    collate_fn=utils.ying_collate_fn
                    )

                # Define the model
                base_predictors = models.YingPredictors()
                wav_model = models.YingWavClassifier()
                wav_model.to(device)
                bert_model = models.YingBertClassifier()
                bert_model.to(device)
                
                # Define the loss functions and optimizer
                class_weights = compute_class_weight(
                    class_weight='balanced', 
                    classes=np.unique(train_data_df['dx']), 
                    y=train_data_df['dx'].values
                )
                weights_dict = {index: value for index, value in enumerate(class_weights)}
                
                bert_optimizer = torchAdamW(
                    bert_model.parameters(), 
                    lr=3e-5)

                w2v2_optimizer = torchAdamW(
                    wav_model.parameters(), 
                    lr=1e-5)
                criterion_mci = nn.BCEWithLogitsLoss()
                
                # Finetune BERT
                for epoch in range(3): #should be 3 epochs
                    bert_model.train()
                    epoch_loss = 0
                    for inputs, targets in bert_train_loader:
                        # Move data to the appropriate device (e.g., GPU if available)
                        X_lang, _ = inputs
                        X_lang = [x.to(device) for x in X_lang]
                        input_ids = torch.vstack([item.input_ids for item in X_lang]).to(device)
                        attention_masks = torch.vstack([item.attention_mask for item in X_lang]).to(device)
                        
                        mci_targets = torch.tensor([t.to(device) for t in targets]).to(device)

                        bert_optimizer.zero_grad()
                        outputs = bert_model(input_ids, attention_masks)

                        # Compute loss
                        loss_mci = criterion_mci(outputs.squeeze(), mci_targets.float())

                        loss_mci.backward()
                        bert_optimizer.step()
                        
                        epoch_loss += loss_mci.item()

                        
                    print(f"Epoch {epoch+1}/{3}, Loss: {epoch_loss/len(bert_train_loader)}")
                      
                    
                # Freeze BERT model parameters
                for param in bert_model.parameters():
                    param.requires_grad = False

                # Finetune W2V2
                for epoch in range(32): #should be 32 epochs
                    wav_model.train()
                    epoch_loss = 0
                    for inputs, targets in w2v2_train_loader:
                        # Move data to the appropriate device (e.g., GPU if available)
                        _, X_aud = inputs                       
                        audio = torch.stack([x.to(device) for x in X_aud]).to(device)
                        
                        mci_targets = torch.tensor([t.to(device) for t in targets]).to(device)

                        w2v2_optimizer.zero_grad()
                        outputs = wav_model(audio)

                        # Compute loss
                        loss_mci = criterion_mci(outputs.squeeze(), mci_targets.float())

                        loss_mci.backward()
                        w2v2_optimizer.step()
                        
                        epoch_loss += loss_mci.item()

                        
                    print(f"Epoch {epoch+1}/{32}, Loss: {epoch_loss/len(w2v2_train_loader)}")
                    
                    
                # Freeze W2V2 model parameters
                for param in wav_model.parameters():
                    param.requires_grad = False
                
                print('Models have been finetuned')

                bert_model.eval()
                wav_model.eval()

                bert_flag = 0.0
                w2v2_flag = 1.0
                iso_flag = 0.0

                #Get finetuned embeddings and fit SVM
                for inputs, targets in full_train_dataloader:
                    # Move data to the appropriate device (e.g., GPU if available)
                    X_lang, X_aud = inputs 
                    
                    X_lang = [x.to(device) for x in X_lang]
                    input_ids = torch.vstack([item.input_ids for item in X_lang]).to(device)
                    attention_masks = torch.vstack([item.attention_mask for item in X_lang]).to(device)

                    audio = torch.stack([x.to(device) for x in X_aud]).to(device)
                    
                    mci_targets = torch.tensor([t.to(device) for t in targets]).to(device)
                    train_y_true = mci_targets.cpu().detach().numpy()

                    bert_embeddings = bert_flag * bert_model.extract_embeding(input_ids, attention_masks).cpu().detach().numpy()
                    w2v2_embeddings = w2v2_flag * wav_model.extract_embeding(audio).cpu().detach().numpy()
                    train_iso_embeddings = iso_flag * train_iso_embeddings

                    train_embeddings = np.concatenate((bert_embeddings,w2v2_embeddings,train_iso_embeddings),axis=1)
                    train_mean = train_embeddings.mean(axis=0)
                    train_std = train_embeddings.std(axis=0)
                    train_embeddings = (train_embeddings - train_mean) / (train_std + 1e-7) 

                    for predictor_name, base_predictor in base_predictors.base_predictors.items():
                        print(f'Fitting {predictor_name}')
                        base_predictor.fit(train_embeddings,train_y_true)
                        print('Base predictor has been fit')
                
                #Test on testing split
                for inputs, targets in full_test_dataloader:
                    # Move data to the appropriate device (e.g., GPU if available)
                    X_lang, X_aud = inputs 
                    
                    X_lang = [x.to(device) for x in X_lang]
                    input_ids = torch.vstack([item.input_ids for item in X_lang]).to(device)
                    attention_masks = torch.vstack([item.attention_mask for item in X_lang]).to(device)

                    audio = torch.stack([x.to(device) for x in X_aud]).to(device)
                    
                    mci_targets = torch.tensor([t.to(device) for t in targets]).to(device)
                    test_y_true = mci_targets.cpu().detach().numpy()

                    bert_embeddings = bert_flag * bert_model.extract_embeding(input_ids, attention_masks).cpu().detach().numpy()
                    w2v2_embeddings = w2v2_flag * wav_model.extract_embeding(audio).cpu().detach().numpy()
                    test_iso_embeddings = iso_flag * test_iso_embeddings

                    test_embeddings = np.concatenate((bert_embeddings,w2v2_embeddings, test_iso_embeddings),axis=1)
                    test_embeddings = (test_embeddings - train_mean) / (train_std + 1e-7)

                    for predictor_name, base_predictor in base_predictors.base_predictors.items():
                        y_pred = base_predictor.predict_proba(test_embeddings)
                        y_pred = y_pred[:,1]
                        print('Test Proba Predicted')
                    
                    test_scores = utils.scores(test_y_true, y_pred)
                    temp_df = {}
                    temp_df['Outer Fold'] = ii
                    temp_df['Ensemble Model'] = args.model_name
                    for key, val in test_scores.items():
                        temp_df[key] = val
                    all_scores.append(temp_df)

                    temp_df = {}
                    temp_df['Outer Fold'] = ii
                    temp_df['Ensemble Model'] = 'Random'
                    rand_df = []
                    class_balance_thresh = (1-(train_data_df['dx'].values.sum()/train_data_df['dx'].values.shape[0]))
                    for jj in range(100):
                        y_pred = np.random.uniform(size=test_data_df['dx'].values.shape)
                        pred_score = utils.scores(test_data_df['dx'].values.astype(int), y_pred)
                        pred_score['Accuracy'] = accuracy_score(
                            test_data_df['dx'].values.astype(int),
                            (y_pred>class_balance_thresh).astype(int))
                        rand_df.append(pred_score)
                    cat_rand_df = pd.DataFrame(rand_df)
                    pred_score = cat_rand_df.mean()
                    for key, val in pred_score.items():
                        temp_df[key] = val
                    all_scores.append(temp_df)           

                del bert_model
                del wav_model 


        elif training_config['architecture']=='Wavbert':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if training_config['target'] == "MCI":
                train_dataset = dataloaders.YingDataset(dataframe=train_data_df)
                bert_train_loader = DataLoader(
                    train_dataset, 
                    batch_size=8,  
                    shuffle=True,
                    collate_fn=utils.ying_collate_fn,
                    drop_last=True
                    )
                
                test_dataset = dataloaders.YingDataset(dataframe=test_data_df)
                full_test_dataloader = DataLoader(
                    test_dataset, 
                    batch_size=len(test_dataset), 
                    shuffle=False,
                    collate_fn=utils.ying_collate_fn
                    )

                # Define the model
                base_predictors = models.YingPredictors()
                bert_model = models.YingBertClassifier()
                bert_model.to(device)
                
                # Define the loss functions and optimizer
                class_weights = compute_class_weight(
                    class_weight='balanced', 
                    classes=np.unique(train_data_df['dx']), 
                    y=train_data_df['dx'].values
                )
                weights_dict = {index: value for index, value in enumerate(class_weights)}
                
                bert_optimizer = torchAdamW(
                    bert_model.parameters(), 
                    lr=1e-6)

                criterion_mci = nn.BCEWithLogitsLoss()
                
                # Finetune BERT
                for epoch in range(2000): #should be 3 epochs
                    bert_model.train()
                    epoch_loss = 0
                    for inputs, targets in bert_train_loader:

                        # Move data to the appropriate device (e.g., GPU if available)
                        X_lang, _ = inputs
                        X_lang = [x.to(device) for x in X_lang]
                        input_ids = torch.vstack([item.input_ids for item in X_lang]).to(device)
                        attention_masks = torch.vstack([item.attention_mask for item in X_lang]).to(device)
                        
                        mci_targets = torch.tensor([t.to(device) for t in targets]).to(device)

                        bert_optimizer.zero_grad()
                        outputs = bert_model(input_ids, attention_masks)

                        # Compute loss
                        loss_mci = criterion_mci(outputs.squeeze(), mci_targets.float())

                        loss_mci.backward()
                        bert_optimizer.step()
                        
                        epoch_loss += loss_mci.item()

                        
                    print(f"Epoch {epoch+1}/{2000}, Loss: {epoch_loss/len(bert_train_loader)}")

                    if epoch_loss/len(bert_train_loader) < 1e-6:
                        break
                      
                    
                # Freeze BERT model parameters
                for param in bert_model.parameters():
                    param.requires_grad = False

                

                bert_model.eval()

                # #Get finetuned embeddings and fit SVM
                # for inputs, targets in full_train_dataloader:
                #     # Move data to the appropriate device (e.g., GPU if available)
                #     X_lang, X_aud = inputs 
                    
                #     X_lang = [x.to(device) for x in X_lang]
                #     input_ids = torch.vstack([item.input_ids for item in X_lang]).to(device)
                #     attention_masks = torch.vstack([item.attention_mask for item in X_lang]).to(device)

                #     audio = torch.stack([x.to(device) for x in X_aud]).to(device)
                    
                #     mci_targets = torch.tensor([t.to(device) for t in targets]).to(device)
                #     train_y_true = mci_targets.cpu().detach().numpy()

                #     bert_embeddings = bert_model.extract_embeding(input_ids, attention_masks).cpu().detach().numpy()
                #     w2v2_embeddings = wav_model.extract_embeding(audio).cpu().detach().numpy()

                #     train_embeddings = np.concatenate((bert_embeddings,w2v2_embeddings,train_iso_embeddings),axis=1)
                #     train_mean = train_embeddings.mean(axis=0)
                #     train_std = train_embeddings.std(axis=0)
                #     train_embeddings = (train_embeddings - train_mean) / (train_std + 1e-7) 

                #     for predictor_name, base_predictor in base_predictors.base_predictors.items():
                #         print(f'Fitting {predictor_name}')
                #         base_predictor.fit(train_embeddings,train_y_true)
                #         print('Base predictor has been fit')
                
                #Test on testing split
                for inputs, targets in full_test_dataloader:
                    # Move data to the appropriate device (e.g., GPU if available)
                    X_lang, X_aud = inputs 
                    
                    X_lang = [x.to(device) for x in X_lang]
                    input_ids = torch.vstack([item.input_ids for item in X_lang]).to(device)
                    attention_masks = torch.vstack([item.attention_mask for item in X_lang]).to(device)

                    audio = torch.stack([x.to(device) for x in X_aud]).to(device)
                    
                    mci_targets = torch.tensor([t.to(device) for t in targets]).to(device)
                    test_y_true = mci_targets.cpu().detach().numpy()

                    y_pred = torch.sigmoid(bert_model(input_ids, attention_masks)).cpu().detach().numpy()
                    
                    test_scores = utils.scores(test_y_true, y_pred)
                    temp_df = {}
                    temp_df['Outer Fold'] = ii
                    temp_df['Ensemble Model'] = args.model_name
                    for key, val in test_scores.items():
                        temp_df[key] = val
                    all_scores.append(temp_df)

                    temp_df = {}
                    temp_df['Outer Fold'] = ii
                    temp_df['Ensemble Model'] = 'Random'
                    rand_df = []
                    class_balance_thresh = (1-(train_data_df['dx'].values.sum()/train_data_df['dx'].values.shape[0]))
                    for jj in range(100):
                        y_pred = np.random.uniform(size=test_data_df['dx'].values.shape)
                        pred_score = utils.scores(test_data_df['dx'].values.astype(int), y_pred)
                        pred_score['Accuracy'] = accuracy_score(
                            test_data_df['dx'].values.astype(int),
                            (y_pred>class_balance_thresh).astype(int))
                        rand_df.append(pred_score)
                    cat_rand_df = pd.DataFrame(rand_df)
                    pred_score = cat_rand_df.mean()
                    for key, val in pred_score.items():
                        temp_df[key] = val
                    all_scores.append(temp_df)           

                del bert_model


        elif training_config['architecture']=='EASIYing':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if training_config['target'] == "MCI":
                train_iso_embeddings = np.vstack(
                    [np.load(filepath) for filepath in train_data_df['Iso_Filename']]
                    )
                test_iso_embeddings = np.vstack(
                    [np.load(filepath) for filepath in test_data_df['Iso_Filename']]
                    )
                train_dataset = dataloaders.YingDataset(dataframe=train_data_df)
                bert_train_loader = DataLoader(
                    train_dataset, 
                    batch_size=16, #should be 16 
                    shuffle=True,
                    collate_fn=utils.ying_collate_fn
                    )
                w2v2_train_loader = DataLoader(
                    train_dataset, 
                    batch_size=8, #should be 8 
                    shuffle=True,
                    collate_fn=utils.ying_collate_fn
                    )
                full_train_dataloader = DataLoader(
                    train_dataset, 
                    batch_size=len(train_dataset),  
                    shuffle=False,
                    collate_fn=utils.ying_collate_fn
                    )
                test_dataset = dataloaders.YingDataset(dataframe=test_data_df)
                full_test_dataloader = DataLoader(
                    test_dataset, 
                    batch_size=len(test_dataset), 
                    shuffle=False,
                    collate_fn=utils.ying_collate_fn
                    )

                # Define the model
                base_predictors = models.YingPredictors()
                wav_model = models.YingWavClassifier()
                wav_model.to(device)
                bert_model = models.YingBertClassifier()
                bert_model.to(device)
                
                # Define the loss functions and optimizer
                class_weights = compute_class_weight(
                    class_weight='balanced', 
                    classes=np.unique(train_data_df['dx']), 
                    y=train_data_df['dx'].values
                )
                weights_dict = {index: value for index, value in enumerate(class_weights)}
                
                bert_optimizer = torchAdamW(
                    bert_model.parameters(), 
                    lr=3e-5)

                w2v2_optimizer = torchAdamW(
                    wav_model.parameters(), 
                    lr=1e-5)
                criterion_mci = nn.BCEWithLogitsLoss()
                
                # Finetune BERT
                # for epoch in range(3): #should be 3 epochs
                #     bert_model.train()
                #     epoch_loss = 0
                #     for inputs, targets in bert_train_loader:
                #         # Move data to the appropriate device (e.g., GPU if available)
                #         X_lang, _ = inputs
                #         X_lang = [x.to(device) for x in X_lang]
                #         input_ids = torch.vstack([item.input_ids for item in X_lang]).to(device)
                #         attention_masks = torch.vstack([item.attention_mask for item in X_lang]).to(device)
                        
                #         mci_targets = torch.tensor([t.to(device) for t in targets]).to(device)

                #         bert_optimizer.zero_grad()
                #         outputs = bert_model(input_ids, attention_masks)

                #         # Compute loss
                #         loss_mci = criterion_mci(outputs.squeeze(), mci_targets.float())

                #         loss_mci.backward()
                #         bert_optimizer.step()
                        
                #         epoch_loss += loss_mci.item()

                        
                #     print(f"Epoch {epoch+1}/{3}, Loss: {epoch_loss/len(bert_train_loader)}")
                      
                    
                # Freeze BERT model parameters
                for param in bert_model.parameters():
                    param.requires_grad = False

                # Finetune W2V2
                # for epoch in range(32): #should be 32 epochs
                #     wav_model.train()
                #     epoch_loss = 0
                #     for inputs, targets in w2v2_train_loader:
                #         # Move data to the appropriate device (e.g., GPU if available)
                #         _, X_aud = inputs                       
                #         audio = torch.stack([x.to(device) for x in X_aud]).to(device)
                        
                #         mci_targets = torch.tensor([t.to(device) for t in targets]).to(device)

                #         w2v2_optimizer.zero_grad()
                #         outputs = wav_model(audio)

                #         # Compute loss
                #         loss_mci = criterion_mci(outputs.squeeze(), mci_targets.float())

                #         loss_mci.backward()
                #         w2v2_optimizer.step()
                        
                #         epoch_loss += loss_mci.item()

                        
                #     print(f"Epoch {epoch+1}/{32}, Loss: {epoch_loss/len(w2v2_train_loader)}")
                    
                    
                # Freeze W2V2 model parameters
                for param in wav_model.parameters():
                    param.requires_grad = False
                
                print('Models have been finetuned')

                bert_model.eval()
                wav_model.eval()

                bert_flag = 0.0
                w2v2_flag = 0.0
                iso_flag = 1.0

                #Get finetuned embeddings and fit SVM
                for inputs, targets in full_train_dataloader:
                    # Move data to the appropriate device (e.g., GPU if available)
                    X_lang, X_aud = inputs 
                    
                    X_lang = [x.to(device) for x in X_lang]
                    input_ids = torch.vstack([item.input_ids for item in X_lang]).to(device)
                    attention_masks = torch.vstack([item.attention_mask for item in X_lang]).to(device)

                    audio = torch.stack([x.to(device) for x in X_aud]).to(device)
                    
                    mci_targets = torch.tensor([t.to(device) for t in targets]).to(device)
                    train_y_true = mci_targets.cpu().detach().numpy()

                    bert_embeddings = bert_flag * bert_model(input_ids, attention_masks).cpu().detach().numpy()
                    w2v2_embeddings = w2v2_flag * wav_model(audio).cpu().detach().numpy()
                    train_iso_embeddings = iso_flag * train_iso_embeddings

                    train_embeddings = np.concatenate((bert_embeddings,w2v2_embeddings,train_iso_embeddings),axis=1)
                    train_mean = train_embeddings.mean(axis=0)
                    train_std = train_embeddings.std(axis=0)
                    train_embeddings = (train_embeddings - train_mean) / (train_std + 1e-7) 

                    for predictor_name, base_predictor in base_predictors.base_predictors.items():
                        print(f'Fitting {predictor_name}')
                        base_predictor.fit(train_embeddings,train_y_true)
                        print('Base predictor has been fit')
                
                #Test on testing split
                for inputs, targets in full_test_dataloader:
                    # Move data to the appropriate device (e.g., GPU if available)
                    X_lang, X_aud = inputs 
                    
                    X_lang = [x.to(device) for x in X_lang]
                    input_ids = torch.vstack([item.input_ids for item in X_lang]).to(device)
                    attention_masks = torch.vstack([item.attention_mask for item in X_lang]).to(device)

                    audio = torch.stack([x.to(device) for x in X_aud]).to(device)
                    
                    mci_targets = torch.tensor([t.to(device) for t in targets]).to(device)
                    test_y_true = mci_targets.cpu().detach().numpy()

                    bert_embeddings = bert_flag * bert_model(input_ids, attention_masks).cpu().detach().numpy()
                    w2v2_embeddings = w2v2_flag * wav_model(audio).cpu().detach().numpy()
                    test_iso_embeddings = iso_flag * test_iso_embeddings

                    test_embeddings = np.concatenate((bert_embeddings,w2v2_embeddings, test_iso_embeddings),axis=1)
                    test_embeddings = (test_embeddings - train_mean) / (train_std + 1e-7)

                    for predictor_name, base_predictor in base_predictors.base_predictors.items():
                        y_pred = base_predictor.predict_proba(test_embeddings)
                        y_pred = y_pred[:,1]
                        print('Test Proba Predicted')
                    
                    test_scores = utils.scores(test_y_true, y_pred)
                    temp_df = {}
                    temp_df['Outer Fold'] = ii
                    temp_df['Ensemble Model'] = args.model_name
                    for key, val in test_scores.items():
                        temp_df[key] = val
                    all_scores.append(temp_df)

                    temp_df = {}
                    temp_df['Outer Fold'] = ii
                    temp_df['Ensemble Model'] = 'Random'
                    rand_df = []
                    class_balance_thresh = (1-(train_data_df['dx'].values.sum()/train_data_df['dx'].values.shape[0]))
                    for jj in range(100):
                        y_pred = np.random.uniform(size=test_data_df['dx'].values.shape)
                        pred_score = utils.scores(test_data_df['dx'].values.astype(int), y_pred)
                        pred_score['Accuracy'] = accuracy_score(
                            test_data_df['dx'].values.astype(int),
                            (y_pred>class_balance_thresh).astype(int))
                        rand_df.append(pred_score)
                    cat_rand_df = pd.DataFrame(rand_df)
                    pred_score = cat_rand_df.mean()
                    for key, val in pred_score.items():
                        temp_df[key] = val
                    all_scores.append(temp_df)           

                del bert_model
                del wav_model

        elif training_config['architecture']=='MISA':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            if training_config['target'] == "MCI":
                train_dataset = MISADataset(dataframe=train_data_df)
                train_loader = DataLoader(
                    train_dataset, 
                    batch_size=training_config['batch_size'], 
                    shuffle=True, 
                    drop_last=True,
                    collate_fn = utils.misa_collate_fn)
                val_dataset = MISADataset(dataframe=val_data_df)
                val_loader = DataLoader(
                    val_dataset, 
                    batch_size=len(val_dataset),
                    shuffle=True, 
                    collate_fn = utils.misa_collate_fn)
                test_dataset = MISADataset(dataframe=test_data_df)
                test_loader = DataLoader(
                    test_dataset, 
                    batch_size=len(test_dataset),
                    shuffle=True, 
                    collate_fn = utils.misa_collate_fn)
                
                # Define the model
                sample_lang = np.load(train_data_df['Language_Filename'][0])
                sample_aud = np.load(train_data_df['Audio_Filename'][0])
                model = MISAClassifier(
                    lang_dim = sample_lang.shape[1],
                    aud_dim = sample_aud.shape[1]
                )
                
                # Define the loss functions and optimizer
                class_weights = compute_class_weight(
                    class_weight='balanced', 
                    classes=np.unique(train_data_df['dx']), 
                    y=train_data_df['dx'].values
                )
                weights_dict = {index: value for index, value in enumerate(class_weights)}
                
                optimizer = torchAdam(model.parameters(), lr=training_config['learning_rate'])
                criterion_mci = nn.BCEWithLogitsLoss()
                criterion_mse = utils.MISAMSE()
                criterion_diff = utils.MISADiffLoss()
                criterion_cmd = utils.MISACMD()
                
                # Training loop
                best_val_loss = float('inf')
                patience = 30
                for epoch in range(training_config['n_epochs']):
                    model.train()
                    epoch_loss = 0
                    for batch in train_loader:
                        # Unpack the batch
                        [X_lang, X_aud_padded], mci_targets, [X_lang_lengths, X_aud_lengths] = batch

                        # Move data to the appropriate device (e.g., GPU if available)
                        X_lang, X_aud_padded = X_lang.to(device), X_aud_padded.to(device)
                        X_aud_lengths = X_aud_lengths.to('cpu')
                        mci_targets = mci_targets.to(device)

                        # Pack the padded sequences
                        X_aud_packed = nn.utils.rnn.pack_padded_sequence(X_aud_padded, X_aud_lengths.cpu(), batch_first=True, enforce_sorted=False).to(device)

                        # Zero the gradients
                        optimizer.zero_grad()

                        # Prepare inputs for the model
                        inputs = [X_lang, X_aud_packed]
                        
                        # Forward pass
                        outputs = model(inputs)
                        #Classification loss
                        loss_mci = criterion_mci(outputs['mci_output'].squeeze(), mci_targets.float())
                        #Reconstruction loss
                        loss_recon = 0.5*criterion_mse(outputs['lang_reconstruction'],outputs['lang_proj'])
                        loss_recon += 0.5*criterion_mse(outputs['aud_reconstruction'],outputs['aud_proj'])
                        #Similarity (CMD) loss
                        loss_sim = criterion_cmd(outputs['lang_public'],outputs['aud_public'],5)
                        #Difference loss
                        loss_diff = 0.33*criterion_diff(outputs['lang_public'],outputs['lang_private'])
                        loss_diff += 0.33*criterion_diff(outputs['aud_public'],outputs['aud_private'])
                        loss_diff += 0.33*criterion_diff(outputs['lang_private'],outputs['aud_private'])

                        loss = loss_mci + 0.7*loss_sim + loss_diff + loss_recon 
                        loss.backward()
                        optimizer.step()
                        
                        epoch_loss += loss.item()
                    print(f"Epoch {epoch+1}/{training_config['n_epochs']}, Loss: {epoch_loss/len(train_loader)}")
                    print(loss_mci)

                    model.eval()
                    val_losses = []
                    for val_batch in val_loader:
                        # Unpack the val_batch
                        [X_lang, X_aud_padded], val_mci_targets, [X_lang_lengths, X_aud_lengths] = val_batch

                        # Move data to the appropriate device (e.g., GPU if available)
                        X_lang, X_aud_padded = X_lang.to(device), X_aud_padded.to(device)
                        X_aud_lengths = X_aud_lengths.to('cpu')
                        val_mci_targets = val_mci_targets.to(device)

                        # Pack the padded sequences
                        X_aud_packed = nn.utils.rnn.pack_padded_sequence(X_aud_padded, X_aud_lengths, batch_first=True, enforce_sorted=False).to(device)
                        val_inputs = [X_lang, X_aud_packed]

                        with torch.no_grad():
                            val_outputs = model(val_inputs)
                            #Classification loss
                            loss_mci = criterion_mci(val_outputs['mci_output'].squeeze(), val_mci_targets.float())
                            #Reconstruction loss
                            loss_recon = 0.5*criterion_mse(val_outputs['lang_reconstruction'],val_outputs['lang_proj'])
                            loss_recon += 0.5*criterion_mse(val_outputs['aud_reconstruction'],val_outputs['aud_proj'])
                            #Similarity (CMD) loss
                            loss_sim = criterion_cmd(val_outputs['lang_public'],val_outputs['aud_public'],5)
                            #Difference loss
                            loss_diff = 0.33*criterion_diff(val_outputs['lang_public'],val_outputs['lang_private'])
                            loss_diff += 0.33*criterion_diff(val_outputs['aud_public'],val_outputs['aud_private'])
                            loss_diff += 0.33*criterion_diff(val_outputs['lang_private'],val_outputs['aud_private'])

                            loss = loss_mci + 0.7*loss_sim + loss_diff + loss_recon
                            val_losses.append(loss_mci.item())
                    
                    avg_val_loss = np.mean(val_losses)
                    print(f'Validation Loss: {avg_val_loss:.4f}')
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        best_model = model.state_dict()
                        early_stop_counter = 0
                    else:
                        early_stop_counter += 1
                    
                    if early_stop_counter >= patience:
                        print(f'Early stopping after epoch {epoch+1} with validation loss {best_val_loss:.4f}')
                        break
                model.load_state_dict(best_model)
                        
            model.to('cpu')
            train_scores = utils.get_scores(train_tkdnames, train_data_df, model, pytorch=True, modality='MISA')
            test_scores = utils.get_scores(test_tkdnames, test_data_df, model, pytorch=True, modality='MISA')

            temp_df = {}
            temp_df['Outer Fold'] = ii
            temp_df['Ensemble Model'] = args.model_name
            for key, val in test_scores.items():
                temp_df[key] = val
            all_scores.append(temp_df)

            temp_df = {}
            temp_df['Outer Fold'] = ii
            temp_df['Ensemble Model'] = 'Random'
            rand_df = []
            class_balance_thresh = (1-(train_data_df['dx'].values.sum()/train_data_df['dx'].values.shape[0]))
            for jj in range(100):
                y_pred = np.random.uniform(size=test_data_df['dx'].values.shape)
                pred_score = utils.scores(test_data_df['dx'].values.astype(int), y_pred)
                pred_score['Accuracy'] = accuracy_score(
                    test_data_df['dx'].values.astype(int),
                    (y_pred>class_balance_thresh).astype(int))
                rand_df.append(pred_score)
            cat_rand_df = pd.DataFrame(rand_df)
            pred_score = cat_rand_df.mean()
            for key, val in pred_score.items():
                temp_df[key] = val
            all_scores.append(temp_df)

            del model 

        elif training_config['architecture']=='MFN':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            if training_config['modality'] == "MFN":
                train_dataset = MFNDataset(dataframe=train_data_df)
                train_loader = DataLoader(
                    train_dataset, 
                    batch_size=training_config['batch_size'], 
                    shuffle=True, 
                    drop_last=True,
                    collate_fn = utils.mfn_collate_fn)
                val_dataset = MFNDataset(dataframe=val_data_df)
                val_loader = DataLoader(
                    val_dataset, 
                    batch_size=len(val_dataset),
                    shuffle=True, 
                    collate_fn = utils.mfn_collate_fn)
                test_dataset = MFNDataset(dataframe=test_data_df)
                test_loader = DataLoader(
                    test_dataset, 
                    batch_size=len(test_dataset),
                    shuffle=True, 
                    collate_fn = utils.mfn_collate_fn)
                
                # Define the model
                model = MFNClassifier()
            elif training_config['modality'] == "MFN EASI-COG":
                train_dataset = MFNDataset(dataframe=train_data_df)
                train_loader = DataLoader(
                    train_dataset, 
                    batch_size=training_config['batch_size'], 
                    shuffle=True, 
                    drop_last=True,
                    collate_fn = utils.mfn_collate_fn)
                val_dataset = MFNDataset(dataframe=val_data_df)
                val_loader = DataLoader(
                    val_dataset, 
                    batch_size=len(val_dataset),
                    shuffle=True, 
                    collate_fn = utils.mfn_collate_fn)
                test_dataset = MFNDataset(dataframe=test_data_df)
                test_loader = DataLoader(
                    test_dataset, 
                    batch_size=len(test_dataset),
                    shuffle=True, 
                    collate_fn = utils.mfn_collate_fn)
                
                # Define the model
                model = models.EASICOGMFNClassifier()
            
            # Define the loss functions and optimizer
            class_weights = compute_class_weight(
                class_weight='balanced', 
                classes=np.unique(train_data_df['dx']), 
                y=train_data_df['dx'].values
            )
            weights_dict = {index: value for index, value in enumerate(class_weights)}
            
            optimizer = torchAdam(model.parameters(), lr=training_config['learning_rate'])
            criterion_mci = nn.BCEWithLogitsLoss()
            criterion_mse = utils.MISAMSE()
            criterion_diff = utils.MISADiffLoss()
            criterion_cmd = utils.MISACMD()
            
            # Training loop
            best_val_loss = float('inf')
            patience = 30
            for epoch in range(training_config['n_epochs']):
                model.train()
                epoch_loss = 0
                for batch in train_loader:
                    # Unpack the batch
                    [X_lang_padded, X_aud_padded], mci_targets, [X_lang_lengths, X_aud_lengths] = batch

                    # Move data to the appropriate device (e.g., GPU if available)
                    X_lang_padded, X_aud_padded = X_lang_padded.to(device), X_aud_padded.to(device)
                    X_lang_lengths, X_aud_lengths = X_lang_lengths.to('cpu'), X_aud_lengths.to('cpu')
                    mci_targets = mci_targets.to(device)

                    # Zero the gradients
                    optimizer.zero_grad()

                    # Prepare inputs for the model
                    inputs = [X_lang_padded, X_aud_padded]
                    lengths = [X_lang_lengths, X_aud_lengths]

                    #Forward pass
                    outputs = model([inputs, lengths])

                    #Classification loss
                    loss = criterion_mci(outputs.squeeze(), mci_targets.float())
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                print(f"Epoch {epoch+1}/{training_config['n_epochs']}, Loss: {epoch_loss/len(train_loader)}")

                model.eval()
                val_losses = []
                for val_batch in val_loader:
                    # Unpack the val_batch
                    [X_lang, X_aud_padded], val_mci_targets, [X_lang_lengths, X_aud_lengths] = val_batch

                    # Move data to the appropriate device (e.g., GPU if available)
                    X_lang_padded, X_aud_padded = X_lang.to(device), X_aud_padded.to(device)
                    X_lang_lengths, X_aud_lengths = X_lang_lengths.to('cpu'), X_aud_lengths.to('cpu')
                    val_mci_targets = val_mci_targets.to(device)

                    # Pack the padded sequences
                    val_inputs = [X_lang_padded, X_aud_padded]
                    val_lengths = [X_lang_lengths, X_aud_lengths]

                    with torch.no_grad():
                        val_outputs = model([val_inputs, val_lengths])
                        #Classification loss
                        loss = criterion_mci(val_outputs.squeeze(), val_mci_targets.float())
                        val_losses.append(loss.item())
                
                avg_val_loss = np.mean(val_losses)
                print(f'Validation Loss: {avg_val_loss:.4f}')
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_model = model.state_dict()
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                
                if early_stop_counter >= patience:
                    print(f'Early stopping after epoch {epoch+1} with validation loss {best_val_loss:.4f}')
                    break
            model.load_state_dict(best_model)
                        
            model.to('cpu')
            model.device='cpu'
            train_scores = utils.get_scores(train_tkdnames, train_data_df, model, pytorch=True, modality='MFN')
            test_scores = utils.get_scores(test_tkdnames, test_data_df, model, pytorch=True, modality='MFN')

            temp_df = {}
            temp_df['Outer Fold'] = ii
            temp_df['Ensemble Model'] = args.model_name
            for key, val in test_scores.items():
                temp_df[key] = val
            all_scores.append(temp_df)

            temp_df = {}
            temp_df['Outer Fold'] = ii
            temp_df['Ensemble Model'] = 'Random'
            rand_df = []
            class_balance_thresh = (1-(train_data_df['dx'].values.sum()/train_data_df['dx'].values.shape[0]))
            for jj in range(100):
                y_pred = np.random.uniform(size=test_data_df['dx'].values.shape)
                pred_score = utils.scores(test_data_df['dx'].values.astype(int), y_pred)
                pred_score['Accuracy'] = accuracy_score(
                    test_data_df['dx'].values.astype(int),
                    (y_pred>class_balance_thresh).astype(int))
                rand_df.append(pred_score)
            cat_rand_df = pd.DataFrame(rand_df)
            pred_score = cat_rand_df.mean()
            for key, val in pred_score.items():
                temp_df[key] = val
            all_scores.append(temp_df)

            del model

        elif training_config['architecture']=='BC-LSTM':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            
            if training_config['target'] == "MCI":
                train_tkdnames = train_data_df['tkdname'].drop_duplicates()
                train_dataset = BCLSTMDataset(dataframe=train_data_df, tkdnames=train_tkdnames)
                lang_train_loader = DataLoader(
                    train_dataset, 
                    batch_size=training_config['batch_size'], 
                    shuffle=True, 
                    drop_last=True,
                    collate_fn = utils.bclstm_collate_lang_fn)
                train_loader = DataLoader(
                    train_dataset, 
                    batch_size=training_config['batch_size'], 
                    shuffle=True, 
                    drop_last=True,
                    collate_fn = utils.bclstm_collate_fn)
                val_tkdnames = val_data_df['tkdname'].drop_duplicates()
                val_dataset = BCLSTMDataset(dataframe=val_data_df, tkdnames=val_tkdnames)
                lang_val_loader = DataLoader(
                    val_dataset, 
                    batch_size=training_config['batch_size'], 
                    shuffle=True, 
                    drop_last=True,
                    collate_fn = utils.bclstm_collate_lang_fn)
                val_loader = DataLoader(
                    val_dataset, 
                    batch_size=len(val_dataset),
                    shuffle=True, 
                    collate_fn = utils.bclstm_collate_fn)
                test_tkdnames = test_data_df['tkdname'].drop_duplicates()
                test_dataset = BCLSTMDataset(dataframe=test_data_df, tkdnames=test_tkdnames)
                test_loader = DataLoader(
                    test_dataset, 
                    batch_size=len(test_dataset),
                    shuffle=True, 
                    collate_fn = utils.bclstm_collate_fn)
                
                # Define the model
                sample_lang = np.load(train_data_df['Language_Filename'][0])
                sample_aud = np.load(train_data_df['Audio_Filename'][0])
                text_model = BCLSTMTextCNN()
                audio_model = BCLSTMUnimodalClassifier(
                    input_dim = 74 #covarep embeddings
                )
                model = BCLSTMClassifier(
                    lang_dim = 300, #glove embeddings
                    aud_dim = 74 #covarep embeddings
                )
                
                # Define the loss functions and optimizer
                class_weights = compute_class_weight(
                    class_weight='balanced', 
                    classes=np.unique(train_data_df['dx']), 
                    y=train_data_df['dx'].values
                )
                weights_dict = {index: value for index, value in enumerate(class_weights)}
                
                text_optimizer = torchAdam(text_model.parameters(), lr=training_config['learning_rate'])
                audio_optimizer = torchAdam(audio_model.parameters(), lr=training_config['learning_rate'])
                optimizer = torchAdam(model.parameters(), lr=training_config['learning_rate'])
                criterion_mci = nn.BCEWithLogitsLoss()
                
                # Language Training loop
                best_val_loss = float('inf')
                patience = 30
                for epoch in range(training_config['n_epochs']):
                    text_model.train()
                    epoch_loss = 0
                    for batch in lang_train_loader:
                        # Zero the gradients
                        text_optimizer.zero_grad()

                        # Unpack the batch
                        X_lang_batch, mci_targets = batch
                        mci_targets = mci_targets.to(device)
                        output_holder = []
                        target_holder = []
                        for (X_lang_transcript, mci_target) in zip(X_lang_batch, mci_targets):
                            X_lang = [utils.pad_and_truncate_bclstm(my_tensor) for my_tensor in X_lang_transcript]
                            stacked_X_lang = torch.stack(X_lang).to(device)
                            target_holder.append(mci_target * torch.ones(stacked_X_lang.shape[0]))

                            #Forward pass
                            output = text_model(stacked_X_lang)
                            output_holder.append(output)

                        
                        outputs = torch.concat(output_holder)
                        mci_targets = torch.concat(target_holder)
                        #Classification loss
                        loss = criterion_mci(outputs.squeeze(), mci_targets.float())
                        loss.backward()
                        text_optimizer.step()
                        
                        epoch_loss += loss.item()
                    print(f"Epoch {epoch+1}/{training_config['n_epochs']}, Loss: {epoch_loss/len(train_loader)}")
                    text_model.eval()
                    val_losses = []
                    for val_batch in lang_val_loader:
                        # Zero the gradients
                        text_optimizer.zero_grad()

                        # Unpack the batch
                        X_lang_batch, mci_targets = val_batch
                        mci_targets = mci_targets.to(device)
                        output_holder = []
                        target_holder = []
                        for (X_lang_transcript, mci_target) in zip(X_lang_batch, mci_targets):
                            X_lang = [utils.pad_and_truncate_bclstm(my_tensor) for my_tensor in X_lang_transcript]
                            stacked_X_lang = torch.stack(X_lang).to(device)
                            target_holder.append(mci_target * torch.ones(stacked_X_lang.shape[0]))

                            #Forward pass
                            with torch.no_grad():
                                output = text_model(stacked_X_lang)
                                output_holder.append(output)

                        
                        val_outputs = torch.concat(output_holder)
                        val_mci_targets = torch.concat(target_holder)
                        loss = criterion_mci(val_outputs.squeeze(), val_mci_targets.float())
                        val_losses.append(loss.item())
                    # for val_batch in val_loader:
                    #     [X_lang_batch, X_aud_batch], val_mci_targets, sentence_counts = val_batch
                    #     val_mci_targets = val_mci_targets.to(device)
                    #     val_output_holder = []
                    #     for X_lang_transcript, X_aud_transcript, sentence_count in zip(X_lang_batch, X_aud_batch, sentence_counts):
                    #         X_lang = [torch.tensor(item) for item in X_lang_transcript]
                    #         X_aud = [torch.tensor(item) for item in X_aud_transcript]

                    #         # Calculate lengths of the sequences before padding
                    #         X_lang_lengths = torch.tensor([len(x) for x in X_lang]).to('cpu')
                    #         X_aud_lengths = torch.tensor([len(x) for x in X_aud]).to('cpu')

                    #         # Pad sequences for X_lang and X_aud
                    #         X_lang_padded = nn.utils.rnn.pad_sequence(X_lang, batch_first=True).to(device)
                    #         X_aud_padded = nn.utils.rnn.pad_sequence(X_aud, batch_first=True).to(device)
                    #         X_lang_packed = nn.utils.rnn.pack_padded_sequence(X_lang_padded, X_lang_lengths, batch_first=True, enforce_sorted=False).to(device)
                    #         X_aud_packed = nn.utils.rnn.pack_padded_sequence(X_aud_padded, X_aud_lengths, batch_first=True, enforce_sorted=False).to(device)

                    #         # Prepare inputs for the model
                    #         inputs = [X_lang_packed, X_aud_packed]
                    #         lengths = [X_lang_lengths, X_aud_lengths]

                    #         inputs = X_lang_packed
                    #         lengths = X_lang_lengths

                    #         #Forward pass
                    #         with torch.no_grad():
                    #             val_output, _ = text_model([inputs, lengths])
                    #             val_output_holder.append(val_output)
                        
                    #     val_outputs = torch.concat(val_output_holder)
                    #     loss = criterion_mci(val_outputs.squeeze(), val_mci_targets.float())
                    #     val_losses.append(loss.item())
                    
                    avg_val_loss = np.mean(val_losses)
                    print(f'Validation Loss: {avg_val_loss:.4f}')
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        best_model = text_model.state_dict()
                        early_stop_counter = 0
                    else:
                        early_stop_counter += 1
                    
                    if early_stop_counter >= patience:
                        print(f'Early stopping after epoch {epoch+1} with validation loss {best_val_loss:.4f}')
                        break
                text_model.load_state_dict(best_model)

                # Audio Training loop
                best_val_loss = float('inf')
                patience = 30
                for epoch in range(training_config['n_epochs']):
                    model.train()
                    epoch_loss = 0
                    for batch in train_loader:
                        # Zero the gradients
                        optimizer.zero_grad()

                        # Unpack the batch
                        [X_lang_batch, X_aud_batch], mci_targets, sentence_counts = batch
                        output_holder = []
                        for X_lang_transcript, X_aud_transcript, sentence_count in zip(X_lang_batch, X_aud_batch, sentence_counts):
                            X_lang = [torch.tensor(item) for item in X_lang_transcript]
                            X_aud = [torch.tensor(item) for item in X_aud_transcript]

                            # Calculate lengths of the sequences before padding
                            X_lang_lengths = torch.tensor([len(x) for x in X_lang]).to('cpu')
                            X_aud_lengths = torch.tensor([len(x) for x in X_aud]).to('cpu')

                            # Pad sequences for X_lang and X_aud
                            X_lang_padded = nn.utils.rnn.pad_sequence(X_lang, batch_first=True).to(device)
                            X_aud_padded = nn.utils.rnn.pad_sequence(X_aud, batch_first=True).to(device)
                            X_lang_packed = nn.utils.rnn.pack_padded_sequence(X_lang_padded, X_lang_lengths, batch_first=True, enforce_sorted=False).to(device)
                            X_aud_packed = nn.utils.rnn.pack_padded_sequence(X_aud_padded, X_aud_lengths, batch_first=True, enforce_sorted=False).to(device)

                            mci_targets = mci_targets.to(device)

                            # Prepare inputs for the model
                            inputs = [X_lang_packed, X_aud_packed]
                            lengths = [X_lang_lengths, X_aud_lengths]

                            #Forward pass
                            output = model([inputs, lengths])
                            output_holder.append(output)
                        
                        outputs = torch.concat(output_holder)
                        #Classification loss
                        loss = criterion_mci(outputs.squeeze(), mci_targets.float())
                        loss.backward()
                        optimizer.step()
                        
                        epoch_loss += loss.item()
                    print(f"Epoch {epoch+1}/{training_config['n_epochs']}, Loss: {epoch_loss/len(train_loader)}")

                    model.eval()
                    val_losses = []
                    for val_batch in val_loader:
                        [X_lang_batch, X_aud_batch], val_mci_targets, sentence_counts = val_batch
                        val_mci_targets = val_mci_targets.to(device)
                        val_output_holder = []
                        for X_lang_transcript, X_aud_transcript, sentence_count in zip(X_lang_batch, X_aud_batch, sentence_counts):
                            X_lang = [torch.tensor(item) for item in X_lang_transcript]
                            X_aud = [torch.tensor(item) for item in X_aud_transcript]

                            # Calculate lengths of the sequences before padding
                            X_lang_lengths = torch.tensor([len(x) for x in X_lang]).to('cpu')
                            X_aud_lengths = torch.tensor([len(x) for x in X_aud]).to('cpu')

                            # Pad sequences for X_lang and X_aud
                            X_lang_padded = nn.utils.rnn.pad_sequence(X_lang, batch_first=True).to(device)
                            X_aud_padded = nn.utils.rnn.pad_sequence(X_aud, batch_first=True).to(device)
                            X_lang_packed = nn.utils.rnn.pack_padded_sequence(X_lang_padded, X_lang_lengths, batch_first=True, enforce_sorted=False).to(device)
                            X_aud_packed = nn.utils.rnn.pack_padded_sequence(X_aud_padded, X_aud_lengths, batch_first=True, enforce_sorted=False).to(device)

                            # Prepare inputs for the model
                            inputs = [X_lang_packed, X_aud_packed]
                            lengths = [X_lang_lengths, X_aud_lengths]

                            #Forward pass
                            with torch.no_grad():
                                val_output = model([inputs, lengths])
                                val_output_holder.append(val_output)
                        
                        val_outputs = torch.concat(val_output_holder)
                        loss = criterion_mci(val_outputs.squeeze(), val_mci_targets.float())
                        val_losses.append(loss.item())
                    
                    avg_val_loss = np.mean(val_losses)
                    print(f'Validation Loss: {avg_val_loss:.4f}')
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        best_model = model.state_dict()
                        early_stop_counter = 0
                    else:
                        early_stop_counter += 1
                    
                    if early_stop_counter >= patience:
                        print(f'Early stopping after epoch {epoch+1} with validation loss {best_val_loss:.4f}')
                        break
                model.load_state_dict(best_model)

            model.to('cpu')
            model.device='cpu'
            # train_scores = utils.get_scores(train_tkdnames, train_data_df, model, pytorch=True, modality='MFN')
            test_scores = utils.get_scores(test_tkdnames, test_data_df, model, pytorch=True, modality='BC-LSTM')

            temp_df = {}
            temp_df['Outer Fold'] = ii
            temp_df['Ensemble Model'] = args.model_name
            for key, val in test_scores.items():
                temp_df[key] = val
            all_scores.append(temp_df)

            temp_df = {}
            temp_df['Outer Fold'] = ii
            temp_df['Ensemble Model'] = 'Random'
            rand_df = []
            class_balance_thresh = (1-(train_data_df['dx'].values.sum()/train_data_df['dx'].values.shape[0]))
            for jj in range(100):
                y_pred = np.random.uniform(size=test_data_df['dx'].values.shape)
                pred_score = utils.scores(test_data_df['dx'].values.astype(int), y_pred)
                pred_score['Accuracy'] = accuracy_score(
                    test_data_df['dx'].values.astype(int),
                    (y_pred>class_balance_thresh).astype(int))
                rand_df.append(pred_score)
            cat_rand_df = pd.DataFrame(rand_df)
            pred_score = cat_rand_df.mean()
            for key, val in pred_score.items():
                temp_df[key] = val
            all_scores.append(temp_df)

            del model

        elif training_config['architecture']=='FARZANA':
            language_flag = 1.0
            prosody_flag = 1.0
            if training_config['target'] == "MCI":
                lang_train_embeddings = np.vstack(
                    [np.load(filepath) for filepath in train_data_df['Language_Filename']]
                    )*language_flag
                aud_train_embeddings = np.vstack(
                    [np.load(filepath) for filepath in train_data_df['Audio_Filename']]
                    )*prosody_flag
                train_embeddings = np.concatenate((lang_train_embeddings,aud_train_embeddings), axis=-1)
                train_mean = train_embeddings.mean(axis=0)
                train_std = train_embeddings.std(axis=0)
                train_embeddings = (train_embeddings - train_mean) / (train_std + 1e-7)
                train_y_true = train_data_df['dx'].values.astype(int)
                train_y_true_eval = train_data_df[['tkdname','dx']].drop_duplicates()['dx'].values.astype(int)
                
                lang_test_embeddings = np.vstack(
                    [np.load(filepath) for filepath in test_data_df['Language_Filename']]
                    )*language_flag
                aud_test_embeddings = np.vstack(
                    [np.load(filepath) for filepath in test_data_df['Audio_Filename']]
                    )*prosody_flag
                test_embeddings = np.concatenate((lang_test_embeddings,aud_test_embeddings), axis=-1)
                test_embeddings = (test_embeddings - train_mean) / (train_std + 1e-7)
                test_y_true = test_data_df['dx'].values.astype(int)
                test_y_true_eval = test_data_df[['tkdname','dx']].drop_duplicates()['dx'].values.astype(int)
            
            data_train = {training_config['modality']: train_embeddings}
            data_test = {training_config['modality']: test_embeddings}

            base_predictors = {
                    'LR': LogisticRegression(),
                    'SVM': SVC(probability=True),
                    }

            for predictor_name, base_predictor in base_predictors.items():
                print(f'Fitting {predictor_name}')
                base_predictor.fit(train_embeddings,train_y_true)
                pred_score = utils.get_scores(
                    test_tkdnames, 
                    test_data_df, 
                    base_predictor, 
                    EI=True, 
                    modality=training_config['modality'],
                    data_mean=train_mean, 
                    data_std=train_std)

                temp_df = {}
                temp_df['Outer Fold'] = ii
                temp_df['Ensemble Model'] = predictor_name
                for key, val in pred_score.items():
                    temp_df[key] = val
                all_scores.append(temp_df)

            _, tkdname_indices = utils.average_ei_model_output_per_tkdname(
                test_tkdnames, 
                test_data_df, 
                base_predictor, 
                modality=training_config['modality'])

            temp_df = {}
            temp_df['Outer Fold'] = ii
            temp_df['Ensemble Model'] = 'Random'
            rand_df = []
            class_balance_thresh = (1-(train_data_df['dx'].values.sum()/train_data_df['dx'].values.shape[0]))
            for jj in range(100):
                y_pred = np.random.uniform(size=test_data_df['dx'].values.shape)
                pred_score = utils.scores(test_data_df['dx'].values.astype(int), y_pred)
                pred_score['Accuracy'] = accuracy_score(
                    test_data_df['dx'].values.astype(int),
                    (y_pred>class_balance_thresh).astype(int))
                rand_df.append(pred_score)
            cat_rand_df = pd.DataFrame(rand_df)
            pred_score = cat_rand_df.mean()
            for key, val in pred_score.items():
                temp_df[key] = val
            all_scores.append(temp_df)

    
    scores_df = pd.DataFrame(all_scores)
    os.makedirs(training_config['scores_outdir'], exist_ok=True)
    scores_outpath = os.path.join(training_config['scores_outdir'],args.model_name+'.csv')
    scores_df.to_csv(scores_outpath)
