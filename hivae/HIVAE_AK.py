#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""New library for HI-VAE

@author: fathyshalaby,athro
"""
import sys
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import pprint
pprinter = pprint.PrettyPrinter(depth=3)
import graph_new
import time
import numpy as np
import pandas as pd
#import plot_functions
import read_functions
import os
import csv
from pathlib import Path


verbosity = 1

def vprint(verbosity_level,*args):
    if verbosity_level>=verbosity:
        if verbosity_level==1:
            print('INFO :\t',*args)
        elif verbosity_level==2:
            print('DEBUG:\t',*args)
        elif verbosity_level==3:
            print('ERROR:\t',*args)


    
              

#ak: small method to savely get information from a dictionary
def save_get(aHash,aKey,aDefault):
    try:
        return aHash[aKey]
    except:
        pass
    return aDefault

class HIVAE():
    def __init__(self,
                 types_description,
                 network_dict,
                 network_path     = './models',  # network_path???
                 results_path     = './results', #ak: included a path for results
                 miss_file        = None,  #ak: why?
                 true_miss_file   = None,  #ak: why?
                 save_every_epoch = 1000,
                 ):
        self.model_name       = save_get(network_dict,'model_name','unbkown_model_name')
        self.m_perc           = 0
        self.mask             = save_get(network_dict,'mask',None)
        self.types_list       = read_functions.get_types_list(types_description)
        self.save             = 1000 # parameter?
        self.save_every_epoch = save_every_epoch    
        self.dim_s            = save_get(network_dict,'dim_s',None) 
        self.dim_z            = save_get(network_dict,'dim_z',None) 
        self.dim_y            = save_get(network_dict,'dim_y',None)
        self.batch_size       = save_get(network_dict,'batch_size',32)
        # self.true_miss_file   = true_miss_file
        # self.miss_file        = miss_file
        self.network_path     = network_path
        self.results_path     = results_path
        self.display          = True
        
        ###ak: not sure why the training method is called while constructing the string
        #self.savefile = str(str(self.model_name)+'_'+'_Missing'+str(self.m_perc)+'_'+str(self.mask)+'_z'+str(self.dim_z)+'_y'+str(self.dim_y)+'_s'+str(self.dim_s)+'_batch'+str(self.training()[1]))
        self.savefile = str(str(self.model_name)+'_'+'_Missing'+str(self.m_perc)+'_'+str(self.mask)+'_z'+str(self.dim_z)+'_y'+str(self.dim_y)+'_s'+str(self.dim_s)+'_batch'+str(save_get(network_dict,'batch_size','batch_size_unkown')))
        self.experiment_name = '{}_s{}_z{}_y{}_batch{}'.format(self.model_name,self.dim_s,self.dim_z,self.dim_y,self.batch_size)

        # Create a directoy for the save file
        full_network_path = '{}/{}'.format(self.network_path,self.experiment_name)
        try:
            if not os.path.exists(full_network_path):
                os.makedirs(full_network_path)
        except:
            vprint(3,'Could not create network path <<{}>>'.format(full_network_path))
            pass
        

        self.network_file_name = '{}.ckpt'.format(full_network_path) 
        self.log_file_name = '{}/log_file_{}.txt'.format(full_network_path,self.savefile)


    def print_loss(self,epoch, start_time, avg_loss, avg_test_loglik, avg_KL_s, avg_KL_z):
        
        vprint(1,'Epoch: {:4d}\ttime: {:4.2f}\ttrain_loglik: {:5.2f}\tKL_z: {:5.2f}\tKL_s: {:5.2f}\tELBO: {:5.2f}\tTest_loglik: {:5.2f}'.format(epoch, time.time() - start_time, avg_loss, avg_KL_z, avg_KL_s, avg_loss-avg_KL_z-avg_KL_s, avg_test_loglik))


    #ak: save the data in a more coherent fashion
    def save_data(self,the_main_directory,the_path,the_data):
        dir_present = False
        try:
            if not os.path.exists(the_main_directory):
                os.makedirs(the_main_directory)
            dir_present = True
        except:
            vprint(3,'Could not create path (<<{}>>)'.format(the_main_directory))

        if dir_present:
            full_file_path = '{}/{}'.format(the_main_directory,the_path)
            try:
                with open(full_file_path, 'w') as f:
                    writer = csv.writer(f)
                    writer.writerows(the_data)
            except:
                vprint(3,'Problens writing data to <<{}>>\n{}'.format(full_file_path,str(the_data)))
            
            
        

    def training_ak(self,training_data,epochs=200,
                        learning_rate=1e-3,
                        results_path='./',
                        restore=False,
                        train_or_test=True,
                        restore_session=False,
                        true_missing_mask=None,
                        missing_mask = None,
                        verbosity=3):
        
        vprint(2,'len(training_data)',len(training_data))

        # train_or_test == True - training
        training_phase  = train_or_test
        testing_phase   = not train_or_test

        batch_size_here = self.batch_size
        
        # if testing phase use all as one single batch
        if testing_phase:
            batch_size_here = len(training_data)
        
        # train_data = None
        # # test whether data is pandas DataFrame or numpy.nparray
        # if type(training_data) == type(pd.DataFrame()):
        #     train_data = training_data.to_numpy()
        # elif type(training_data) == type((np.ndarray([1]))):
        #     train_data = training_data
        # else:
        #     print('DEBUG-AK: Data not in correct format, Type = {}'.format(type(training_data)))
        #     sys.exit(-100)

        # train_data, _ , miss_mask, true_miss_mask, n_samples = read_functions.read_data_df_as_input(training_data,
        #                                                                                         self.types_list,
        #                                                                                         self.miss_file,
        #                                                                                         self.true_miss_file)

        true_miss_set = False
        miss_set      = False
        # deal with true missing data
        if type(true_missing_mask)==type(pd.DataFrame()) and not true_missing_mask.empty:
            true_miss_mask = true_missing_mask.to_numpy()
            true_miss_set = True
        else:
            true_miss_mask = np.ones(training_data.shape,dtype=int)

        # deal with non-true missing data
        if type(missing_mask)==type(pd.DataFrame()) and not missing_mask.empty:
            miss_mask = missing_mask.to_numpy()
            miss_set = True
        else:
            miss_mask = np.ones(training_data.shape,dtype=int)

        train_data, _ , n_samples = read_functions.read_data_df(training_data,self.types_list,true_miss_mask)

            
        n_batches = int(np.floor(np.shape(train_data)[0]/batch_size_here))
        #Compute the real miss_mask
        miss_mask = np.multiply(miss_mask, true_miss_mask)
        #Creating graph
        sess_HVAE = tf.Graph()
        
        with sess_HVAE.as_default():

            tf_nodes = graph_new.HVAE_graph(
                self.model_name,
                self.types_list,
                batch_size_here,
                learning_rate=learning_rate,
                z_dim=self.dim_z,
                y_dim=self.dim_y,
                s_dim=self.dim_s,
                y_dim_partition=None)

        ################### Running the VAE Training #################################
        with tf.Session(graph=sess_HVAE) as session:
        
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()
        
            if restore_session:
                saver.restore(session, self.network_file_name)
                vprint(1,"Model restored.")
            else:
                vprint(1,'Initizalizing Variables ...')
                tf.global_variables_initializer().run()
    
            if training_phase:
                vprint(1,'Training the HVAE ...')
            elif testing_phase:
                vprint(1,'Testing the HVAE ...')
                
        
            start_time = time.time()
            # Training cycle
        
            loglik_epoch = []
            testloglik_epoch = []
            error_train_mode_global = []
            error_test_mode_global = []
            KL_s_epoch = []
            KL_z_epoch = []

            for epoch in range(epochs):
                avg_loss = 0.
                avg_KL_s = 0.
                avg_KL_y = 0.
                avg_KL_z = 0.
                samples_list = []
                p_params_list = []
                q_params_list = []
                log_p_x_total = []
                log_p_x_missing_total = []
                encodinglist =[]
            
                # Annealing of Gumbel-Softmax parameter
                tau = None
                if training_phase:
                    tau = np.max([1.0 - 0.01*epoch,1e-3])
                elif testing_phase:
                    tau = 1e-3
                #            tau = 1e-3
                tau2 = np.min([0.001*epoch,1.0])

            
                #Randomize the data in the mini-batches
                np.random.seed(42)
                random_perm = np.random.permutation(range(np.shape(train_data)[0]))
                train_data_aux = train_data[random_perm,:]
                miss_mask_aux = miss_mask[random_perm,:]
                true_miss_mask_aux = true_miss_mask[random_perm,:]
            
                for i in range(n_batches):      
                
                    #Create inputs for the feed_dict
                    data_list, miss_list = read_functions.next_batch(train_data_aux, self.types_list, miss_mask_aux, batch_size_here, index_batch=i)

                    #Delete not known data (input zeros)
                    data_list_observed = [data_list[i]*np.reshape(miss_list[:,i],[batch_size_here,1]) for i in range(len(data_list))]
                
                    #Create feed dictionary
                    feedDict = {i: d for i, d in zip(tf_nodes['ground_batch'], data_list)}
                    feedDict.update({i: d for i, d in zip(tf_nodes['ground_batch_observed'], data_list_observed)})
                    feedDict[tf_nodes['miss_list']] = miss_list
                    feedDict[tf_nodes['tau_GS']] = tau
                    feedDict[tf_nodes['tau_var']] = tau2
                    
                    #Running VAE
                    loss,KL_z,KL_s,samples,log_p_x,log_p_x_missing,p_params,q_params = None,None,None,None,None,None,None,None
                    if training_phase:
                        _,loss,KL_z,KL_s,samples,log_p_x,log_p_x_missing,p_params,q_params  = session.run([tf_nodes['optim'],
                                                                                                               tf_nodes['loss_re'], tf_nodes['KL_z'],
                                                                                                               tf_nodes['KL_s'], tf_nodes['samples'],
                                                                                                               tf_nodes['log_p_x'], tf_nodes['log_p_x_missing'],
                                                                                                               tf_nodes['p_params'],tf_nodes['q_params']],
                                                                                                              feed_dict=feedDict)
                    elif testing_phase:
                        loss,KL_z,KL_s,samples,log_p_x,log_p_x_missing,p_params,q_params  = session.run([tf_nodes['loss_re'], tf_nodes['KL_z'],
                                                                                                             tf_nodes['KL_s'], tf_nodes['samples'],
                                                                                                             tf_nodes['log_p_x'], tf_nodes['log_p_x_missing'],
                                                                                                             tf_nodes['p_params'],tf_nodes['q_params']],
                                                                                                            feed_dict=feedDict)


                    samples_test,log_p_x_test,log_p_x_missing_test,test_params  = session.run([tf_nodes['samples_test'],
                                                                                                   tf_nodes['log_p_x_test'],tf_nodes['log_p_x_missing_test'],
                                                                                                   tf_nodes['test_params']],
                                                                                                  feed_dict=feedDict)
                
                    encoding = session.run(tf_nodes['encoding'],feed_dict=feedDict)
                    #Evaluate results on the imputation with mode, not on the samlpes!
                    encodinglist.append(encoding)
                    samples_list.append(samples_test)
                    p_params_list.append(test_params)
                    #                        p_params_list.append(p_params)
                    q_params_list.append(q_params)
                    log_p_x_total.append(log_p_x_test)
                    log_p_x_missing_total.append(log_p_x_missing_test)
                
                    # Compute average loss
                    avg_loss += np.mean(loss)
                    avg_KL_s += np.mean(KL_s)
                    avg_KL_z += np.mean(KL_z)
                
                #Concatenate samples in arrays
                s_total, z_total, y_total, est_data = read_functions.samples_concatenation(samples_list)
                #Transform discrete variables back to the original values
                train_data_transformed = read_functions.discrete_variables_transformation(train_data_aux[:n_batches*batch_size_here,:], self.types_list)
                est_data_transformed = read_functions.discrete_variables_transformation(est_data, self.types_list)
                est_data_imputed = read_functions.mean_imputation(train_data_transformed, miss_mask_aux[:n_batches*batch_size_here,:], self.types_list)
            
                #est_data_transformed[np.isinf(est_data_transformed)] = 1e20
            
                #Create global dictionary of the distribution parameters
                p_params_complete = read_functions.p_distribution_params_concatenation(p_params_list, self.types_list, self.dim_z, self.dim_s)
                q_params_complete = read_functions.q_distribution_params_concatenation(q_params_list,  self.dim_z, self.dim_s)
            
                #Number of clusters created
                cluster_index = np.argmax(q_params_complete['s'],1)
                cluster = np.unique(cluster_index)
                vprint(1,'Clusters: ' + str(len(cluster)))
            
                #Compute mean and mode of our loglik models
                loglik_mean, loglik_mode = read_functions.statistics(p_params_complete['x'],self.types_list)
                #            loglik_mean[np.isinf(loglik_mean)] = 1e20

                # print('train_data_transformed',type(train_data_transformed))
                # print('train_data_transformed',train_data_transformed.shape)
                # print('loglik_mean',type(loglik_mean))
                # print('loglik_mean',loglik_mean.shape)
                # print('miss_mask_aux',type(miss_mask_aux))
                # print('miss_mask_aux',miss_mask_aux.shape)
                # print('n_batches',n_batches)
                        
                #Try this for the errors
                #ak: lost examples when batch_size and number of variables are not modulus 0
                error_train_mean, error_test_mean = read_functions.error_computation(train_data_transformed, loglik_mean, self.types_list, miss_mask_aux[:n_batches*batch_size_here,:])
                error_train_mode, error_test_mode = read_functions.error_computation(train_data_transformed, loglik_mode, self.types_list, miss_mask_aux[:n_batches*batch_size_here,:])
                error_train_samples, error_test_samples = read_functions.error_computation(train_data_transformed, est_data_transformed, self.types_list, miss_mask_aux[:n_batches*batch_size_here,:])
                error_train_imputed, error_test_imputed = read_functions.error_computation(train_data_transformed, est_data_imputed, self.types_list, miss_mask_aux[:n_batches*batch_size_here,:])

                #Compute test-loglik from log_p_x_missing
                log_p_x_total = np.transpose(np.concatenate(log_p_x_total,1))
                log_p_x_missing_total = np.transpose(np.concatenate(log_p_x_missing_total,1))
                if true_miss_set:
                    log_p_x_missing_total = np.multiply(log_p_x_missing_total,true_miss_mask_aux[:n_batches*batch_size_here,:])
                avg_test_loglik = 100000  ##  np.finfo(float).max #ak - maximal float number = ugly for printing
                if miss_set:
                    avg_test_loglik = np.sum(log_p_x_missing_total)/np.sum(1.0-miss_mask_aux)

                # Display logs per epoch step
                if self.display:
                    self.print_loss(epoch, start_time, avg_loss/n_batches, avg_test_loglik, avg_KL_s/n_batches, avg_KL_z/n_batches)
                    # print('Test error mode: ' + str(np.round(np.mean(error_test_mode),3)))
                    # print("")
                
                #Compute train and test loglik per variables
                loglik_per_variable = np.sum(log_p_x_total,0)/np.sum(miss_mask_aux,0)

                #ak: only compute if a missing file was supplied
                loglik_per_variable_missing =  100000  ##  np.finfo(float).max #ak - maximal float number = ugly for printing
                if miss_set:
                    loglik_per_variable_missing = np.sum(log_p_x_missing_total,0)/np.sum(1.0-miss_mask_aux,0)
            
                #Store evolution of all the terms in the ELBO
                loglik_epoch.append(loglik_per_variable)
                testloglik_epoch.append(loglik_per_variable_missing)
                KL_s_epoch.append(avg_KL_s/n_batches)
                KL_z_epoch.append(avg_KL_z/n_batches)
                error_train_mode_global.append(error_train_mode)
                error_test_mode_global.append(error_test_mode)
            
            
                if epoch % self.save == 0:
                    vprint(1,'Saving Variables ...')  
                    save_path = saver.save(session, self.network_file_name)
                
            if training_phase:
                vprint(1,'Training Finished ...')
                
                self.save_data(self.results_path,'{}_loglik.csv'.format(self.experiment_name),loglik_epoch)
                self.save_data(self.results_path,'{}_KL_s.csv'.format(self.experiment_name),np.reshape(KL_s_epoch,[-1,1]))
                self.save_data(self.results_path,'{}_KL_z.csv'.format(self.experiment_name),np.reshape(KL_z_epoch,[-1,1]))
                self.save_data(self.results_path,'{}_train_error.csv'.format(self.experiment_name),error_train_mode_global)
                self.save_data(self.results_path,'{}_test_error.csv'.format(self.experiment_name),error_test_mode_global)
                #ak: hack for now - as entries are not lists 
                self.save_data(self.results_path,'{}_testloglik.csv'.format(self.experiment_name),[[x] for x in testloglik_epoch])
                # Save the variables to disk at the end
                save_path = saver.save(session, self.network_file_name)
                return encoding['z']
            
            elif testing_phase:
                vprint(1,'Testing Finished ...')
                #Compute the data reconstruction
                # print('AK-DEBUG')
                # print(n_batches)
                # print(n_batches*batch_size_here)
                # print(len(miss_mask_aux[:n_batches*batch_size_here,:]))
                # print(len(train_data_transformed))
                # print('AK-DEBUG')
            
                data_reconstruction = train_data_transformed * miss_mask_aux[:n_batches*batch_size_here,:] + \
                  np.round(loglik_mode,3) * (1 - miss_mask_aux[:n_batches*batch_size_here,:])
                train_data_transformed = train_data_transformed[np.argsort(random_perm)]
                loglik_mean_reconstructed = loglik_mean[np.argsort(random_perm)]
                data_reconstruction = data_reconstruction[np.argsort(random_perm)]
                self.save_data(self.results_path,'{}_data_reconstruction.csv'.format(self.experiment_name),data_reconstruction)
                self.save_data(self.results_path,'{}_data_true.csv'.format(self.experiment_name),train_data_transformed)
                self.save_data(self.results_path,'{}_data_loglik_mean_reconstructed.csv'.format(self.experiment_name),loglik_mean_reconstructed)

                df_real        = pd.DataFrame(train_data_transformed)
                df_loglik_mean = pd.DataFrame(loglik_mean_reconstructed)
                vprint(2,'Reconstruction Correlation:')
                vprint(2,df_real.corrwith(df_loglik_mean))
                

                return (train_data_transformed,data_reconstruction,loglik_mean_reconstructed,
                            z_total[np.argsort(random_perm)],
                            s_total[np.argsort(random_perm)])

        
            



