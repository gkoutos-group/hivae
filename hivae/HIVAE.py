import sys
import tensorflow as tf
import graph_new
import time
import numpy as np
#import plot_functions
import read_functions
import os
import csv

class HIVAE():
    def __init__(self,miss_file = None,true_miss_file = None,types_dict = {'AGE':'count,1','SEX':'cat,2,2','BMI':'pos,1','BP':'pos,1','S1':'pos,1','S2':'pos,1','S3':'pos,1','S4':'pos,1','S5':'pos,1','S6':'pos,1','Y':'pos,1'},network_dict ={'batch_size': 32,'model':'model_HIVAE_inputDropout','z_dim': 5,'y_dim':5,'s_dim': 3,'mask':1},networkl_path = './models'):
        self.model_name = network_dict['model_name']
        self.m_perc = 0
        self.mask = network_dict['mask']
        self.types_dict = types_dict
        self.save = 1000
        self.dim_s= network_dict['dim_s']
        self.dim_z = network_dict['dim_z']
        self.dim_y = network_dict['dim_y']

        self.savefile = str(str(self.model_name)+'_'+'_Missing'+str(self.m_perc)+'_'+str(self.mask)+'_z'+str(self.dim_z)+'_y'+str(self.dim_y)+'_s'+str(self.dim_s)+'_batch'+str(self.training()[1]))
        # Create a directoy for the save file
        if not os.path.exists(networkl_path + self.savefile):
            os.makedirs('./saved_networks/' + self.savefile)
        self.network_file_name = networkl_path + self.savefile  + '.ckpt'
        self.log_file_name = networkl_path + self.savefile + '/log_file_' + self.savefile + '.txt'

    def _initialize_net(self,types_dict,batch_size):
        if self.types_dict:
            sess_HVAE = tf.Graph()
            with sess_HVAE.as_default():
                tf_nodes = graph_new.HVAE_graph(self.model_name, types_dict, batch_size, learning_rate=1e-3,
                                                z_dim=self.dim_latent_z, y_dim=self.dim_latent_y,
                                                s_dim=self.dim_latent_s, y_dim_partition=self.dim_latent_y)
                return tf_nodes,sess_HVAE


    def _train(self,traindata,typesdict,miss_file=None,true_miss_file=None,epochs=100,batchsize=1000):
        train_data, types_dict, miss_mask, true_miss_mask, n_samples = read_functions.read_data_df(traindata,
                                                                                                typesdict,
                                                                                                miss_file,
                                                                                                true_miss_file)

        n_batches = int(np.floor(np.shape(traindata)[0] / batchsize))
        miss_mask = np.multiply(miss_mask, true_miss_mask)

        with tf.Session(graph=self._iniatilze_net(types_dict,batchsize)[1]) as session:

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()

            if (epochs==1):
                saver.restore(session, self.network_file_name)
                print("Model restored.")
            else:
                #        saver = tf.train.Saver()
                print('Initizalizing Variables ...')
                tf.global_variables_initializer().run()
        tf_nodes = self.initialize()
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
            avg_KL_z = 0.
            samples_list = []
            p_params_list = []
            q_params_list = []
            log_p_x_total = []
            log_p_x_missing_total = []

            # Annealing of Gumbel-Softmax parameter
            if epochs==1:
                tau = 1e-3
            else:
                tau = np.max([1.0 - 0.01 * epoch, 1e-3])
                tau2 = np.min([0.001 * epoch, 1.0])


            # Randomize the data in the mini-batches
            random_perm = np.random.permutation(range(np.shape(train_data)[0]))
            train_data_aux = train_data[random_perm, :]
            miss_mask_aux = miss_mask[random_perm, :]
            true_miss_mask_aux = true_miss_mask[random_perm, :]

            for i in range(n_batches):
                # Create inputs for the feed_dict
                data_list, miss_list = read_functions.next_batch(train_data_aux, types_dict, miss_mask_aux, batchsize,
                                                                 index_batch=i)

                # Delete not known data (input zeros)
                data_list_observed = [data_list[i] * np.reshape(miss_list[:, i], [batchsize, 1]) for i in
                                      range(len(data_list))]

                # Create feed dictionary
                feedDict = {i: d for i, d in zip(tf_nodes['ground_batch'], data_list)}
                feedDict.update({i: d for i, d in zip(tf_nodes['ground_batch_observed'], data_list_observed)})
                feedDict[tf_nodes['miss_list']] = miss_list
                feedDict[tf_nodes['tau_GS']] = tau
                if epoch!=1:
                    feedDict[tf_nodes['tau_var']] = tau2

                # Running VAE
                _, loss, KL_z, KL_s, samples, log_p_x, log_p_x_missing, p_params, q_params = session.run(
                    [tf_nodes['optim'], tf_nodes['loss_re'], tf_nodes['KL_z'], tf_nodes['KL_s'], tf_nodes['samples'],
                     tf_nodes['log_p_x'], tf_nodes['log_p_x_missing'], tf_nodes['p_params'], tf_nodes['q_params']],
                    feed_dict=feedDict)

                samples_test, log_p_x_test, log_p_x_missing_test, test_params = session.run(
                    [tf_nodes['samples_test'], tf_nodes['log_p_x_test'], tf_nodes['log_p_x_missing_test'],
                     tf_nodes['test_params']],
                    feed_dict=feedDict)

                # summary_writer.add_summary(samples_test, epoch * n_batches + i)
                # # write out networks structure
                # if epoch > 5:
                #     tf.train.write_graph(tf.get_default_graph(), os.getcwd(), 'graph.json')
                #     tf.train.write_graph(tf.get_default_graph(), os.getcwd(), 'graph.txt')

                #     ops = session.graph.get_operations()

                #     print('*'*80)
                #     print(ops)
                #     print('*'*80)
                #     #op.name gives you the name and op.values() g

                #                #Collect all samples, distirbution parameters and logliks in lists
                #                samples_list.append(samples)
                #                p_params_list.append(p_params)
                #                q_params_list.append(q_params)
                #                log_p_x_total.append(log_p_x)
                #                log_p_x_missing_total.append(log_p_x_missing)

                # Evaluate results on the imputation with mode, not on the samlpes!
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

            # Concatenate samples in arrays
            s_total, z_total, y_total, est_data = read_functions.samples_concatenation(samples_list)

            # Transform discrete variables back to the original values
            train_data_transformed = read_functions.discrete_variables_transformation(
                train_data_aux[:n_batches * batchsize, :], self.types_dict)
            est_data_transformed = read_functions.discrete_variables_transformation(est_data, self.types_dict)
            est_data_imputed = read_functions.mean_imputation(train_data_transformed,
                                                              miss_mask_aux[:n_batches * batchsize, :],
                                                              self.types_dict)

            #            est_data_transformed[np.isinf(est_data_transformed)] = 1e20

            # Create global dictionary of the distribution parameters
            p_params_complete = read_functions.p_distribution_params_concatenation(p_params_list, self.types_dict,
                                                                                   self.dim_z, self.dim_s)
            q_params_complete = read_functions.q_distribution_params_concatenation(q_params_list, self.dim_z,
                                                                                   self.dim_s)

            # Number of clusters created
            cluster_index = np.argmax(q_params_complete['s'], 1)
            cluster = np.unique(cluster_index)
            print('Clusters: ' + str(len(cluster)))

            # Compute mean and mode of our loglik models
            loglik_mean, loglik_mode = read_functions.statistics(p_params_complete['x'], self.types_dict)
            #            loglik_mean[np.isinf(loglik_mean)] = 1e20

            # Try this for the errors
            error_train_mean, error_test_mean = read_functions.error_computation(train_data_transformed, loglik_mean,
                                                                                 self.types_dict, miss_mask_aux[
                                                                                             :n_batches * batchsize,
                                                                                             :])
            error_train_mode, error_test_mode = read_functions.error_computation(train_data_transformed, loglik_mode,
                                                                                 self.types_dict, miss_mask_aux[
                                                                                             :n_batches * batchsize,
                                                                                             :])
            error_train_samples, error_test_samples = read_functions.error_computation(train_data_transformed,
                                                                                       est_data_transformed, self.types_dict,
                                                                                       miss_mask_aux[
                                                                                       :n_batches * batchsize, :])
            error_train_imputed, error_test_imputed = read_functions.error_computation(train_data_transformed,
                                                                                       est_data_imputed, self.types_dict,
                                                                                       miss_mask_aux[
                                                                                       :n_batches * batchsize, :])

            # Compute test-loglik from log_p_x_missing
            log_p_x_total = np.transpose(np.concatenate(log_p_x_total, 1))
            log_p_x_missing_total = np.transpose(np.concatenate(log_p_x_missing_total, 1))
            if true_miss_file:
                log_p_x_missing_total = np.multiply(log_p_x_missing_total,
                                                    true_miss_mask_aux[:n_batches * batchsize, :])
            avg_test_loglik = np.sum(log_p_x_missing_total) / np.sum(1.0 - miss_mask_aux)



    def training(self,traindata,results_path='./results',epochs=200,batchsize=1000,miss_mask=None,true_miss_mask=None):
        train = self._train(traindata,self.types_dict,miss_mask,true_miss_mask,epochs,batchsize)
        # Display logs per epoch step
        '''if epoch % display == 0:
            print_loss(epoch, train.start_time, train.avg_loss / n_batches, avg_test_loglik, avg_KL_s / n_batches,
                       avg_KL_z / n_batches)
            print('Test error mode: ' + str(np.round(np.mean(train.error_test_mode), 3)))
            print("")'''


        # Compute train and test loglik per variables
        loglik_per_variable = np.sum(train.log_p_x_total, 0) / np.sum(train.miss_mask_aux, 0)
        loglik_per_variable_missing = np.sum(train.log_p_x_missing_total, 0) / np.sum(1.0 - train.miss_mask_aux, 0)

        # Store evolution of all the terms in the ELBO
        train.loglik_epoch.append(loglik_per_variable)
        train.testloglik_epoch.append(loglik_per_variable_missing)
        train.KL_s_epoch.append(train.avg_KL_s / train.n_batches)
        train.KL_z_epoch.append(train.avg_KL_z / train.n_batches)
        train.error_train_mode_global.append(train.error_train_mode)
        train.error_test_mode_global.append(train.error_test_mode)

        if train.epoch % self.save == 0:
            print('Saving Variables ...')
            save_path = train.saver.save(train.session, self.network_file_name)

        print('Training Finished ...')

        # Saving needed variables in csv
        if not os.path.exists(results_path + '/' + self.savefile):
            os.makedirs(results_path + '/' + self.savefile)

        with open(results_path + '/' + self.savefile + '/loglik.csv', "w") as f:
            writer = csv.writer(f)
            writer.writerows(train.loglik_epoch)

        with open(results_path + '/' + self.savefile + '/testloglik.csv', "w") as f:
            writer = csv.writer(f)
            writer.writerows(train.testloglik_epoch)

        with open(results_path + '/' + self.savefile + '/KL_s.csv', "w") as f:
            writer = csv.writer(f)
            writer.writerows(np.reshape(train.KL_s_epoch, [-1, 1]))

        with open(results_path + '/' + self.savefile + '/KL_z.csv', "w") as f:
            writer = csv.writer(f)
            writer.writerows(np.reshape(train.KL_z_epoch, [-1, 1]))

        with open(results_path+ '/' + self.savefile + '/train_error.csv', "w") as f:
            writer = csv.writer(f)
            writer.writerows(train.error_train_mode_global)

        with open(results_path + '/' + self.savefile + '/test_error.csv', "w") as f:
            writer = csv.writer(f)
            writer.writerows(train.error_test_mode_global)

        # Save the variables to disk at the end
        save_path = train.saver.save(train.session, self.network_file_name)



    def testing(self,testdata,result_path='./results_test/',batchsize = 1000000,):
        test = self._train(1,batchsize,self.types_dict)

        '''# Display logs per epoch step
        if args.display == 1:
            #            print_loss(0, start_time, avg_loss/n_batches, avg_test_loglik, avg_KL_s/n_batches, avg_KL_z/n_batches)
            print(np.round(test.error_test_mode, 3))
            print('Test error mode: ' + str(np.round(np.mean(test.error_test_mode), 3)))
            print("")'''

        # Plot evolution of test loglik
        loglik_per_variable = np.sum(np.concatenate(test.log_p_x_total, 1), 1) / np.sum(test.miss_mask, 0)
        loglik_per_variable_missing = np.sum(test.log_p_x_missing_total, 0) / np.sum(1.0 - test.miss_mask, 0)

        test.loglik_epoch.append(loglik_per_variable)
        test.testloglik_epoch.append(loglik_per_variable_missing)

        print('Test loglik: ' + str(np.round(np.mean(loglik_per_variable_missing), 3)))

        # Re-run test error mode
        test.error_train_mode_global.append(test.error_train_mode)
        test.error_test_mode_global.append(test.error_test_mode)
        test.error_imputed_global.append(test.error_test_imputed)

        # Store data samples
        test.est_data_transformed_total.append(test.est_data_transformed)

        # Compute the data reconstruction

        data_reconstruction = test.train_data_transformed * test.miss_mask_aux[:test.n_batches * batchsize, :] + \
                          np.round(test.loglik_mode, 3) * (1 - test.miss_mask_aux[:test.n_batches * batchsize, :])

        #        data_reconstruction = -1 * miss_mask_aux[:n_batches*args.batch_size,:] + \
        #                                np.round(loglik_mode,3) * (1 - miss_mask_aux[:n_batches*args.batch_size,:])

        train_data_transformed = test.train_data_transformed[np.argsort(test.random_perm)]
        data_reconstruction = data_reconstruction[np.argsort(test.random_perm)]

        train_data_transformed = train_data_transformed[np.argsort(test.random_perm)]
        data_reconstruction = data_reconstruction[np.argsort(test.random_perm)]

        if not os.path.exists(result_path + self.savefile):
            os.makedirs(result_path + self.save_file)

        with open(result_path + self.savefile + '/' + self.savefile + '_data_reconstruction.csv', "w") as f:
            writer = csv.writer(f)
            writer.writerows(data_reconstruction)
        with open(result_path + self.savefile + '/' + self.savefile + '_data_true.csv', "w") as f:
            writer = csv.writer(f)
            writer.writerows(train_data_transformed)

        # Saving needed variables in csv
        if not os.path.exists(result_path+'/Results_test_csv/' + self.savefile):
            os.makedirs('./Results_test_csv/' + self.savefile)

        # Train loglik per variable
        with open(result_path+'Results_test_csv/' + self.savefile + '/' + self.savefile + '_loglik.csv', "w") as f:
            writer = csv.writer(f)
            writer.writerows(test.loglik_epoch)

        # Test loglik per variable
        with open(result_path+'Results_test_csv/' + self.savefile + '/' + self.savefile + '_testloglik.csv', "w") as f:
            writer = csv.writer(f)
            writer.writerows(test.testloglik_epoch)

        # Train NRMSE per variable
        with open(result_path+'Results_test_csv/' + self.savefile + '/' + self.savefile + '_train_error.csv', "w") as f:
            writer = csv.writer(f)
            writer.writerows(test.error_train_mode_global)

        # Test NRMSE per variable
        with open(result_path+'Results_test_csv/' + self.savefile + '/' + self.savefile + '_test_error.csv', "w") as f:
            writer = csv.writer(f)
            writer.writerows(test.error_test_mode_global)

        # Number of clusters
        with open(result_path+'Results_test_csv/' + self.savefile + '/' + self.savefile + '_clusters.csv', "w") as f:
            writer = csv.writer(f)
            writer.writerows([[len(test.cluster)]])



