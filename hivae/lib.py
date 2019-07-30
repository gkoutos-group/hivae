import sys
import tensorflow as tf
import graph_new
import time
import numpy as np
#import plot_functions
import read_functions
import os
import csv
import parser_arguments
class HIVAE():
    def __init__(self,parameters = {'model_name':"model_HIVAE_inputDropout",'data':"Adult/data.csv",'typesdict':"Adult/data_types.csv",'missing': "Adult/Missing/20_1.csv,true_missing:Adult/MissingTrue.csv",'z_dim':2,'y_dim':5,'s_dim':10,'dim_latent_z':2,'dim_latent_y':3,'dim_latent_s':4,'m_perc':20,'mask':1}):
        self.model_name = parameters['model_name']
        self.dataset = parameters['data']
        self.m_perc = parameters['m_perc']
        self.mask = parameters['mask']
        self.miss_file = parameters['missing']
        self.true_miss_file = parameters['true_missing']
        self.types_dict = None
        self.NN = None
        self.dim_latent_s = parameters['dim_latent_s']
        self.dim_latent_y = parameters['dim_latent_y']
        self.dim_latent_z = parameters['dim_latent_z']
        self.save = 1000
        self.dim_s= parameters['dim_s']
        self.dim_z = parameters['dim_z']
        self.dim_y = parameters['dim_y']
        self.train_data, self.types_dict, self.miss_mask, self.true_miss_mask, self.n_samples = read_functions.read_data(self.dataset,
                                                                                                self.types_file,
                                                                                                self.miss_file,
                                                                                                self.true_miss_file)
        self.savefile = str(str(self.model_name)+'_'+str(self.dataset)+'_Missing'+str(self.m_perc)+'_'+str(self.mask)+'_z'+str(self.dim_z)+'_y'+str(self.dim_y)+'_s'+str(self.dim_s)+'_batch'+str(self.training()[1]))
        # Create a directoy for the save file
        if not os.path.exists('./saved_networks/' + self.savefile):
            os.makedirs('./saved_networks/' + self.savefile)
        self.network_file_name = './saved_Networks/' + self.savefile + '/' + self.savefile + '.ckpt'
        self.log_file_name = './saved_Network/' + self.savefile + '/log_file_' + self.savefile + '.txt'

    def initialize(self):
        if self.types_dict:
            sess_HVAE = tf.Graph()
            with sess_HVAE.as_default():
                tf_nodes = graph_new.HVAE_graph(self.model_name, self.types_file, self.batch_size, learning_rate=1e-3,
                                                z_dim=self.dim_latent_z, y_dim=self.dim_latent_y,
                                                s_dim=self.dim_latent_s, y_dim_partition=self.dim_latent_y)
                return tf_nodes,sess_HVAE


    def _train(self,epochs,train_data,true_miss_mask,batchsize,types_dict):
        n_batches = int(np.floor(np.shape(self.train_data)[0] / batchsize))
        miss_mask = np.multiply(self.miss_mask, self.true_miss_mask)

        with tf.Session(graph=self.iniatilze()[1]) as session:

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
                if tau2!=None:
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
                train_data_aux[:n_batches * batchsize, :], types_dict)
            est_data_transformed = read_functions.discrete_variables_transformation(est_data, types_dict)
            est_data_imputed = read_functions.mean_imputation(train_data_transformed,
                                                              miss_mask_aux[:n_batches * args.batch_size, :],
                                                              types_dict)

            #            est_data_transformed[np.isinf(est_data_transformed)] = 1e20

            # Create global dictionary of the distribution parameters
            p_params_complete = read_functions.p_distribution_params_concatenation(p_params_list, types_dict,
                                                                                   args.dim_latent_z, args.dim_latent_s)
            q_params_complete = read_functions.q_distribution_params_concatenation(q_params_list, args.dim_latent_z,
                                                                                   args.dim_latent_s)

            # Number of clusters created
            cluster_index = np.argmax(q_params_complete['s'], 1)
            cluster = np.unique(cluster_index)
            print('Clusters: ' + str(len(cluster)))

            # Compute mean and mode of our loglik models
            loglik_mean, loglik_mode = read_functions.statistics(p_params_complete['x'], types_dict)
            #            loglik_mean[np.isinf(loglik_mean)] = 1e20

            # Try this for the errors
            error_train_mean, error_test_mean = read_functions.error_computation(train_data_transformed, loglik_mean,
                                                                                 types_dict, miss_mask_aux[
                                                                                             :n_batches * batchsize,
                                                                                             :])
            error_train_mode, error_test_mode = read_functions.error_computation(train_data_transformed, loglik_mode,
                                                                                 types_dict, miss_mask_aux[
                                                                                             :n_batches * batchsize,
                                                                                             :])
            error_train_samples, error_test_samples = read_functions.error_computation(train_data_transformed,
                                                                                       est_data_transformed, types_dict,
                                                                                       miss_mask_aux[
                                                                                       :n_batches * batchsize, :])
            error_train_imputed, error_test_imputed = read_functions.error_computation(train_data_transformed,
                                                                                       est_data_imputed, types_dict,
                                                                                       miss_mask_aux[
                                                                                       :n_batches * batchsize, :])

            # Compute test-loglik from log_p_x_missing
            log_p_x_total = np.transpose(np.concatenate(log_p_x_total, 1))
            log_p_x_missing_total = np.transpose(np.concatenate(log_p_x_missing_total, 1))
            if self.true_miss_file:
                log_p_x_missing_total = np.multiply(log_p_x_missing_total,
                                                    true_miss_mask_aux[:n_batches * args.batch_size, :])
            avg_test_loglik = np.sum(log_p_x_missing_total) / np.sum(1.0 - miss_mask_aux)



    def training(self,epoch = 10, batchsize = 50):
        train = self._train(epoch,batchsize)
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

        if epoch % self.save == 0:
            print('Saving Variables ...')
            save_path = train.saver.save(train.session, self.network_file_name)

        print('Training Finished ...')

        # Saving needed variables in csv
        if not os.path.exists('./Results' + '/' + self.savefile):
            os.makedirs('./Results' + '/' + self.savefile)

        with open('./Results' + '/' + self.savefile + '/loglik.csv', "w") as f:
            writer = csv.writer(f)
            writer.writerows(train.loglik_epoch)

        with open('./Results' + '/' + self.savefile + '/testloglik.csv', "w") as f:
            writer = csv.writer(f)
            writer.writerows(train.testloglik_epoch)

        with open('./Results' + '/' + self.savefile + '/KL_s.csv', "w") as f:
            writer = csv.writer(f)
            writer.writerows(np.reshape(train.KL_s_epoch, [-1, 1]))

        with open('./Results' + '/' + self.savefile + '/KL_z.csv', "w") as f:
            writer = csv.writer(f)
            writer.writerows(np.reshape(train.KL_z_epoch, [-1, 1]))

        with open('./Results'+ '/' + self.savefile + '/train_error.csv', "w") as f:
            writer = csv.writer(f)
            writer.writerows(train.error_train_mode_global)

        with open('./Results' + '/' + self.savefile + '/test_error.csv', "w") as f:
            writer = csv.writer(f)
            writer.writerows(train.error_test_mode_global)

        # Save the variables to disk at the end
        save_path = train.saver.save(train.session, self.network_file_name)



    def testing(self,epoch=1,batchsize=10000):
        test = self._train()

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

        if not os.path.exists('./Results/' + self.savefile):
            os.makedirs('./Results/' + self.save_file)

        with open('Results/' + self.savefile + '/' + self.savefile + '_data_reconstruction.csv', "w") as f:
            writer = csv.writer(f)
            writer.writerows(data_reconstruction)
        with open('Results/' + self.savefile + '/' + self.savefile + '_data_true.csv', "w") as f:
            writer = csv.writer(f)
            writer.writerows(train_data_transformed)

        # Saving needed variables in csv
        if not os.path.exists('./Results_test_csv/' + self.savefile):
            os.makedirs('./Results_test_csv/' + self.savefile)

        # Train loglik per variable
        with open('Results_test_csv/' + self.savefile + '/' + self.savefile + '_loglik.csv', "w") as f:
            writer = csv.writer(f)
            writer.writerows(test.loglik_epoch)

        # Test loglik per variable
        with open('Results_test_csv/' + self.savefile + '/' + self.savefile + '_testloglik.csv', "w") as f:
            writer = csv.writer(f)
            writer.writerows(test.testloglik_epoch)

        # Train NRMSE per variable
        with open('Results_test_csv/' + self.savefile + '/' + self.savefile + '_train_error.csv', "w") as f:
            writer = csv.writer(f)
            writer.writerows(test.error_train_mode_global)

        # Test NRMSE per variable
        with open('Results_test_csv/' + self.savefile + '/' + self.savefile + '_test_error.csv', "w") as f:
            writer = csv.writer(f)
            writer.writerows(test.error_test_mode_global)

        # Number of clusters
        with open('Results_test_csv/' + self.savefile + '/' + self.savefile + '_clusters.csv', "w") as f:
            writer = csv.writer(f)
            writer.writerows([[len(test.cluster)]])



