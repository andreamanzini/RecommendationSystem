# -*- coding: utf-8 -*-
"""some functions for plots."""

import numpy as np
import matplotlib.pyplot as plt
import scipy

def plot_traintest_over_factors(factors_list, tot_rmse_tr, tot_rmse_te, filepath):
    """ Plot test and train RMSE over factors in a smooth graph """
    # Interpolate results for smoother graph
    interval = factors_list[1]-factors_list[0]
    interp_tr = scipy.interpolate.interp1d(factors_list, tot_rmse_tr, kind='cubic')
    interp_te = scipy.interpolate.interp1d(factors_list, tot_rmse_te, kind='cubic')
    x_smooth = np.arange(factors_list[0], factors_list[-1]+1)
    plt.plot(x_smooth, interp_tr(x_smooth), '.-', x_smooth, interp_te(x_smooth), '.-', markevery=interval);
    plt.title('SVD over different numbers of factors');
    plt.xticks(np.arange(0,factors_list[-1]+1,interval))
    plt.xlabel('N Factors');
    plt.ylabel('RMSE');
    plt.legend(['train','test']);
    plt.grid()
    plt.savefig(filepath);
    
def plot_train_over_factors(factors_list, tot_rmse_tr, filepath):
    """ Plot train RMSE over factors in a smooth graph """
    # Interpolate results for smoother graph
    interval = factors_list[1]-factors_list[0]
    interp_tr = scipy.interpolate.interp1d(factors_list, tot_rmse_tr, kind='cubic')
    x_smooth = np.arange(factors_list[0], factors_list[-1]+1)
    plt.plot(x_smooth, interp_tr(x_smooth), '.-', x_smooth, markevery=interval);
    plt.title('SVD over different numbers of factors (train)');
    plt.xticks(np.arange(0,factors_list[-1]+1,interval))
    plt.xlabel('N Factors');
    plt.ylabel('RMSE Train');
    plt.grid()
    plt.savefig(filepath);
    
def plot_test_over_factors(factors_list, tot_rmse_te, filepath):
    """ Plot test RMSE over factors in a smooth graph """
    # Interpolate results for smoother graph
    interval = factors_list[1]-factors_list[0]
    interp_te = scipy.interpolate.interp1d(factors_list, tot_rmse_te, kind='cubic')
    x_smooth = np.arange(factors_list[0], factors_list[-1]+1)
    plt.plot(x_smooth, interp_te(x_smooth), '.-', x_smooth, markevery=interval);
    plt.title('SVD over different numbers of factors (test)');
    plt.xticks(np.arange(0,factors_list[-1]+1,interval))
    plt.xlabel('N Factors');
    plt.ylabel('RMSE Test');
    plt.grid()
    plt.savefig(filepath);


def plot_raw_data(ratings):
    """plot the statistics result on raw rating data."""
    # do statistics.
    num_items_per_user = np.array((ratings != 0).sum(axis=1).T).flatten()
    num_users_per_item = np.array((ratings != 0).sum(axis=0)).flatten()
    sorted_num_movies_per_user = np.sort(num_items_per_user)[::-1]
    sorted_num_users_per_movie = np.sort(num_users_per_item)[::-1]

    # plot
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(sorted_num_movies_per_user, color='blue')
    ax1.set_xlabel("users")
    ax1.set_ylabel("number of ratings (sorted)")
    ax1.grid()

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(sorted_num_users_per_movie)
    ax2.set_xlabel("items")
    ax2.set_ylabel("number of ratings (sorted)")
    ax2.set_xticks(np.arange(0, 1000, 300))
    ax2.grid()

    plt.tight_layout()
    plt.savefig("stat_ratings")
    plt.show()
    # plt.close()
    return num_items_per_user, num_users_per_item


def plot_train_test_data(train, test):
    """visualize the train and test data."""
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.spy(train, precision=0.01, markersize=0.5)
    ax1.set_xlabel("Users")
    ax1.set_ylabel("Items")
    ax1.set_title("Training data")
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.spy(test, precision=0.01, markersize=0.5)
    ax2.set_xlabel("Users")
    ax2.set_ylabel("Items")
    ax2.set_title("Test data")
    plt.tight_layout()
    plt.savefig("train_test")
    plt.show()
