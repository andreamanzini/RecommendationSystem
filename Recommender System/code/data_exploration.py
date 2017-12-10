#%%
from helpers import load_data
from plots import plot_raw_data
from scipy.sparse import csr_matrix
import numpy as np
from scipy import stats
import matplotlib.pyplot as pyplot

#%% 
#%%
path_dataset = '../data/data_train.csv'
ratings = load_data(path_dataset)
#%%
#%%
num_items_per_user, num_users_per_item = plot_raw_data(ratings)
print("min # of items per user = {}, min # of users per item = {}.".format(
        min(num_items_per_user), min(num_users_per_item)))

ratings_csr = ratings.asformat("csr").copy()
ratings_dense = ratings.todense()
ratings_nan = np.array(np.where(ratings_dense!=0,ratings_dense, np.nan))
#%%%

#%%
mean_rating_user_1= ratings_csr.sum(1) / (ratings_csr != 0).sum(1)
mean_rating_user_2= ratings_dense.sum(1)/(ratings_dense!=0).sum(1)
output_filename = '../figures/users_mean_histogram'
fig = pyplot.figure()
ax = fig.add_subplot(1,1,1,)
n, bins, patches = ax.hist(mean_rating_user_1, bins='auto')
pyplot.title('Mean Ratings of Users')
pyplot.xlabel('Mean Ratings')
pyplot.ylabel('Number of Users')
pyplot.savefig(output_filename)
#%%

#%%
mode_rating_user = stats.mode(ratings_nan , 1, 'omit')[0].data
output_filename = '../figures/users_mode_histogram'
fig = pyplot.figure()
ax = fig.add_subplot(1,1,1,)
n, bins, patches = ax.hist(mode_rating_user,bins=5)
pyplot.title('Mode Ratings of Users')
pyplot.xlabel('Mode Ratings')
pyplot.ylabel('Number of Users')
pyplot.xticks([1,2,3,4,5])
pyplot.savefig(output_filename)
#%%

#%%
median_rating_user = np.nanmedian(ratings_nan,1)
output_filename = '../figures/users_median_histogram'
fig = pyplot.figure()
ax = fig.add_subplot(1,1,1,)
n, bins, patches = ax.hist(median_rating_user, bins =30)
pyplot.title('Median Ratings of Users')
pyplot.xlabel('Median Ratings')
pyplot.ylabel('Number of Users')
pyplot.savefig(output_filename)
#%%

#%%
variance_rating_user= np.nanvar(ratings_nan,1)
output_filename = '../figures/users_variance_histogram'
fig = pyplot.figure()
ax = fig.add_subplot(1,1,1,)
n, bins, patches = ax.hist(variance_rating_user, bins ='auto')
pyplot.title('Variance Ratings of Users')
pyplot.xlabel('Variance of Ratings')
pyplot.ylabel('Number of Users')
pyplot.savefig(output_filename)
#%%

#%%
num_items_per_user = np.array((ratings != 0).sum(axis=1)).flatten()
output_filename = '../figures/ratings_per_user_histogram'
fig = pyplot.figure()
ax = fig.add_subplot(1,1,1,)
n, bins, patches = ax.hist(num_items_per_user, bins ='auto')
pyplot.title('Number of Ratings Per User')
pyplot.xlabel('Number of Ratings')
pyplot.ylabel('Number of Users')
pyplot.savefig(output_filename)
#%%

#%%
min_rating_user = np.nanmin(ratings_nan,1)
output_filename = '../figures/min_ratings_users'
fig = pyplot.figure()
ax = fig.add_subplot(1,1,1,)
n, bins, patches = ax.hist(min_rating_user, bins =30)
pyplot.title('Minimum Ratings of Users')
pyplot.xlabel('Minimum Rating')
pyplot.ylabel('Number of Users')
pyplot.savefig(output_filename)
#%%

#%%
max_rating_user = np.nanmax(ratings_nan,1)
output_filename = '../figures/max_ratings_users'
fig = pyplot.figure()
ax = fig.add_subplot(1,1,1,)
n, bins, patches = ax.hist(max_rating_user, bins =30)
pyplot.title('Maximum Ratings of Users')
pyplot.xlabel('Maximum Rating')
pyplot.ylabel('Number of Users')
pyplot.savefig(output_filename)
#%%

#%%
ratings_always_low = np.array(np.where(ratings_dense<3,0,1))
number_user_high_ratings=(ratings_always_low!=0).sum(axis=1)
number_negative_users = (number_user_high_ratings==0).sum()
print("The number of users that only rate movies they dislike is {}".format(number_negative_users))
#%%

#%%
ratings_always_high = np.array(np.where(ratings_dense<=3,1,0))
number_user_low_ratings=(ratings_always_high!=0).sum(axis=1)
number_always_positive_users = (number_user_low_ratings==0).sum()
print("The number of users that only rate movies they like is {}".format(number_always_positive_users))
#%%

#%%
ratio_ones= (ratings_csr==1).sum(1) / (ratings_csr != 0).sum(1)
output_filename = '../figures/users_rating1_ratio_histogram'
fig = pyplot.figure()
ax1 = fig.add_subplot(1,1,1)
n, bins, patches = ax1.hist(ratio_ones, bins='auto')
pyplot.title('Ratio of a 1-rating Per User')
pyplot.xlabel('Ratio of 1-rating')
pyplot.ylabel('Number of Users')
pyplot.savefig(output_filename)
#%%

#%%
ratio_two = (ratings_csr==2).sum(1) / (ratings_csr != 0).sum(1)
output_filename = '../figures/users_rating2_ratio_histogram'
fig = pyplot.figure()
ax2 = fig.add_subplot(1,1,1)
n, bins, patches = ax2.hist(ratio_two, bins='auto')
pyplot.title('Ratio of a 2-rating Per User')
pyplot.xlabel('Ratio of 2-rating')
pyplot.ylabel('Number of Users')
pyplot.savefig(output_filename)
#%%

#%%
ratio_three = (ratings_csr==3).sum(1) / (ratings_csr != 0).sum(1)
output_filename = '../figures/users_rating_ratio3_histogram'
fig = pyplot.figure()
ax = fig.add_subplot(1,1,1)
n, bins, patches = ax.hist(ratio_three, bins='auto')
pyplot.title('Ratio of a 3-rating Per User')
pyplot.xlabel('Ratio of 3-rating')
pyplot.ylabel('Number of Users')
pyplot.savefig(output_filename)
#%%

#%%
ratio_four = (ratings_csr==4).sum(1) / (ratings_csr != 0).sum(1)
output_filename = '../figures/users_rating_ratio4_histogram'
fig = pyplot.figure()
ax = fig.add_subplot(1,1,1)
n, bins, patches = ax.hist(ratio_four, bins='auto')
pyplot.title('Ratio of a 4-rating Per User')
pyplot.xlabel('Ratio of 4-rating')
pyplot.ylabel('Number of Users')
pyplot.savefig(output_filename)
#%%

#%%
ratio_five = (ratings_csr==5).sum(1) / (ratings_csr != 0).sum(1)
output_filename = '../figures/users_rating_ratio5_histogram'
fig = pyplot.figure()
ax = fig.add_subplot(1,1,1)
n, bins, patches = ax.hist(ratio_five, bins='auto')
pyplot.title('Ratio of a 5-rating Per User')
pyplot.xlabel('Ratio of 5-rating')
pyplot.ylabel('Number of Users')
pyplot.savefig(output_filename)
#%%Movies Exploration
