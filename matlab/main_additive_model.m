clear
close all
load('..\data\matfiles\product_20.mat')
%% split data to id and val
N =length(y);
N_val= 12;
N_test = 6;
N_train = N-N_val-N_test;
% train
y_train = y(1:N_train);
time_train = time(1:N_train);
% val
y_val = y(N_train+1:N_train+N_val);
time_val = time(N_train+1:N_train+N_val);
% test
y_test = y(N_train+N_val+1:end);
time_test = time(N_train+N_val+1:end);
% make structure
y_struct.y = y;
y_struct.y_train = y_train;
y_struct.y_val = y_val;
y_struct.y_test = y_test;

time_struct.y = time;
time_struct.time_train = time_train;
time_struct.time_val = time_val;
time_struct.time_test = time_test;
%% model hyperparams
additive_params.max_poly_degree = 5;
additive_params.max_Fourier_degree = 7;
additive_params.seasonality_period = 12;
%% plot and save options
plot_option = 1;
save_option = 1;
%% get the model estimates
[yhat_train_additive, yhat_val_additive, yhat_test_additive, best_additive_estimate] = get_additive_forecast(y_struct,time_struct,additive_params,plot_option, save_option);
