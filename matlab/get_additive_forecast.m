function [yhat_train, yhat_val, yhat_test, best_model_estimate] = get_additive_forecast(y_struct, time_struct,additive_params, plot_option, save_option)

y = y_struct.y;
y_train = y_struct.y_train;
y_val = y_struct.y_val;
y_test = y_struct.y_test;

time = time_struct.y;
time_train = time_struct.time_train;
time_val = time_struct.time_val;
time_test = time_struct.time_test;

N_train = length(y_train);
N_val = length(y_val);
N_test = length(y_test);

%% additive model params
max_poly_degree = additive_params.max_poly_degree;
max_Fourier_degree = additive_params.max_Fourier_degree;
period = additive_params.seasonality_period;

%% MSE vectors initialization
MSE_train = zeros(max_poly_degree, max_Fourier_degree);
MSE_val = zeros(max_poly_degree, max_Fourier_degree);
%% approximate with the regressor
for n=1:max_poly_degree
    for m=1:max_Fourier_degree
        % train
        regressor_function = @(k) get_poly_regressors(n,m,period,k);
        PHI_train = regressor_function(time_train);
        theta = PHI_train\y_train;
        yhat_train = PHI_train*theta;
        MSE_train(n, m) =mean((y_train-yhat_train).^2);
        % validation
        PHI_val = regressor_function(time_val);
        yhat_val = PHI_val*theta;
        MSE_val(n, m) = mean((y_val-yhat_val).^2);
    end
end
%% optimal solution
[n_optim, m_optim]=find(MSE_val==min(min(MSE_val)),1,'first');
% train
regressor_function = @(k) get_poly_regressors(n_optim,m_optim,period,k);
PHI_train = regressor_function(time_train);
theta = PHI_train\y_train;
yhat_train = PHI_train*theta;
% validation
PHI_val = regressor_function(time_val);
yhat_val = PHI_val*theta;
% test
PHI_test = regressor_function(time_test);
yhat_test = PHI_test*theta;
best_model_estimate.regressor_function = @(k) regressor_function(k);
best_model_estimate.params = theta;
%% plot optimal solution
if plot_option
    figure
    plot(time, y,'b')
    hold on
    plot(time_train, yhat_train,'r');
    plot(time_val, yhat_val,'g')
    plot(time_test, yhat_test)
end
%% save outputs
if save_option
    yhat_train_additive = yhat_train;
    yhat_val_additive = yhat_val;
    yhat_test_additive = yhat_test;
    save ../data/yhat_additive yhat_train_additive yhat_val_additive yhat_test_additive
end