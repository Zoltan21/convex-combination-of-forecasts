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
%% plot and save options
plot_option = 1;
save_option = 1;
%% arima model
arima_params.pq_max = 4;  % maximum p and q of arima model arima(p,d,q)
arima_params.d = 1;
arima_params.seasonality_period = 12;
[yhat_train_arima, yhat_val_arima, yhat_test_arima, best_arima_estimate] = get_arima_forecast(y_struct,time_struct,arima_params,plot_option, save_option);
%%  additive model
additive_params.max_poly_degree = 5;
additive_params.max_Fourier_degree = 7;
additive_params.seasonality_period = 12;
[yhat_train_additive, yhat_val_additive, yhat_test_additive, best_additive_estimate] = get_additive_forecast(y_struct,time_struct,additive_params,plot_option, save_option);

%% plotting all 3 together
plot(time_train, y_train);
hold on
plot(time_train, yhat_train_additive);
plot(time_train, yhat_train_arima);
%% find the least square estimate
n_max = 5;
m_max = 7;
for n=1:n_max
    for m = 1:m_max
        period = 12;
        regressor_function = @(k) get_poly_regressors(n,m,period,k);
        PHI_alpha_train = regressor_function(time_train);
        e_train_add_minus_arima = (yhat_train_additive-yhat_train_arima);
        e_train_true_minus_arima = (y_train-yhat_train_arima);
        %
        objective_fun = @(theta_alpha) sum((e_train_true_minus_arima-e_train_add_minus_arima.*PHI_alpha_train*theta_alpha).^2);
        theta_opt=  fmincon(objective_fun, zeros(min(size(PHI_alpha_train)),1),PHI_alpha_train,ones(length(PHI_alpha_train),1));
        alpha = PHI_alpha_train*theta_opt;
        PHI_alpha_val = regressor_function(time_train);
    end
end
%
figure
yhat_comb = alpha.*yhat_train_additive+(1-alpha).*yhat_train_arima;
yhat_comb_simple = 0.5*yhat_train_additive+0.5*yhat_train_arima;
plot(time_train,y_train)
hold on% plot estimates
plot(time_train, yhat_comb)
plot(time_train, yhat_comb_simple)
plot(time_train, yhat_train_additive)
plot(time_train, yhat_train_arima)
legend('y','ycomb','ycombsimple','yadd','yarim')
MSE_comb = mean((y_train-yhat_comb).^2)
MSE_comb_simple = mean((y_train-yhat_comb_simple).^2)
MSE_add = mean((y_train-yhat_train_additive).^2)
MSE_arima = mean((y_train-yhat_train_arima).^2)
%% validation data
PHI_alpha_val = regressor_function(time_val);
alpha_val = PHI_alpha_val*theta_opt;
yhat_val_comb = alpha_val.*yhat_val_additive+(1-alpha_val).*yhat_val_arima;
yhat_val_comb_simple = 0.5*yhat_val_additive+0.5*yhat_val_arima;
plot(time_val,y_val)
hold on% plot estimates
plot(time_val, yhat_val_comb)
plot(time_val, yhat_val_comb_simple)
plot(time_val, yhat_val_additive)
plot(time_val, yhat_val_arima)
legend('y','ycomb','ycombsimple','yadd','yarim')
MSE_comb = mean((y_val-yhat_val_comb).^2)
MSE_comb_simple = mean((y_val-yhat_val_comb_simple).^2)
MSE_add = mean((y_val-yhat_val_additive).^2)
MSE_arima = mean((y_val-yhat_val_arima).^2)
%% test
%% validation data
PHI_alpha_test = regressor_function(time_test);
alpha_test = PHI_alpha_test*theta_opt;
yhat_test_comb = alpha_test.*yhat_test_additive+(1-alpha_test).*yhat_test_arima;
yhat_test_comb_simple = 0.5*yhat_test_additive+0.5*yhat_test_arima;
plot(time_test,y_test)
hold on% plot estimates
plot(time_test, yhat_test_comb)
plot(time_test, yhat_test_comb_simple)
plot(time_test, yhat_test_additive)
plot(time_test, yhat_test_arima)
legend('y','ycomb','ycombsimple','yadd','yarim')
MSE_comb = mean((y_test-yhat_test_comb).^2)
MSE_comb_simple = mean((y_test-yhat_test_comb_simple).^2)
MSE_add = mean((y_test-yhat_test_additive).^2)
MSE_arima = mean((y_test-yhat_test_arima).^2)





