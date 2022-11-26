function [yhat_train, yhat_val, yhat_test, best_model_estimate] = get_arima_forecast(y_struct, time_struct,arima_params, plot_option, save_option)

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

pq_max = arima_params.pq_max;
d = arima_params.d;
seasonality_period = arima_params.seasonality_period;
for pq = 1:pq_max
    %model_architecture(pq) = arima(pq,d,pq);
    model_architecture(pq) = arima('Constant',1,'ARLags',pq,'SARLags',1,'D',d,...
             'Seasonality',seasonality_period,'MALags',pq,'SMALags',1);
    [model_estimate,~,logL(pq)] = estimate(model_architecture(pq),y_train);
    results = summarize(model_estimate);
    numParam(pq) = results.NumEstimatedParameters;
end
aic = aicbic(logL,numParam,N_train);
[~,min_aic_index] = min(aic);
%%
best_model_estimate= estimate(model_architecture(min_aic_index),y_train);
%% compute forecasts
yhat_val = forecast(best_model_estimate,N_val,y_train);
yhat_test = forecast(best_model_estimate,N_test,[y_train; y_val]);
res_train_arima = infer(best_model_estimate,y_train);
yhat_train = y_train-res_train_arima;
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
    yhat_train_arima = yhat_train;
    yhat_val_arima = yhat_val;
    yhat_test_arima = yhat_test;
    save ../data/yhat_arima yhat_train_arima yhat_val_arima yhat_test_arima
end

end