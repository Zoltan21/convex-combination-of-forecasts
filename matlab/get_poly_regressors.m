function regressors = get_poly_regressors(n,m,period,k)
%GET_POLY_REGRESSORS returns the polynomial trend regressor with Fourier
%for seasonalities
% regressors = get_poly_regressors(n,m,period,k)
% n - maximum polynomial order
% m - maximum Fourier order
% period - the period of the Fourier series
% regressors - a vector with the computed regressors

regressors = [];
% poly
for n_i = 0:n
    regressors = [regressors k.^n_i];
end
for m_i = 1:m
    regressors = [regressors cos(2*pi*m_i*k/period) sin(2*pi*m_i*k/period)];
end