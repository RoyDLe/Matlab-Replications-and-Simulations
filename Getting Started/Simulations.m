%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%AR(1) using loop and filter
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng(100);
T = 500;
phi = 0.4;
sigma2 = 0.2;

%White noise error term
epsilon = sqrt(sigma2) * randn(T,1);

X = zeros(T, 1);
X(1)=epsilon(1);
for t = 2:T
    X(t) = phi * X(t-1) + epsilon(t);
end

%%%%%% 1b) using filter()%%%%%%
Y = filter(1, [1 -phi], epsilon); %use same epsilon.
Y(1)=epsilon(1);

figure; plot(X, 'DisplayName','Loop');
hold on
plot(Y, 'DisplayName','Filter');
title('AR(1): Loop vs Filter');
xlabel('Time')
ylabel('Value')
legend;
hold off

%Plot the difference series
figure;plot(X -Y);
title('Difference between X(loop) and Y(Filter)');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%AR(1) with positive mean and starting value
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clearvars -global;

rng(1);
T = 200;
sigma2 = 0.4;
mu = 3;
init = 20;
phi = 0.6;

c = mu*(1-phi);

%Using a for loop
epsilon = sqrt(sigma2) * randn(T,1); %Generate White Noise
Y = zeros(T, 1);
Y(1) = init;
for t = 2:T
    Y(t) = c + phi * Y(t-1) + epsilon(t);
end
figure;plot(Y);
title('AR(1) with non-zero starting value.');
xlabel('Time');
ylabel('Y');
grid on;

%Check the mean
mean(Y)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%MA(1) using loop and filter
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clearvars -global;

T = 500;
sigma2 = 0.3;
theta = 0.3;

rng(1);
epsilon = sqrt(sigma2) * randn(T,1);
Y1  = zeros(T,1); %No need to set starting value, since its already zero.
for t = 2:T
    Y1(t) = epsilon(t) + theta * epsilon(t-1);
end
Y1(1)=epsilon(1);

figure;plot(Y1);
title('MA(1)');
xlabel('Time');
ylabel('Y');

hold on 

Y2 = filter([1 theta], 1, epsilon); %Use the same epsilon with filter
Y2(1)=epsilon(1);

plot(Y2);
legend('MA with for loop', 'MA with filter');

hold off

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%ARMA(p,q) from roots of AR /MA polynomials.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clearvars -global;

sigma2 = 0.4;
T = 200;

%Define AR/MA polynomials.
AR = [1 -0.5 -0.3 -0.1];
MA = [1 0.6 0.2];

%Convert them to roots since function takes roots.
AR_rt = roots(AR);
MA_rt = roots(MA);

rng(1)
[y,epsilon] = sim_ARMA(T, sigma2, AR_rt, MA_rt);

figure;plot(y);
title('ARMA');
xlabel('Time');
ylabel('Y');

hold on

%Check our output with built-in ARIMA function.
rng(1)
arma = arima('Constant', 0, 'AR', -AR(2:end), 'MA', MA(2:end), 'Variance',sigma2);
sim = simulate(arma,T);

plot(sim);
plot(epsilon);
legend('Our Function', 'ARIMA built-in function', 'White Noise');

hold off

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Empirical distribution of the OLS estimator of an AR(1) - Monte Carlo
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clearvars -global;
T = 250;
phi = 0.4;
sigma2 = 1;
N = 1000;
rejections = 0;

h0 = 0; %null hypothesis.
[phihat, c_hat] = deal(zeros(N,1));
rng(100)
for sim = 1:N
    epsilon = sqrt(sigma2) * randn(T,1);
    Y = zeros(T,1);
    for t=2:T
        Y(t) = phi *Y(t-1) + epsilon(t);
    end
    X = [ones(T-1, 1), Y(1:T-1)];
    y = Y(2:T);

    %Run the regression with intercept.
    beta_ols = X\y; %beta_ols solves (X'X)b = (X'y).
    c_hat(sim) = beta_ols(1);
    phihat(sim) = beta_ols(2);

    e = y - X*beta_ols; %Compute the residuals.

    s2 = (e'*e)/(T-1-width(X)); %Unbiased estimate of sigma^2 (we lose 1 observation). 
    Sigma = s2 * inv(X'*X);

    se_phihat = sqrt(Sigma(2,2)); %variance of phihat is the second diagonal element of Sigma.
    %se_c = sqrt(Sigma(1,1)); variance of intercept is the first diagonal element of Sigma.

    t_phihat = (beta_ols(2)-h0) / se_phihat;

    if abs(t_phihat)>1.96
        rejections = rejections + 1;
    end
end
%Plot histogram of the distribution of the estimators.
figure;phi_hist = histogram(phihat,50, 'FaceColor', [0, 0.4470, 0.7410], 'FaceAlpha', 0.5);
hold on 
c_hist = histogram(c_hat,50, 'FaceColor',[0.8500, 0.3250, 0.0980], 'FaceAlpha', 0.5);
hold off
legend({sprintf('phi'), ...
        sprintf('intercept')}, 'Location', 'northwest');
(rejections/N)*100

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Empirical distribution of the OLS estimator of an AR(1) with increasing T
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clearvars -global;

T=[50, 100, 200, 1000];
phi = 0.9;
N = 1000;
sigma2 = 0.8;

% Pre-allocate list of output vectors
output = cell(1,length(T));
for i=1:length(T)
    output{i} = zeros(1,N);
end

rng(1)
for i=1:length(T)
    phihat = zeros(N,1);
    for sim=1:N
        epsilon = sqrt(sigma2) * randn(T(i), 1);
        Y = zeros(T(i), 1);
        for t=2:T(i)
            Y(t) = phi * Y(t-1) + epsilon(t);
        end
        X = [ones(T(i)-1, 1), Y(1:T(i)-1)];
        y = Y(2:end);

        beta_ols = X\y;
        phihat(sim) = beta_ols(2);
    end
    output{i} = phihat;
end
%Overlay histograms on one graph.
figure;histogram(output{1}, 'FaceColor', [0, 0.4470, 0.7410], 'FaceAlpha', 0.5);
hold on 
histogram(output{2}, 'FaceColor', [0.8500, 0.3250, 0.0980], 'FaceAlpha', 0.4);
histogram(output{3}, 'FaceColor', [0.9290, 0.6940, 0.1250], 'FaceAlpha', 0.4);
histogram(output{4}, 'FaceColor', [0.4940, 0.1840, 0.5560], 'FaceAlpha', 0.4);
legend({sprintf('Histogram for phi with T=%.2f', T(1)), ...
        sprintf('Histogram for phi with T=%.2f', T(2)), ...
        sprintf('Histogram for phi with T=%.2f', T(3)), ...
        sprintf('Histogram for phi with T=%.2f', T(4))}, 'Location', 'northwest');
hold off

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Empirical distribution of OLS when DGP is MA(1) but we mis-specify as AR(1)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clearvars -global;

theta = 0.6;
sigma2 = 0.5;
N = 1000;
steps = 5;
step_size = 250;

rng(1);
output = cell(1,steps);
for i=1:steps
    output{i} = zeros(1,N);
end
for i=1:steps
    a_hat = zeros(N, 1);
    for sim=1:N
        epsilon = sqrt(sigma2) * randn([step_size* 2^(i-1),1]);
        x = zeros(step_size* 2^(i-1),1); 
        for t = 2:step_size* 2^(i-1)
            x(t) = epsilon(t) + theta * epsilon(t-1);
        end
        X = x(1: (step_size* 2^(i-1)) -1);
        Y = x(2:end);
        a_ols = X \ Y;

        a_hat(sim) = a_ols;
    end
    output{i} = a_hat;
end
figure;
hold on
legendEntries = cell(1,steps);
for j=1:steps
    mean(output{j})
    [f,xi] = ksdensity(output{j});
    plot(xi, f, 'LineWidth', 2);
    legendEntries{j} = sprintf('T: %d', step_size* 2^(j-1));
end
legend(legendEntries, 'Location', 'best');
title('Kernel Density Estimates for Different T');
xlabel('Estimated coefficient');
ylabel('Density');

hold off
















            













