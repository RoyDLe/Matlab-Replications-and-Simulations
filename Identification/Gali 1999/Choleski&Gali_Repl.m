%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Exercise 1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc
clear

%Loading Data
data = readtable('data_ps3.xlsx');
logGDP = data.LOG_GDP_;
logP = data.LOG_P_;
FFR = data.FFR;
Y = [logGDP, logP, FFR];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
hor = 50;                   %Horizon over which to compute impulse responses.
K = 1000;                   %Number of bootstrap iterations.
lag_number = 4;             %Number of lags to include in the VAR.
N = width(Y);               %Number of variables and shocks.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Estimate VAR Model
[redAR, ~, sidui, ~, constant, ~, ~] = varestimy(lag_number, Y, 1);
                                                                              
%% Repeating Steps 2, 3, and 4 K Times
all_irfs = zeros(N,N,hor,K);
all_var_decomp = zeros(N,N,hor,K);
for k = 1:K
%% Step 2: Resampling Residuals
    row_residuals = size(sidui, 1);
    PER = randi(row_residuals, row_residuals, 1); % resample of residuals with replacement
    epsilon_tilde = sidui(PER, :);  
    
%% Step 3: Generating New Time Series
    row_new = size(epsilon_tilde, 1);
    Y_tilde = zeros(row_new, size(Y, 2)); 
    
    % Starting values is the first rows of the original Y
    Y_tilde(1:lag_number, :) = Y(1:lag_number, :);
    
    % Generating the new series
    for t = (lag_number + 1):row_new
        Y_tilde(t, :) = constant';
        % adding lags
        for p = 1:lag_number
            Y_tilde(t, :) = Y_tilde(t, :) + (Y_tilde(t - p, :) * (-redAR(:,:,p+1)'));
        end
        % adding the resampled residual for time t
        Y_tilde(t, :) = Y_tilde(t, :) + epsilon_tilde(t, :);
    end
    
%% Step 4: Estimating new VAR, IRFs and Variance decomposition
    [redAR_boot, sigma_boot, ~, ~, ~, ~, ~] = varestimy(lag_number, Y_tilde, 1);
    %Computing IRFs from companion form
    [~, N] = size(Y);
    AA = [];
    for t = 2:lag_number+1
        AA = [AA, squeeze(-redAR_boot(:,:,t))];
    end
    jota = [AA ; [eye(N*(lag_number-1),N*(lag_number-1)) zeros(N*(lag_number-1),N)]];
    
    %Triangular identification
    G = chol(sigma_boot, 'lower');

    irf = zeros(N, N, hor);  
    for s=1:hor
        IR_j = jota^(s-1);
        irf(:, :, s) = IR_j(1:N, 1:N,:)*G;
    end
    all_irfs(:, :, :, k) = irf;
 
    %% Variance decomposition

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Note: this part is very computationally intensive (any alternatives?)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    var_decomp_k = zeros(N, N, hor);
    for h = 1:hor
        FE_cov = zeros(N, N);
        for t = 1:h
            FE_cov = FE_cov + irf(:, :, t) * sigma_boot * (irf(:, :, t))';
        end
        var_tot = diag(FE_cov);
        for j = 1:N
            Gj = G(:, j); %j-th column of Cholesky factor
            contrib_j = zeros(N, 1);
            for t = 1:h
                Psi_t = irf(:, :, t);
                contrib_j = contrib_j + diag(Psi_t * (Gj) * Gj' * Psi_t');
            end
            var_decomp_k(:, j, h) = contrib_j ./ var_tot;
        end
    end
    all_var_decomp(:, :, :, k) = var_decomp_k;
end
%% Step 5: CIs and plots
[num_variables, num_shocks, horizon, K] = size(all_irfs);
time = 0:hor-1;

mean_irf = zeros(num_variables, num_shocks, horizon);
lower_bound = zeros(num_variables, num_shocks, horizon);
upper_bound = zeros(num_variables, num_shocks, horizon);

% Compute mean IRF and confidence bands
for var = 1:num_variables
    for shock = 1:num_shocks
        irfs_for_shock = squeeze(all_irfs(var, shock, :, :)); 
        mean_irf(var, shock, :) = mean(irfs_for_shock, 2);
        lower_bound(var, shock, :) = prctile(irfs_for_shock, 2.5, 2);
        upper_bound(var, shock, :) = prctile(irfs_for_shock, 97.5, 2);
    end
end
%Finally: plotting the results
figure('Position', [100, 100, 1440, 1080]);
mean_var_decomp = mean(all_var_decomp, 4); 
lower_var_decomp = prctile(all_var_decomp, 2.5, 4);
upper_var_decomp = prctile(all_var_decomp, 97.5, 4); 

for var = 1:num_variables
    for shock = 1:num_shocks
        subplot(num_variables, num_shocks, (var-1)*num_shocks + shock);
        hold on;
    
        plot(time, squeeze(mean_irf(var, shock, :)), 'b', 'LineWidth', 1.5);
        
        fill([time, fliplr(time)], ...
             [squeeze(lower_bound(var, shock, :))', fliplr(squeeze(upper_bound(var, shock, :))')], ...
             'r', 'FaceAlpha', 0.3, 'EdgeColor', 'none');
        
        title(['Response of Variable ', num2str(var), ' to Shock ', num2str(shock)]);
        xlabel('Time');
        ylabel('Response');
        grid on;
        hold off;
    end
end
%Plotting variance decompositions
figure('Position', [100, 100, 1440, 1080]);
for var = 1:num_variables
    subplot(ceil(num_variables / 2), 2, var); 
    hold on;
    for shock = 1:num_shocks
        plot(time, squeeze(mean_var_decomp(var, shock, :)), 'LineWidth', 1.5, ...
            'DisplayName', ['Shock ', num2str(shock)]);
    end
    title(['Contribution to variance of variable ', num2str(var)]);
    xlabel('Horizon');
    ylabel('Contribution');
    grid on;
    legend('show');
    hold off;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Exercise 2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc
clear

% Loading the data
[data, ~, ~] = xlsread('data_ps3.xlsx', 2);
log_GDP = data(:, 1);
hrs = data(:, 2);
prod = log_GDP - log(hrs);
Y = [diff(prod), 100*diff(log(hrs))];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
numLags = 4;            %Number of lags in the VAR.
nvar = width(Y);        %Number of variables.
numBootstrap = 1000;    %Number of bootstrap iterations.
alpha = 0.05;           %Significance level for the confidence bands
IRF_Horizon = 12;       %Horizon over which to compute the impulse responses.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Estimate VAR
[redAR, sigma, sidui, R2, constant, Yhat, ex] = varestimy(numLags, Y, 1);

%% Identification
summat = eye(size(redAR, 1));
for k = 2:size(redAR, 3)
    summat = summat + squeeze(redAR(:, :, k));
end
S = inv(summat);
D_1 = chol(S * sigma * S')';

summat2 = eye(size(redAR, 1));
for l = 2:size(redAR, 3)
    summat2 = summat2 + redAR(:, :, l);
end
G = summat2 * D_1;

%% Companion form
[~, N] = size(Y);
AA = [];
for t = 2:numLags+1
    AA = [AA, squeeze(-redAR(:,:,t))];
end
jota = [AA ; [eye(N*(numLags-1),N*(numLags-1)) zeros(N*(numLags-1),N)]];

%% Impulse responses
SVAR_IRF = zeros(2, 2, IRF_Horizon);  
for s=1:IRF_Horizon
    IR_j = jota^(s-1);
    SVAR_IRF(:, :, s) = IR_j(1:N, 1:N,:)*G;
end
%Convert to cumulative IRFs.
Cumulative_IRF = cumsum(SVAR_IRF, 3);

%% Bootstrap cumulative IRFs. 
[lower_bound, upper_bound] = bootstrap_cumulative_IRF(Y, numLags, redAR, constant, sidui, IRF_Horizon, numBootstrap, alpha);

%Back out the IRF for GDP by adding back hours.
IR_GDP = Cumulative_IRF(1,:,:) + Cumulative_IRF(2,:,:);
IR_GDP_lower = lower_bound(1,:,:) + lower_bound(2,:,:);
IR_GDP_upper = upper_bound(1,:,:) + upper_bound(2,:,:);

%% Plot cumulative IRFs with confidence bands.
figure('Position', [100, 100, 1440, 1080]);
axesHandles = gobjects(nvar + 1, 2); 
for i = 1:nvar +1
    for j = 1:2
        axesHandles(i, j) = subplot(nvar + 1, 2, (i - 1) * 2 + j);
    end
end
for i = 1:nvar
    for j = 1:nvar
        axes(axesHandles(i, j));
        plot(0:IRF_Horizon -1, squeeze(Cumulative_IRF(i, j, :)), '-o', 'Color', 'black');
        hold on;
        plot(0:IRF_Horizon - 1, squeeze(lower_bound(i, j, :)), '-^', 'Color', 'black');
        plot(0:IRF_Horizon - 1, squeeze(upper_bound(i, j, :)), '-^', 'Color', 'black');
        
        % Set titles based on the position
        if i == 1
            var_name = "Productivity";
        else
            var_name = "Hours";
        end
        if j == 1
            title('Technology Shock');
            xlabel(var_name)
        else
            title('Non-Technology Shock');
            xlabel(var_name)
        end
        grid on;
    end
end
axes(axesHandles(3, 1));
hold on
plot(0:IRF_Horizon -1,squeeze(IR_GDP(1,1,:)), '-o', Color = "black");
plot(0:IRF_Horizon -1,squeeze(IR_GDP_lower(1,1,:)),'-^', Color = "black");
plot(0:IRF_Horizon -1,squeeze(IR_GDP_upper(1,1,:)),'-^', Color = "black");
title('Technology Shock'); xlabel('GDP'); grid on;
hold off
axes(axesHandles(3, 2));
hold on
plot(0:IRF_Horizon -1,squeeze(IR_GDP(1,2,:)),'-o', Color = "black");
plot(0:IRF_Horizon -1,squeeze(IR_GDP_lower(1,2,:)),'-^', Color = "black");
plot(0:IRF_Horizon -1,squeeze(IR_GDP_upper(1,2,:)),'-^', Color = "black");
title('Non-Technology Shock'); xlabel('GDP'); grid on;
hold off;
sgtitle('Cumulative IRFs with 95% Confidence Intervals (based on Gali, 1999)');
hold off 