function [lower_bound, upper_bound] = bootstrap_cumulative_IRF(Y, numLags, coeffs, constant, resids, IRF_Horizon, numBootstrap, alpha)
    % BOOTSTRAP_CUMULATIVE_IRF Generates bootstrapped confidence intervals for cumulative IRFs.
    % Inputs:
    %   Y             - Endogenous variables (matrix of size T × nvar).
    %   numLags       - Number of lags in the VAR model.
    %   coeffs        - VAR coefficients (nvar × nvar × (numLags + 1)).
    %   constant      - Constant terms (vector of size nvar × 1).
    %   resids        - Residuals of the original VAR model.
    %   IRF_Horizon   - Horizon over which to compute the impulse responses.
    %   numBootstrap  - Number of bootstrap iterations.
    %   alpha         - Significance level for CIs
    % Outputs:
    %   lower_bound   - Lower bounds of the confidence intervals (nvar × nvar × IRF_Horizon).
    %   upper_bound   - Upper bounds of the confidence intervals (nvar × nvar × IRF_Horizon).
    
    % Number of variables
    nvar = size(Y, 2);
    
    % Initialize array to store bootstrap IRFs
    bootstrap_IRFs = zeros(nvar, nvar, IRF_Horizon, numBootstrap);
    
    % Perform bootstrapping
    for b = 1:numBootstrap
        % Resample residuals with replacement
        resampled_residuals = resids(randi(size(resids, 1), size(resids, 1), 1), :);
    
        % Generate bootstrap sample
        Y_bootstrap = zeros(size(Y));
        Y_bootstrap(1:numLags, :) = Y(1:numLags, :);  % Preserve initial lags
        for t = numLags + 1:size(Y, 1)
            lagged_contrib = zeros(1, nvar);  % Initialize lagged contributions
            for lag = 1:numLags
                lagged_contrib = lagged_contrib + Y_bootstrap(t - lag, :) * (-coeffs(:, :, lag + 1))';
            end
            Y_bootstrap(t, :) = constant' + lagged_contrib + resampled_residuals(t - numLags, :);
        end
    
        % Re-estimate VAR on bootstrap sample
        [redAR_boot, sigma_boot, ~, ~, ~, ~, ~] = varestimy(numLags, Y_bootstrap, 1);
    
        % Recompute the IRFs for the bootstrap sample
        AA_boot = [];
        for t = 2:numLags + 1
            AA_boot = [AA_boot, squeeze(-redAR_boot(:, :, t))];
        end
        jota_boot = [AA_boot; [eye(nvar * (numLags - 1)), zeros(nvar * (numLags - 1), nvar)]];
        summat_boot = eye(nvar);
        for k = 2:size(redAR_boot, 3)
            summat_boot = summat_boot + squeeze(redAR_boot(:, :, k));
        end
        S_boot = inv(summat_boot);
        D_1_boot = chol(S_boot * sigma_boot * S_boot')';
        summat2_boot = eye(nvar);
        for l = 2:size(redAR_boot, 3)
            summat2_boot = summat2_boot + redAR_boot(:, :, l);
        end
        G_boot = summat2_boot * D_1_boot;
    
        IRF_boot = zeros(nvar, nvar, IRF_Horizon);
        for s = 1:IRF_Horizon
            IR_j_boot = jota_boot^(s - 1);
            IRF_boot(:, :, s) = IR_j_boot(1:nvar, 1:nvar) * G_boot;
        end
    
        % Store cumulative IRFs for bootstrap sample
        bootstrap_IRFs(:, :, :, b) = cumsum(IRF_boot, 3);
    end
    
    % Compute confidence intervals
    lower_bound = prctile(bootstrap_IRFs, 100 * alpha / 2, 4);
    upper_bound = prctile(bootstrap_IRFs, 100 * (1 - alpha / 2), 4);

    end
    