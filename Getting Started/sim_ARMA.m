function [Y, epsilon] = sim_ARMA(T, sigma2, AR_roots, MA_roots)
    AR_coeff = poly(AR_roots);
    MA_coeff = poly(MA_roots);

    AR_coeff = AR_coeff(2:end); %We do not need the leading 1 of the polynomial to compute AR/MA components.
    MA_coeff = MA_coeff(2:end);

    epsilon = sqrt(sigma2) * randn(T,1);
    Y = zeros(T,1);

    for t =1:T
        AR_component = 0;
        for p=1:min(t-1, length(AR_coeff)) %min() ensures we loop only over available data points.
            AR_component = AR_component - AR_coeff(p) * Y(t-p);
        end
        MA_component = 0;
        for q = 1:min(t-1, length(MA_coeff))
            MA_component = MA_component + MA_coeff(q) * epsilon(t-q);
        end
        Y(t) = AR_component + epsilon(t) + MA_component;
    end
end


