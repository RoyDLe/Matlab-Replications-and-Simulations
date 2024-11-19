function [redAR,sigma,sidui,R2,constant,Yhat,ex]=varestimy(nlags,X,k,exog);

% [redAR,sigma,sidui,R2,constant,Yhat,ex]=varestim(nlags,X,k,exog);
% OLS estimation of VAR. 
% nlags     : n. of lags to include. 
% X         :(n*k) panel of time series
% k         : dummy. 1 if you want the constant, 0 if not
% exog      : series of exogenous series (by default is empty).
%
% redAR:    3 dim matrix with the coefficients in the form (I-A(1)L-A(2)L^2-...)
% sigma:    Vcov matrix of the residuals
% sidui:    series of residuals
% R2:       R2 for each equation
% constant: vector of constants for each equations
% ex:       coefficients on the exogenous variables 

if nargin == 3;
    exog=[];
    ex = 'no exogenous in the system';
end
[r c] = size(exog);
aggre2=X;
[obs variab]=size(aggre2);
clear X
regressori = [];
for i = 1:nlags
   %[nlags-i+1 size(aggre2,1)-i]
   regressori = [regressori aggre2(nlags-i+1:size(aggre2,1)-i,:)];
   %size(regressori)   
end

if isempty(exog);
    regressori = regressori;
else
    regressori = [exog(nlags+1:end,:) regressori];
end

if k==1
regressori = [ones(size(aggre2,1)-nlags,1) regressori];
end

    
%[nlags+1 size(aggre2,1)]
Y = aggre2(nlags+1:size(aggre2,1),:);
%be = (inv(regressori'*regressori))*(regressori')*Y;
iXX = inv(regressori'*regressori);
save iXX iXX
be = regressori\Y;
%size(be)
Yhat = regressori*be;
sidui = Y-Yhat;


%%%%%%%%%%%%%%%%%% Coefficents in a 3-D matrix representation 

redAR=zeros(variab,variab,nlags+1);
if k==1
    constant=be(1,:)';
else
   constant='no costant in the model';
end

if isempty(exog) == 0
    ex = be(k+1:c+k,:)';
    % be(k+1:c+k,:)';
end

for i=1:nlags
   %(i-1)*variab+1+1:(i-1)*variab+variab+1
   redAR(:,:,i+1)=-be((i-1)*variab+k+c+1:(i-1)*variab+variab+k+c,:);
   %size(be((i-1)*variab+1+1:(i-1)*variab+variab+1,:))
   redAR(:,:,i+1)=redAR(:,:,i+1)';
end

redAR(:,:,1)=eye(variab);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i=1:variab
   R2(i,1)=1-(sum(sidui(:,i).^2)/sum(center(Y(:,i)).^2));
end

% estimated vcov 

sigma=(1/obs)*sidui'*sidui;
%sigma2=cov(sidui)*(c-1)/c;
