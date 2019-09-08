function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

hypotesis=sigmoid(X*theta);
J=(1/m)*((-y')*log(hypotesis)-((1-y)')*log(1-hypotesis));
%[J, grad] = costFunction(theta, X, y)
sum1=0;
theta = [0 ; theta(2:size(theta), :)];
%theta(0,:)=0;
%sum2=0;
%for j=2:length(theta),
 % sum1=sum1+theta(j)^2;
%J=J+(sum1*lambda)/(2*m);
J=J+(theta'*theta)*lambda/(2*m);
%grad=grad+lambda*theta/m;
grad=(X'*(hypotesis-y)+lambda*theta)/m;



% =============================================================

end
