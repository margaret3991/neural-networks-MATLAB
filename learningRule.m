function [W, b] = learningRule(W, b, P, t)
%   The function learningRule implements the perceptron learning 
%   rule by iterating through the input matrix, P, using the 
%   perceptron function and comparing it against the target 
%   values present in the target vector, t. The error is calculated 
%   and then used to adjust W and b for use in the next comparison. 

%   The algorithm stops once it is able to pass through the entire 
%   input matrix without changing W and b, or it has reached the 
%   maximum iterations allowed. The variable maxIter controls the 
%   number of times the algorithm will pass through the input matrix
%   to avoid an infinite loop. 

%   W starts as initial weight matrix
%   b starts as initial bias (likely 0)
%   P = input matrix (each column is an input pattern vector) 
%   t = target vector 

%   Returns W, the adjusted weight matrix and b, the adjusted bias

[rows cols] = size(P);
control = 0; maxIter = 25; iter = 0;
while control < cols+1 && iter < maxIter
    for p = 1:cols
        a = perceptron(W, P(:,p), b); % use the perceptron
        control = control + 1; 
        e = t(p) - a; %calculate error 
        if e ~= 0 % use learning rule to change b and W if e not 0
            b = b + e;
            W = W + (e * P(:,p)');
            control = 0;
        end
    end
    iter = iter + 1; 
    if iter == maxIter 
        disp('Max iterations reached') % tells me it might not be solved
    end
end

