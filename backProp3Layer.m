function [W1, b1, W2, b2, W3, b3, mseValues] = backProp3Layer(trainInputs, trainTargets, learningRate, iterations, architecture)
%   ------- BACKPROPAGATION FUNCTION -------
%   
%   INPUTS:
%   trainInputs = input vector for training set 
%   trainTargets = target vector for training set
%   learning rate = alpha learning rate 
%   iterations = set value for maximum iterations 
%   architecture = parameter not implemented (for different architectures)

%   transfer function definitions (HARDCODED):
%   layer1 function: logsigmoid()
%   layer2 function: logsigmoid()

%   OUTPUTS:
%   W1 = updated weight matrix for layer 1
%   b1 = updated bias vector for layer 1
%   W2 = updated weight matrix for layer 2
%   b2 = updated bias vector for layer 2
%   mseValues = vector of average MSE for each epoch 

% HIDDEN LAYER SIZE HARDCODED HERE 
hiddenLayer1 = 10;
hiddenLayer2 = 10;

% get size of target space to determine output layer size 
[targRows targCols] = size(trainTargets);
outputRows = targRows;

%   Initial Weights and Biases created using small random values  
[trainRows trainCols] = size(trainInputs);

% ---- Layer 1 ----- %
W1 = zeros(hiddenLayer1, trainRows); %creates empty weight matrix
b1 = rand(hiddenLayer1, 1);
for m = 1:hiddenLayer1
    for n = 1:trainRows
        W1(m,n) = rand(1)/10;
    end
end

% ---- Layer 2 FIX W2 SIZE ----- %
%[w1rows w1cols] = size(W1);
W2 = zeros(hiddenLayer2, hiddenLayer1);
b2 = rand(hiddenLayer2, 1);

for m = 1:hiddenLayer2
    for n = 1:trainRows
        W2(m,n) = rand(1)/10;
        %b2(m) = rand(1)/10;
    end
end

% ---- Layer 3 ----- %
%[w2rows w2cols] = size(W2);
W3 = zeros(outputRows, hiddenLayer2);
b3 = rand(outputRows, 1);

for m = 1:outputRows
    for n = 1:hiddenLayer2
        W3(m,n) = rand(1)/10;
        %b2(m) = rand(1)/10;
    end
end


alpha = learningRate;
mseIter = 1;    % just a start value for the while loop 
iters = 1;      % iteration counter 
mseValues = zeros(iterations, 1);   %initialize MSE vector 

% outside loop controls the training iterations 
while( mseIter > 0.0005 && iters < iterations + 1)
    
    mseIter = 0; % cumulative MSE variable for training iteration
    
    % for each input and target in the training sets:
    for passes = 1:trainCols 
        
        % get input and target from their matrices 
        input = trainInputs(:,passes);
        target = trainTargets(:,passes);
        
        %------ Now Propagate Forwards ------%
        a1 = logSigmoid((W1 * input) + b1);
       
        a2 = logSigmoid((W2 * a1) + b2);
        
        a3 = logSigmoid((W3 * a2) + b3);

        error = target - a3; % error vector 
        mseIter = mseIter + mse(target, a3); % add this pair's MSE
        
        %------ Now Calculate Sensitivities and Backpropagate ------%
        
        F2n3 = zeros(outputRows, outputRows);
        for i = 1:outputRows
            for j = 1:outputRows
                if i == j
                    F2n3(i,j) = (1 - a3(i,1))*(a3(i,1));
                end
            end
        end

        
        % create and populate the f2(n2) derivative matrix 
        F2n2 = zeros(hiddenLayer2, hiddenLayer2);
        for i = 1:hiddenLayer2
            for j = 1:hiddenLayer2
                if i == j
                    F2n2(i,j) = (1 - a2(i,1))*(a2(i,1));
                end
            end
        end
        
        % create and populate the f1(n1) derivative matrix
        F2n1 = zeros(hiddenLayer1, hiddenLayer1);
        [f1rows, f1cols] = size(F2n1);
        for i = 1:f1rows
            for j = 1:f1cols
                if i == j
                    F2n1(i,j) = (1 - a1(i,1))*(a1(i,1));
                end
            end
        end
        
        % calculate first sensitivity s^M
        s3 = -2 * F2n3 * error;  % creates vector sized outputRows x 1
        
        % calculate first sensitivity s^M-1
        s2 = F2n2 * W3' * s3;  % creates vector sized outputRows x 1
        
        % calculate next layer's sensitivity s^M-2
        s1 = F2n1 * W2' * s2; %creates vector hiddenLayer x 1 sized 

        %------ Update Weights and Biases ------%
        
        W3 = W3 - (alpha * s3 * a2');
        b3 = b3 - (alpha * s3);

        W2 = W2 - (alpha * s2 * a1');
        b2 = b2 - (alpha * s2);

        W1 = W1 - (alpha * s1 * input');
        b1 = b1 - (alpha * s1);
    end
    mseIter = mseIter/targCols; % get the average for the epoch
    mseValues(iters, 1) = mseIter; % save it to output array 
    iters = iters + 1; % update iters for while loop control 
end

end

