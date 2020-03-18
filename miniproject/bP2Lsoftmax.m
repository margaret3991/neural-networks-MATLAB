function [W1, b1, W2, b2, mseValues, accValues] = bP2Lsoftmax(trainInputs, trainTargets, learningRate, iterations, testInputs, testTargets, testLabels)
%   ------- BACKPROPAGATION FUNCTION -------
%   
%   INPUTS:
%   trainInputs = input vector for training set 
%   trainTargets = target vector for training set
%   learning rate = alpha learning rate 
%   iterations = set value for maximum iterations 
%   architecture = parameter not implemented (for different architectures)

%   transfer function definitions (HARDCODED):
%   layer1 function: ReLU()
%   layer2 function: ReLU()

%   OUTPUTS:
%   W1 = updated weight matrix for layer 1
%   b1 = updated bias vector for layer 1
%   W2 = updated weight matrix for layer 2
%   b2 = updated bias vector for layer 2
%   W3 = updated weight matrix for layer 3
%   b3 = updated bias vector for layer 3 
%   mseValues = vector of average MSE for each epoch 

% HIDDEN LAYER SIZE HARDCODED HERE 
hiddenLayer1 = 40;
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

% ---- Layer 2 ----- %
%[w1r w1c] = size(W1);
W2 = zeros(outputRows, hiddenLayer1);
b2 = rand(outputRows, 1);

for m = 1:outputRows
    for n = 1:hiddenLayer1
        W2(m,n) = rand(1)/10;
        %b2(m) = rand(1)/10;
    end
end


alpha = learningRate;
mseIter = 1;    % just a start value for the while loop 
iters = 1;      % iteration counter 
mseValues = zeros(iterations, 1);   %initialize MSE vector
accValues = zeros(max((iterations/50),1), 1);

% outside loop controls the training iterations 
while( mseIter > 0.005 && iters < iterations + 1)
    
    mseIter = 0; % cumulative MSE variable for training iteration
    
    % for each input and target in the training sets:
    for passes = 1:trainCols 
        
        % get input and target from their matrices 
        input = trainInputs(:,passes);
        target = trainTargets(:,passes);
        
        %------ Now Propagate Forwards ------%
        a1 = logSigmoid((W1 * input) + b1);

        a2 = softmax((W2 * a1) + b2);

        error = target - a2; % error vector 
        mseIter = mseIter + mse(target, a2); % add this pair's MSE
        
        %------ Now Calculate Sensitivities and Backpropagate ------%
        
        % create and populate the f2(n2) derivative matrix 
        F2n2 = zeros(hiddenLayer2, hiddenLayer2);
        for i = 1:hiddenLayer2
            for j = 1:hiddenLayer2
                 if i == j
                    F2n2(i,j) = (1 - a2(i,1))*(a2(i,1));
                else
                    F2n2(i,j) = -a2(i,1) * a2(j,1);
                end
            end
        end
        
        F2n1 = zeros(hiddenLayer1, hiddenLayer1);
        [f1rows f1cols] = size(F2n1);
        for i = 1:f1rows
            for j = 1:f1cols
                if i == j
                    F2n1(i,j) = (1 - a1(i,1))*(a1(i,1));
                end
            end
        end
        
        % calculate first sensitivity s^M
        s2 = -2 * F2n2 * error;  % creates vector sized outputRows x 1
        
        % calculate first sensitivity s^M-1
        
        % calculate next layer's sensitivity s^M-2
        s1 = F2n1 * W2' * s2; %creates vector hiddenLayer x 1 sized 

        %------ Update Weights and Biases ------%

        W2 = W2 - (alpha * s2 * a1');
        b2 = b2 - (alpha * s2);

        W1 = W1 - (alpha * s1 * input');
        b1 = b1 - (alpha * s1);
    end
    mseIter = mseIter/targCols; % get the average for the epoch
    mseValues(iters, 1) = mseIter; % save it to output array 
    iters = iters + 1; % update iters for while loop control 
    x = mod(iters,50);
    if(x == 0)
        accValues(iters/50,1) = determineAccuracy((validTest2LaySM(testInputs, testTargets, W1, b1, W2, b2)), testLabels)
    end
end

end