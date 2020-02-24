function [W1, b1, W2, b2, mseValues] = backProp(trainInputs, trainTargets, learningRate, iterations, architecture)
%   ------- BACKPROPAGATION FUNCTION -------
%   
%   trainInputs = input vector for training set 
%   trainTargets = target vector for training set
%   learning rate = alpha learning rate 
%   iterations = set value for maximum iterations 
%   architecture = vector representing number of neurons and layers for the
%       neural network i.e. [2 1] for a 1-2-1 pattern (doesn't count input)

%   transfer function definitions:
%   layer1 function: logsigmoid()
%   layer2 function: linear() 

hiddenLayer = 15;

[outR outC] = size(trainTargets);
outputRows = outR;

%   Initial Weights and biases 
[inpRows cols] = size(trainInputs);
W1 = zeros(hiddenLayer, inpRows); %creates empty weight matrix
b1 = rand(hiddenLayer, 1);
for m = 1:hiddenLayer
    for n = 1:inpRows
        W1(m,n) = rand(1)/10;
    end
end

%disp(W1)

%disp(b1)

[w1r w1c] = size(W1);
W2 = zeros(outputRows, w1r);
b2 = rand(outputRows, 1);

for m = 1:outputRows
    for n = 1:outputRows
        W2(m,n) = rand(1)/10;
        %b2(m) = rand(1)/10;
    end
end

%disp(W2)
%disp(b2)

% Some test data 
%W1 = [-0.27; -0.41]
%b1 = [-0.48; -0.13]

%W2 = [0.09 -0.17]
%b2 = [0.48]

%input = 1;
%target = (1 + sin(pi/4))
alpha = learningRate;
mseIter = 1; %just a start value for the while loop 
iters = 1; %iteration counter 
mseValues = zeros(iterations, 1);

while( mseIter > 0.002 && iters < iterations + 1)
    mseIter = 0;
    for passes = 1:cols
        input = trainInputs(:,passes);
        %disp(input)
        target = trainTargets(:,passes);
        %disp(target)
           %------ Now Propagate Forwards ------%

        %product = (W1 * trainInputs + b1)
        %n = (W1 * input) + b1;
        a1 = logSigmoid((W1 * input) + b1);
        %disp("A1 = ")
        %disp(a1)

        a2 = logSigmoid((W2 * a1) + b2);
        %disp("A2 = ")
        %disp(a2)

        error = target - a2;
        mseIter = mseIter + mse(target, a2);
        
        %------ Now Calculate Sensitivities and Backpropagate ------%
        F2n2 = zeros(outputRows, outputRows);
        for i = 1:outputRows
            for j = 1:outputRows
                if i == j
                    F2n2(i,j) = (1 - a2(i,1))*(a2(i,1));
                end
            end
        end

        %f2n2 = logSigDeriv(a2)'
        s2 = -2 * F2n2 * error;
        %disp(s2)

        F1n1 = zeros(hiddenLayer, hiddenLayer);
        [f1rows f1cols] = size(F1n1);
        for i = 1:f1rows
            for j = 1:f1cols
                if i == j
                    F1n1(i,j) = (1 - a1(i,1))*(a1(i,1));
                end
            end
        end
        %disp(F1n1)

        s1 = F1n1 * W2' * s2;

        %------ Update Weights and Biases ------%
        %disp(W2)
        %disp(b2)
        W2 = W2 - (alpha * s2 * a1');
        b2 = b2 - (alpha * s2);

        W1 = W1 - (alpha * s1 * input');
        b1 = b1 - (alpha * s1);
        
        
    end
    mseIter = mseIter/outC; % get the average for the epoch
    mseValues(iters, 1) = mseIter; % save it to output array 
    iters = iters + 1;
end

end

