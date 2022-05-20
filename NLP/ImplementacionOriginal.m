

mdl = bert


tokenizer = mdl.Tokenizer

filename = "train.csv";
data = readtable(filename,"TextType","string");
%data = datasample(data,500);

head(data)


data.target = categorical(data.target);


classes = categories(data.target);
numClasses = numel(classes)


figure
histogram(data.target)
xlabel("Class")
ylabel("Frequency")
title("Class Distribution")

data.Tokens = encode(tokenizer, data.text);


 cvp = cvpartition(data.target,"Holdout",0.2);
 dataTrain = data(training(cvp),:);
 dataValidation = data(test(cvp),:);



 
documents = preprocessText(dataTrain.text);
bag = bagOfWords(documents)
bag = removeInfrequentWords(bag,2);
[bag,idx] = removeEmptyDocuments(bag);

bag


numObservationsTrain = size(dataTrain,1)
numObservationsValidation = size(dataValidation,1)

textDataTrain = dataTrain.text;
textDataValidation = dataValidation.text;

TTrain = dataTrain.target;
TValidation = dataValidation.target;

tokensTrain = dataTrain.Tokens;
tokensValidation = dataValidation.Tokens;

figure
wordcloud(textDataTrain);
title("Training Data")

tokensTrain{1:5}

dsXTrain = arrayDatastore(tokensTrain,"OutputType","same");
dsTTrain = arrayDatastore(TTrain);
cdsTrain = combine(dsXTrain,dsTTrain);

dsXValidation = arrayDatastore(tokensValidation,"OutputType","same");
dsTValidation = arrayDatastore(TValidation);
cdsValidation = combine(dsXValidation,dsTValidation);


miniBatchSize = 32;
paddingValue = mdl.Tokenizer.PaddingCode;
maxSequenceLength = mdl.Parameters.Hyperparameters.NumContext;

mbqTrain = minibatchqueue(cdsTrain,1,...
    "MiniBatchSize",miniBatchSize, ...
    "MiniBatchFcn",@(X) preprocessPredictors(X,paddingValue,maxSequenceLength));


mbqValidation = minibatchqueue(cdsValidation,1,...
    "MiniBatchSize",miniBatchSize, ...
    "MiniBatchFcn",@(X) preprocessPredictors(X,paddingValue,maxSequenceLength));


if canUseGPU
    mdl.Parameters.Weights = dlupdate(@gpuArray,mdl.Parameters.Weights);
end

featuresTrain = [];
reset(mbqTrain);
while hasdata(mbqTrain)
    X = next(mbqTrain);
    features = bertEmbed(X,mdl.Parameters);
    featuresTrain = [featuresTrain gather(extractdata(features))];
end
 

featuresTrain = featuresTrain.';

featuresValidation = [];

reset(mbqValidation);
while hasdata(mbqValidation)
    X = next(mbqValidation);
    features = bertEmbed(X,mdl.Parameters);
    featuresValidation = cat(2,featuresValidation,gather(extractdata(features)));
end
featuresValidation = featuresValidation.';


numFeatures = mdl.Parameters.Hyperparameters.HiddenSize;
layers = [
    featureInputLayer(numFeatures)
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];



opts = trainingOptions('adam',...
    "MiniBatchSize",64,...
    "ValidationData",{featuresValidation,dataValidation.target},...
    "Shuffle","every-epoch", ...
    "Plots","training-progress", ...
    "Verbose",0);

net = trainNetwork(featuresTrain,dataTrain.target,layers,opts);

YPredValidation = classify(net,featuresValidation);

figure
confusionchart(TValidation,YPredValidation)


accuracy = mean(dataValidation.target == YPredValidation)

 filenametest='test.csv';

NewTweets = readtable(filenametest,"TextType","string");
NewTweets = NewTweets.text;





tokensNew = encode(tokenizer,NewTweets);


XNew = padsequences(tokensNew,2,"PaddingValue",tokenizer.PaddingCode);

filenameprediction = "sample_submission.csv";
new = readtable(filenameprediction,"TextType","string");

new =new(1:10,:) ;

featuresNew = bertEmbed(XNew,mdl.Parameters)';
featuresNew = gather(extractdata(featuresNew));
[new.target,scores] = classify(net,featuresNew);
writetable(new,'table.csv');



%% Supporting Functions
%% Predictors Preprocessing Functions
% The |preprocessPredictors| function truncates the mini-batches to have the 
% specified maximum sequence length, pads the sequences to have the same length. 
% Use this preprocessing function to preprocess the predictors only.

function X = preprocessPredictors(X,paddingValue,maxSeqLen)

X = truncateSequences(X,maxSeqLen);
X = padsequences(X,2,"PaddingValue",paddingValue);

end
%% BERT Embedding Function
% The |bertEmbed| function maps input data to embedding vectors and optionally 
% applies dropout using the "DropoutProbability" name-value pair.

function Y = bertEmbed(X,parameters,args)

arguments
    X
    parameters
    args.DropoutProbability = 0
end

dropoutProbabilitiy = args.DropoutProbability;

Y = bert.model(X,parameters, ...
    "DropoutProb",dropoutProbabilitiy, ...
    "AttentionDropoutProb",dropoutProbabilitiy);

% To return single feature vectors, return the first element.
Y = Y(:,1,:);
Y = squeeze(Y);

end