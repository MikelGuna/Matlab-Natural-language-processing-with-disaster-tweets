%% Trabajo implementacion alternativa

%% Carga de datos (Con opcion limite de datos)
filename = "train.csv";
data = readtable(filename,"TextType","string");
% data = datasample(data,3000); % seleccion una parte de los datos optimas

%% Seleccionamos el objetivo para procesar y vemos la distribucion de las variables objetivo

data.target = categorical(data.target);
figure
histogram(data.target)
xlabel("Class")
ylabel("Frequency")
title("Class Distribution")

%% Cuantas clases tenemos ?

classes = categories(data.target);
numClasses = numel(classes)

%%  Dividir los datos de entrenamiento en Cross validation (CV)

cvp = cvpartition(data.target,"Holdout",0.2);
dataTrain = data(training(cvp),:);
dataTest = data(test(cvp),:);
%% Extraer el texto 
textDataTrain = dataTrain.text;
textDataTest = dataTest.text;
YTrain = dataTrain.target;
YTest = dataTest.target;



%% Preprocesado del texto usando la funcion "ProcessText" de la toolbox de matlab Text Anlitycs

documents = preprocessText(textDataTrain);
bag = bagOfWords(documents)
bag = removeInfrequentWords(bag,2);
[bag,idx] = removeEmptyDocuments(bag);
 YTrain(idx) = [];
bag

%% Entrenar el clasificador supervisado con la funcion 
XTrain = bag.Counts;
mdl = fitcecoc(XTrain,YTrain,'Learners','linear')

%% Comprobar eficiencia del clasificador

documentsTest = preprocessText(textDataTest);
XTest = encode(bag,documentsTest);
YPred = predict(mdl,XTest);
acc = sum(YPred == YTest)/numel(YTest) ; 


%% Predecir nuevos datos Pillar los datos de Test real

filenametest='test.csv';
NewTweets = readtable(filenametest,"TextType","string");
NewTweets = NewTweets.text;

%% Escribir en el fichero objetivos para la entrega en kaggle (opciones para reducir la dimension de los datos y hacer testing)

filenameprediction = "sample_submission.csv";
new = readtable(filenameprediction,"TextType","string");
% new =new(1:10,:) ;
featuresNew = preprocessText(NewTweets)';
featuresNew = encode(bag,featuresNew);
new.target = predict(mdl,featuresNew);
writetable(new,'tablealternativa.csv')







