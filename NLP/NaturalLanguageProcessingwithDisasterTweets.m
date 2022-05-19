%% Natural Language Processing with Disaster Tweets, implementación con BERT
% Este modelo de ML tiene como proposito clasificar Tweets, concretamente
% resolviendo la competición "Natural Language Processing with Disaster
% Tweets" de kaggle. 
% Para ello se utiliza un modelo BERT pre entrenado el cual usaremos para
% extraer el significado inherente y contextual de las palabras. Una vez
% que extraemos el significado a nuestro texto como tokens, podemos
% entrenar una red neuronal para que trabaje como clasificador.

%% Descarga del modelo BERT pre-entrenado
% La función BERT, perteneciente al repositorio transformer-models for
% matlab de "bwdGitHub Ben" , nos da acceso a un encoder que tokeniza las
% palabras y una serie de parámetros.

mdl = bert

%% Visualizacion de los tokens 
% BERT utiliza diferentes tipos de tokens (CLS, SEP...), son usados para
% dividir y clasificar las palabras y oracones.

tokenizer = mdl.Tokenizer

%% Caragar Datos de entrenamiento
% Lectura del fichero de datos de entrenamiento, hay una opcion que se
% puede descomentar para trabajar con un set de datos de una cierta
% longuitud

filename = "train.csv";
data = readtable(filename,"TextType","string");
data = datasample(data,10);


%% Mostrar si los datos tienen el formato adecuado

head(data)

%% Seleccionar la caracteristica objetivo, en este caso en particular es "target"  
%Esta caracteristica sera sobre la cual se entrenara el modelo para
%posteriormente predecirla

data.target = categorical(data.target);

target=data.target;


%% Distribución de la clase a predecir

classes = categories(data.target);
numClasses = numel(classes);
figure
histogram(data.target)
xlabel("Tipo")
ylabel("Frecuencia")
title("Distribución de la clase a predecir")

 %% Preprocesamiento y limpieza del texto 
 %Esta función eficienta mucho el tiempo de caclulo del algoritmo, ya que
 %limpia fuertemente el texto
 
 % La función preprocessText limpia de ciertas palabras el texto, elimina
 % los signos de puntuación del texto, elimina las palabras extremadamente
 % cortas (menos de 2 letras) o extremadamente largras (más de 15 letras) y
 % finalmente usa normalizeWords de matlab para la lematización de las
 % palabras (obtener la raiz original de una palabra ej: Running a run).

% documents = preprocessText(data.text);
% bag = bagOfWords(documents);
% bag = removeInfrequentWords(bag,2);
% [bag,idx] = removeEmptyDocuments(bag);
% bag;

%% Tokenizar el texto usando la función "encode"
% Encode usa el modelo pre entrenado de BERT para asignar variables
% numericas a cada palabra 

data.Tokens = encode(tokenizer, data.text);

%% Partición del conjunto de entrenamiento para obtener conjunto de CV 
%Se expecifica que la retención (Holdout) del conjunto de entrenamiento sea
%del 20%, es decir el set de validación sera el 20% del set de
%entrenamiento original

 cvp = cvpartition(data.target,"Holdout",0.2);
 dataTrain = data(training(cvp),:);
 dataValidation = data(test(cvp),:);
 
%Comprobación de que el numero de datos en cada conjunto es correcto 

numObservationsTrain = size(dataTrain,1)
numObservationsValidation = size(dataValidation,1)



%% Extraer las diferentes partes de los datos
%Cogemos los datos de texto, target y los token del conjunto de
%entrenamiento y de validación

textDataTrain = dataTrain.text;
textDataValidation = dataValidation.text;

TTrain = dataTrain.target;
TValidation = dataValidation.target;

tokensTrain = dataTrain.Tokens;
tokensValidation = dataValidation.Tokens;

%% Visualizar fromato del texto importado y tokens

figure
wordcloud(textDataTrain);
title("Conjunto de entrenamiento");
tokensTrain{1:5};

%% Preparación de los datos para entrenamiento (Mini-Bach queue)

%Necesitamos que BERT nos mapee los tokens a caracteristicas para
%posteriormenete clasificar, para ello necesitamos proporcionarle los datos
%de una forma ordenada, iteración tras iteración. Para ello alamcenamos los tokens 
%con su correspondiente clase con la función arrayDatastore y luego lo
%fusionamos para generar un objeto con la función minibatchqueue (necesita un único input,
%por eso se fusiona)


% Mini-batch queues require a single datastore that outputs both the
% predictors and responses. Create array datastores containing the training
% BERT tokens and labels and combine them using the |combine| function.
dsXTrain = arrayDatastore(tokensTrain,"OutputType","same");
dsTTrain = arrayDatastore(TTrain);
cdsTrain = combine(dsXTrain,dsTTrain);

% Se repite para la parte de validación
dsXValidation = arrayDatastore(tokensValidation,"OutputType","same");
dsTValidation = arrayDatastore(TValidation);
cdsValidation = combine(dsXValidation,dsTValidation);

%% Generación de la cola de datos para entrenamiento  
% Una vez que tenemos los datos "fusionados" podemos usar la función
% minibatchqueue para formar una cola de datos (un objeto) para ir pasando
% la información.
%Definimos la longuitud de la cola, puede ser 32 o 64
% mbqTrain es la fila de datos en si misma

miniBatchSize = 32;
paddingValue = mdl.Tokenizer.PaddingCode;
maxSequenceLength = mdl.Parameters.Hyperparameters.NumContext;

mbqTrain = minibatchqueue(cdsTrain,1,...
    "MiniBatchSize",miniBatchSize, ...
    "MiniBatchFcn",@(X) preprocessPredictors(X,paddingValue,maxSequenceLength));

%% Generación de la cola de datos para entrenamiento  


mbqValidation = minibatchqueue(cdsValidation,1,...
    "MiniBatchSize",miniBatchSize, ...
    "MiniBatchFcn",@(X) preprocessPredictors(X,paddingValue,maxSequenceLength));

%% Calculos en GPU si esta disponible

if canUseGPU
    mdl.Parameters.Weights = dlupdate(@gpuArray,mdl.Parameters.Weights);
end

%% Convertir tokens en vectores caracteristica
%La función bertembed definida al final del script combierte los tokens
%asigandos a cada ejemplo de entrenamiento en vectores embebidos de
%caracteristicas
%hastdata es un función de matlab que devueleve un booleano (1 si hay datos en la cola 0 si ya no hay)
% En el bucle lo que hacemos es pedir datos a la cola mbqTrain (minibatchqueue) 
% y pasarselos a bertembed

featuresTrain = [];
reset(mbqTrain);

while hasdata(mbqTrain)
    X = next(cdsValidation);
    features = bertEmbed(X,mdl.Parameters);
    featuresTrain = [featuresTrain gather(extractdata(features))];
end

%% Trasponer las caracteristicas de entrenamiento (N x EmbeddingDimension)

featuresTrain = featuresTrain.';

%% Realizar los mismos pasos para el set de validación

featuresValidation = [];

reset(mbqValidation);
while hasdata(mbqValidation)
    X = next(mbqValidation); 
    features = bertEmbed(X,mdl.Parameters);
    featuresValidation = cat(2,featuresValidation,gather(extractdata(features)));
end
featuresValidation = featuresValidation.';

%% Definimos la red neuronal para la clasificación de vectores caracteristicas
%En los pasos previos hemos mapaeado de palabras a tokens, de tokens a
%caracteristicas y ahora esas caracteristicas seran usadas para clasificar

numFeatures = mdl.Parameters.Hyperparameters.HiddenSize;
layers = [
    featureInputLayer(numFeatures)
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

%% Opciones de entrenamiento 
% Especificar opciones de entremiento.

opts = trainingOptions('adam',...
    "MiniBatchSize",64,...
    "ValidationData",{featuresValidation,dataValidation.target},...
    "Shuffle","every-epoch", ...
    "Plots","training-progress", ...
    "Verbose",0);


%% Entrenamiento

% Entrenamos la red neuronal con la función trainNetwork.
net = trainNetwork(featuresTrain,dataTrain.target,layers,opts);

%% Testamosel algoritmo
% Hacemos predicciones sobre los datos de validación, asi mismo vemos el
% sesgo de nuesto algortimo, vemos los verdaderos y falsos
% positivos/Negativos. 

YPredValidation = classify(net,featuresValidation);

figure
confusionchart(TValidation,YPredValidation)


%% Precisión del modelo en Validación

accuracy = mean(dataValidation.target == YPredValidation);

%% Clasificar nuevos Tweets
%Cargamos la información
%De forma similar a lo que hemos hecho antes, cargamos el fichero de nuevos
%tweets, si se quiere probar de forma más rapida descomentar la opción de
%coger solo una parte del conjunto.

 filenametest='test.csv';

NewTweets = readtable(filenametest,"TextType","string");
NewTweets = NewTweets.text;

%  NewTweets = NewTweets(1:1000,1) %Truncar el set



%% Tokenizamos el texto de los nuevos Tweets

tokensNew = encode(tokenizer,NewTweets);

%% Combertimos los nuevos Tweets a la misma longuitud


XNew = padsequences(tokensNew,2,"PaddingValue",tokenizer.PaddingCode);


%% Clasificar nuevos Tweets
% Clasificamos los nuevos tweets con la función classify y escribimos los
% resultados en un fichero.
%El clasificador nos devuelve tanto la potencial clase como la prob de 1 o
%de 0, esto será muy valioso para en el futuro poder usar alguna estrategia
%de fuerza bruta y conocer mejor las debilidades del algortimo

filenameprediction = "sample_submission.csv";
new = readtable(filenameprediction,"TextType","string");

%new =new(1:1000,:); % Si tenemos habilitada la opción de truncar el set de nuevos tweets debemos habilitar esta nueva opción para la escritura

featuresNew = bertEmbed(XNew,mdl.Parameters)';
featuresNew = gather(extractdata(featuresNew));
[new.target,scores] = classify(net,featuresNew); %Scores de la probabilidad de cada nuevo tweet
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
%% Función BERT Embedding 

% La función bertEmbed perteneciente a la toolbox BERT NLP mapea los datos
% de entrada a un vector numerico de salida en el cual se tiene en cuenta
% el significado contectual.

% En NLP word embedding es un termino que describe la representación de
% palabras como vectores de numeros reales, a parecidos valores parecido
% numero.
%Mejor explicado:
% In natural language processing (NLP), word embedding is a term used for the representation of words for text analysis, 
% typically in the form of a real-valued vector that encodes the meaning of the word such that the words that are closer in 
% the vector space are expected to be similar in meaning.

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