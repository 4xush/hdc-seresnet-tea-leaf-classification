% MATLAB Code for Tea Leaf Classification based on HD-SEResTeaNet Concepts

clear; close all; clc;

%% 1. Setup Configuration
% Dataset path - MODIFY THIS PATH
datasetPath = 'C:\Users\MATR!X\Desktop\GambungTeaLeave\augmented'; % Make sure this is correct!

% Image Properties
imgHeight = 224; % Standard input size, adjust if needed
imgWidth = 224;
numChannels = 3; % RGB input
inputSize = [imgHeight imgWidth numChannels];

% Network Parameters
numClasses = 5; % Number of Gambung tea leaf classes
initialFilters = 32; % Reduced initial filters as per paper concept
seReduction = 16;   % Reduction factor 'r' for SE block bottleneck
dropoutProb = 0.5;  % Dropout probability before final layer

% Training Parameters
maxEpochs = 30; % Adjust as needed
miniBatchSize = 32; % Adjust based on GPU memory
initialLearnRate = 1e-3; % Adam optimizer is often good
% validationFrequency calculation moved below after imdsTrain is created
executionEnvironment = 'auto'; % 'gpu', 'cpu', 'auto'

%% 2. Load and Prepare Data
fprintf('Loading and preparing dataset from: %s\n', datasetPath);

% Create an imageDatastore
imds = imageDatastore(datasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% --- Check if images were loaded ---
if isempty(imds.Files)
    error('No image files found in the specified dataset path: %s. Please check the path and folder structure.', datasetPath);
end
% --- End Check ---

% Display class names and counts
labelCounts = countEachLabel(imds);
disp(labelCounts);
classNames = categories(imds.Labels); % Use categories() for cell array of unique labels

% Split data into Training (70%) and Validation (30%) sets
[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.7, 'randomized');
numTrainImages = numel(imdsTrain.Files);
numValImages = numel(imdsValidation.Files);
fprintf('Dataset split: %d Training images, %d Validation images\n', numTrainImages, numValImages);

% *** MOVED THIS LINE HERE (from V2) ***
validationFrequency = floor(numTrainImages / miniBatchSize); % Validate once per epoch (based on training images)


%% 3. Data Augmentation
% Define augmentation pipeline for training data
pixelRange = [-30 30]; % Color Jitter
scaleRange = [0.8 1.2]; % Scaling
rotationRange = [-15 15]; % Rotation

imageAugmenter = imageDataAugmenter( ...
    'RandXReflection', true, ...
    'RandYReflection', true, ...
    'RandRotation', rotationRange, ...
    'RandXScale', scaleRange, ...
    'RandYScale', scaleRange, ...
    'RandXTranslation', pixelRange, ...
    'RandYTranslation', pixelRange);

% Create augmented datastores
augImdsTrain = augmentedImageDatastore(inputSize, imdsTrain, ...
    'DataAugmentation', imageAugmenter, 'ColorPreprocessing', 'gray2rgb'); % Use gray2rgb if needed, else remove

augImdsValidation = augmentedImageDatastore(inputSize, imdsValidation, ...
     'ColorPreprocessing', 'gray2rgb'); % No augmentation for validation, just resizing

%% 4. Define Network Architecture (HD-SEResTeaNet Inspired)

lgraph = layerGraph();

% --- Input Layer ---
inputLayer = imageInputLayer(inputSize, 'Name', 'input', 'Normalization', 'zscore');
lgraph = addLayers(lgraph, inputLayer);

% --- Initial Convolution Block (Stem) ---
stemConv = convolution2dLayer(7, initialFilters, 'Padding', 'same', 'Stride', 2, 'Name', 'stem_conv');
stemBn = batchNormalizationLayer('Name', 'stem_bn');
stemRelu = reluLayer('Name', 'stem_relu');
stemPool = maxPooling2dLayer(3, 'Padding', 'same', 'Stride', 2, 'Name', 'stem_pool');

lgraph = addLayers(lgraph, stemConv);
lgraph = addLayers(lgraph, stemBn);
lgraph = addLayers(lgraph, stemRelu);
lgraph = addLayers(lgraph, stemPool);

lgraph = connectLayers(lgraph, 'input', 'stem_conv');
lgraph = connectLayers(lgraph, 'stem_conv', 'stem_bn');
lgraph = connectLayers(lgraph, 'stem_bn', 'stem_relu');
lgraph = connectLayers(lgraph, 'stem_relu', 'stem_pool');

currentLayerName = 'stem_pool';
currentFilters = initialFilters;

% --- Stacking Hybrid Dilated SE-ResBlocks ---
% Example: Add a few blocks, increasing filters. Adjust counts/configs as needed.

% --- Block 1 (e.g., 64 filters) ---
blockName = 'block1';
numFilters = 64;
stride = 1; % No downsampling within block initially if using residual
blockLayers = hybridDilatedSEResBlock(currentFilters, numFilters, seReduction, stride, blockName); % Get layer definitions

% --- MODIFICATION START: Add Layers Individually ---
fprintf('Adding layers individually for %s...\n', blockName);
for i = 1:numel(blockLayers)
    % hybridDilatedSEResBlock might return empty layers (e.g., identity shortcut)
    if ~isempty(blockLayers(i)) && ~isa(blockLayers(i),'nnet.cnn.layer.LayerPlaceholder') % Check if layer is valid
        fprintf('Adding layer: %s\n', blockLayers(i).Name);
        lgraph = addLayers(lgraph, blockLayers(i));
    end
end
fprintf('Finished adding layers individually for %s.\n', blockName);
% --- MODIFICATION END ---

% Connect block internally and to previous layer (using the *full* connectHybridBlock)
fprintf('Connecting layers for %s...\n', blockName);
lgraph = connectHybridBlock(lgraph, currentLayerName, currentFilters, numFilters, stride, blockName);
fprintf('Finished connecting %s.\n', blockName);
currentLayerName = [blockName '_final_relu']; % Output of the block
currentFilters = numFilters;


% --- Block 2 (e.g., 128 filters, with downsampling) ---
blockName = 'block2';
numFilters = 128;
stride = 2; % Downsample input to match output size change
blockLayers = hybridDilatedSEResBlock(currentFilters, numFilters, seReduction, stride, blockName); % Get layer definitions

% --- MODIFICATION START: Add Layers Individually ---
fprintf('Adding layers individually for %s...\n', blockName);
for i = 1:numel(blockLayers)
     if ~isempty(blockLayers(i)) && ~isa(blockLayers(i),'nnet.cnn.layer.LayerPlaceholder') % Check if layer is valid
        fprintf('Adding layer: %s\n', blockLayers(i).Name);
        lgraph = addLayers(lgraph, blockLayers(i));
    end
end
fprintf('Finished adding layers individually for %s.\n', blockName);
% --- MODIFICATION END ---

% Connect block internally and to previous layer (using the *full* connectHybridBlock)
fprintf('Connecting layers for %s...\n', blockName);
lgraph = connectHybridBlock(lgraph, currentLayerName, currentFilters, numFilters, stride, blockName);
fprintf('Finished connecting %s.\n', blockName);
currentLayerName = [blockName '_final_relu'];
currentFilters = numFilters;


% --- Block 3 (e.g., 256 filters, with downsampling) ---
blockName = 'block3';
numFilters = 256;
stride = 2;
blockLayers = hybridDilatedSEResBlock(currentFilters, numFilters, seReduction, stride, blockName); % Get layer definitions

% --- MODIFICATION START: Add Layers Individually ---
fprintf('Adding layers individually for %s...\n', blockName);
for i = 1:numel(blockLayers)
     if ~isempty(blockLayers(i)) && ~isa(blockLayers(i),'nnet.cnn.layer.LayerPlaceholder') % Check if layer is valid
        fprintf('Adding layer: %s\n', blockLayers(i).Name);
        lgraph = addLayers(lgraph, blockLayers(i));
    end
end
fprintf('Finished adding layers individually for %s.\n', blockName);
% --- MODIFICATION END ---

% Connect block internally and to previous layer (using the *full* connectHybridBlock)
fprintf('Connecting layers for %s...\n', blockName);
lgraph = connectHybridBlock(lgraph, currentLayerName, currentFilters, numFilters, stride, blockName);
fprintf('Finished connecting %s.\n', blockName);
currentLayerName = [blockName '_final_relu'];
currentFilters = numFilters;

% Add more blocks as needed, following the paper's channel expansion strategy (32->64->128->256)

% --- Classification Head ---
globalPool = globalAveragePooling2dLayer('Name', 'global_pool');
dropoutLayer = dropoutLayer(dropoutProb, 'Name', 'dropout');
fcLayer = fullyConnectedLayer(numClasses, 'Name', 'fc_final');
softmaxLayer = softmaxLayer('Name', 'softmax');
classOutputLayer = classificationLayer('Name', 'ClassificationLayer');

lgraph = addLayers(lgraph, globalPool);
lgraph = addLayers(lgraph, dropoutLayer);
lgraph = addLayers(lgraph, fcLayer);
lgraph = addLayers(lgraph, softmaxLayer);
lgraph = addLayers(lgraph, classOutputLayer);

lgraph = connectLayers(lgraph, currentLayerName, 'global_pool');
lgraph = connectLayers(lgraph, 'global_pool', 'dropout');
lgraph = connectLayers(lgraph, 'dropout', 'fc_final');
lgraph = connectLayers(lgraph, 'fc_final', 'softmax');
lgraph = connectLayers(lgraph, 'softmax', 'ClassificationLayer');

% --- Analyze the Network ---
fprintf('Analyzing network structure...\n');
analyzeNetwork(lgraph);
% plot(lgraph); % Uncomment to visualize the network graph

%% 5. Specify Training Options
options = trainingOptions('adam', ... % Optimizer
    'InitialLearnRate', initialLearnRate, ...
    'MaxEpochs', maxEpochs, ...
    'MiniBatchSize', miniBatchSize, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augImdsValidation, ...
    'ValidationFrequency', validationFrequency, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', executionEnvironment, ...
    'LearnRateSchedule', 'piecewise', ... % Example: Drop learning rate
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 10);

%% 6. Train the Network
fprintf('Starting network training...\n');
[net, trainInfo] = trainNetwork(augImdsTrain, lgraph, options);
fprintf('Training finished.\n');

%% 7. Evaluate Network Performance (Optional)
fprintf('Evaluating network on validation set...\n');
YPred = classify(net, augImdsValidation, 'MiniBatchSize', miniBatchSize); % Corrected miniBatch_Size variable name
YValidation = imdsValidation.Labels;

accuracy = mean(YPred == YValidation);
fprintf('Validation Accuracy: %.4f\n', accuracy);

% Display confusion matrix
figure;
cm = confusionchart(YValidation, YPred);
cm.Title = 'Confusion Matrix for Validation Data';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';


%% --- Helper Function for Hybrid Dilated SE-ResBlock (WITH SE using functionLayer) ---
function layers = hybridDilatedSEResBlock(inputFilters, outputFilters, reduction, stride, blockName)
    % Creates layers for a Hybrid Dilated Squeeze-and-Excitation Residual Block.
    % Uses '_' instead of '/' in layer names.
    % VERSION 6: Uses functionLayer for SE scaling.

    % --- Residual Path (shortcut) ---
    if stride > 1 || inputFilters ~= outputFilters
        shortcutConv = convolution2dLayer(1, outputFilters, 'Stride', stride, 'Name', [blockName '_shortcut_conv']);
        shortcutBn = batchNormalizationLayer('Name', [blockName '_shortcut_bn']);
        shortcutLayers = [shortcutConv; shortcutBn];
    else
        shortcutLayers = []; % Identity shortcut
    end

    % --- Main Path ---
    % HDC... (same as before)
    rate1Conv = convolution2dLayer(3, outputFilters, 'Padding', 'same', 'DilationFactor', 1, 'Stride', stride, 'Name', [blockName '_hdc_rate1_conv']);
    rate1Bn = batchNormalizationLayer('Name', [blockName '_hdc_rate1_bn']);
    rate1Relu = reluLayer('Name', [blockName '_hdc_rate1_relu']);
    rate2Conv = convolution2dLayer(3, outputFilters, 'Padding', 'same', 'DilationFactor', 2, 'Stride', stride, 'Name', [blockName '_hdc_rate2_conv']);
    rate2Bn = batchNormalizationLayer('Name', [blockName '_hdc_rate2_bn']);
    rate2Relu = reluLayer('Name', [blockName '_hdc_rate2_relu']);
    rate3Conv = convolution2dLayer(3, outputFilters, 'Padding', 'same', 'DilationFactor', 3, 'Stride', stride, 'Name', [blockName '_hdc_rate3_conv']);
    rate3Bn = batchNormalizationLayer('Name', [blockName '_hdc_rate3_bn']);
    rate3Relu = reluLayer('Name', [blockName '_hdc_rate3_relu']);
    concatHDC = concatenationLayer(3, 3, 'Name', [blockName '_hdc_concat']);
    reduceConv = convolution2dLayer(1, outputFilters, 'Name', [blockName '_hdc_reduce_conv']);
    reduceBn = batchNormalizationLayer('Name', [blockName '_hdc_reduce_bn']);
    reduceRelu = reluLayer('Name', [blockName '_hdc_reduce_relu']); % F_reduce output

    % --- Squeeze-and-Excitation (SE) Path --- (same as before)
    sePool = globalAveragePooling2dLayer('Name', [blockName '_se_pool']);
    seFc1 = fullyConnectedLayer(ceil(outputFilters / reduction), 'Name', [blockName '_se_fc1']);
    seRelu = reluLayer('Name', [blockName '_se_relu']);
    seFc2 = fullyConnectedLayer(outputFilters, 'Name', [blockName '_se_fc2']);
    seSigmoid = sigmoidLayer('Name', [blockName '_se_sigmoid']); % Scaling factors output

    % --- Scaling using functionLayer --- % MODIFIED
 % --- Scaling using functionLayer --- 
scaleFn = @(features, scales) features .* reshape(scales, 1, 1, size(features,3), size(features,4));
seScaleFunLayer = functionLayer(scaleFn, ...
    'Formattable', true, ...
    'InputNames', {'features', 'scales'}, ...
    'OutputNames', {[blockName '_scaled_features']}, ... % Wrap in cell array
    'Name', [blockName '_se_scale_func']); 

    % --- Final Addition & ReLU --- (same as before)
    finalAdd = additionLayer(2, 'Name', [blockName '_final_add']);
    finalRelu = reluLayer('Name', [blockName '_final_relu']);

    % --- Assemble Layers for the Block ---
    layers = [
        shortcutLayers;
        % Main HDC Path
        rate1Conv; rate1Bn; rate1Relu;
        rate2Conv; rate2Bn; rate2Relu;
        rate3Conv; rate3Bn; rate3Relu;
        concatHDC;
        reduceConv; reduceBn; reduceRelu;
        % SE Path Calculation
        sePool; seFc1; seRelu; seFc2; seSigmoid;
        % SE Scaling Application (Function Layer) % MODIFIED
        seScaleFunLayer;
        % Final Add and ReLU
        finalAdd;
        finalRelu;
    ];
end
%% --- Helper Function to Connect Layers within a Hybrid Block (WITH SE using functionLayer) ---
function lgraph = connectHybridBlock(lgraph, inputLayerName, inputFilters, outputFilters, stride, blockName)
    % Connects the layers generated by hybridDilatedSEResBlock.
    % VERSION 6: Uses functionLayer for SE scaling connections.

    % Connect input to parallel HDC convolutions (same as before)
    lgraph = connectLayers(lgraph, inputLayerName, [blockName '_hdc_rate1_conv']);
    lgraph = connectLayers(lgraph, inputLayerName, [blockName '_hdc_rate2_conv']);
    lgraph = connectLayers(lgraph, inputLayerName, [blockName '_hdc_rate3_conv']);

    % Connect HDC conv -> bn -> relu chains (same as before)
    lgraph = connectLayers(lgraph, [blockName '_hdc_rate1_conv'], [blockName '_hdc_rate1_bn']);
    lgraph = connectLayers(lgraph, [blockName '_hdc_rate1_bn'], [blockName '_hdc_rate1_relu']);
    lgraph = connectLayers(lgraph, [blockName '_hdc_rate2_conv'], [blockName '_hdc_rate2_bn']);
    lgraph = connectLayers(lgraph, [blockName '_hdc_rate2_bn'], [blockName '_hdc_rate2_relu']);
    lgraph = connectLayers(lgraph, [blockName '_hdc_rate3_conv'], [blockName '_hdc_rate3_bn']);
    lgraph = connectLayers(lgraph, [blockName '_hdc_rate3_bn'], [blockName '_hdc_rate3_relu']);

    % Connect HDC ReLU outputs to concatenation (same as before)
    lgraph = connectLayers(lgraph, [blockName '_hdc_rate1_relu'], [blockName '_hdc_concat/in1']);
    lgraph = connectLayers(lgraph, [blockName '_hdc_rate2_relu'], [blockName '_hdc_concat/in2']);
    lgraph = connectLayers(lgraph, [blockName '_hdc_rate3_relu'], [blockName '_hdc_concat/in3']);

    % Connect concat to reduction path (same as before)
    lgraph = connectLayers(lgraph, [blockName '_hdc_concat'], [blockName '_hdc_reduce_conv']);
    lgraph = connectLayers(lgraph, [blockName '_hdc_reduce_conv'], [blockName '_hdc_reduce_bn']);
    lgraph = connectLayers(lgraph, [blockName '_hdc_reduce_bn'], [blockName '_hdc_reduce_relu']);

    % Connect SE Path (same as before)
    featureMapLayerName = [blockName '_hdc_reduce_relu'];
    lgraph = connectLayers(lgraph, featureMapLayerName, [blockName '_se_pool']);
    lgraph = connectLayers(lgraph, [blockName '_se_pool'], [blockName '_se_fc1']);
    lgraph = connectLayers(lgraph, [blockName '_se_fc1'], [blockName '_se_relu']);
    lgraph = connectLayers(lgraph, [blockName '_se_relu'], [blockName '_se_fc2']);
    lgraph = connectLayers(lgraph, [blockName '_se_fc2'], [blockName '_se_sigmoid']);
    seFactorsLayerName = [blockName '_se_sigmoid'];

    % --- Connect Inputs to functionLayer using InputNames --- % MODIFIED
    funcLayerName = [blockName '_se_scale_func'];
    lgraph = connectLayers(lgraph, featureMapLayerName, [funcLayerName '/features']); % Connect F_reduce to 'features' input port
    lgraph = connectLayers(lgraph, seFactorsLayerName, [funcLayerName '/scales']);   % Connect SE_fc2 to 'scales' input port

    % --- Connect Shortcut and SCALED Main Path (from functionLayer) to Final Addition --- % MODIFIED
    % Output port name is layerName + '/' + OutputName
% --- Connect Shortcut and SCALED Main Path to Final Addition ---
scaledFeaturesOutputPort = [blockName '_se_scale_func/' blockName '_scaled_features']; % Match OutputNames

    if stride > 1 || inputFilters ~= outputFilters
        % Convolutional shortcut exists
        lgraph = connectLayers(lgraph, inputLayerName, [blockName '_shortcut_conv']);
        lgraph = connectLayers(lgraph, [blockName '_shortcut_conv'], [blockName '_shortcut_bn']);
        shortcutOutputName = [blockName '_shortcut_bn'];
        lgraph = connectLayers(lgraph, shortcutOutputName, [blockName '_final_add/in1']);           % Connect shortcut bn to add/in1
        lgraph = connectLayers(lgraph, scaledFeaturesOutputPort, [blockName '_final_add/in2']);    % Connect SCALED features output port to add/in2
    else
        % Identity shortcut
        lgraph = connectLayers(lgraph, inputLayerName, [blockName '_final_add/in1']);               % Connect input layer (identity) to add/in1
        lgraph = connectLayers(lgraph, scaledFeaturesOutputPort, [blockName '_final_add/in2']);    % Connect SCALED features output port to add/in2
    end

    % Connect final add to final relu (same as before)
    lgraph = connectLayers(lgraph, [blockName '_final_add'], [blockName '_final_relu']);

end