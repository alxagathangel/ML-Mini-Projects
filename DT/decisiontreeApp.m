clear;clc;

% Dataset
gender = {'F', 'F', 'M', 'F', 'M', 'M'};
age = {'young', 'adult', 'adult', 'adult', 'young', 'young'};
app = {'atom count', 'check mate mate', 'beehive finder', 'check mate mate', 'atom count', 'atom count'};

% Μετατροπή του dataset σε table
dataset = table(gender', age', app', 'VariableNames', {'Gender', 'Age', 'App'});


% Υπολογισμός Εντροπίας
H = entropy(dataset.App);

% Υπολογισμός Information Gain
IGgender = information_gain(dataset, dataset.App, 'Gender');
IGage = information_gain(dataset, dataset.App, 'Age');

disp(['Entropy: ', num2str(H)]);
disp(['Information Gain - Gender: ', num2str(IGgender)]);
disp(['Information Gain - Age: ', num2str(IGage)]);


% Δομή του Decision Tree με βάση το information gain
if IGage > IGgender
    bestSplit = 'Age';
else
    bestSplit = 'Gender';
end

disp(['Best feature to split: ', bestSplit]);


% Ταξηνόμηση νέων πελατών
newCustomers = {'F', 'young'; 'F', 'adult'; 'M', 'adult'};

for i = 1:size(newCustomers, 1)
    gender = newCustomers{i, 1};
    age = newCustomers{i, 2};
    
    if strcmp(bestSplit, 'Age')
        if strcmp(age, 'young')
            prediction = 'atom count';
        else
            if strcmp(gender, 'F')
                prediction = 'check mate mate';
            else
                prediction = 'beehive finder';
            end
        end
    else
        if strcmp(gender, 'F')
            prediction = 'check mate mate';
        else
            if strcmp(age, 'young')
                prediction = 'atom count';
            else
                prediction = 'beehive finder';
            end
        end
    end
    
    fprintf('Customer %d: %s\n', i, prediction);
end



% Συνάρτηση Υπολογισμού Εντροπίας
function H = entropy(labels)
    values = unique(labels);
    H = 0;
    for i = 1:length(values)
        p = sum(strcmp(labels, values{i})) / length(labels);
        H = H - p * log2(p);
    end
end

% Συνάρτηση υπολογισμού Information Gain
function IG = information_gain(data, labels, feature)
    values = unique(data.(feature));
    H = entropy(labels);
    H_feature = 0;
    for i = 1:length(values)
        sub_labels = labels(strcmp(data.(feature), values{i}));
        H_feature = H_feature + (length(sub_labels) / length(labels)) * entropy(sub_labels);
    end
    IG = H - H_feature;
end
