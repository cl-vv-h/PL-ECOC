function model = ECOC(datas,labels,length,setTHR)

labels = full(labels);
labels = labels>=0.5;
[num_,q] = size(datas);
num_class=size(labels,2);
code_matrix=[];
models = cell(length,1);

for i=1:length
    disp(num2str(i));
    model = clfer_generate(num_class,datas,labels,setTHR);
    models{i,1}=model.svmmodel;
    %disp(class(models{1,1}));
    code_matrix=[code_matrix;model.code];
end
code_matrix = code_matrix.';


matrix_H = performance_matrix(models,num_class,length,labels,datas,code_matrix);

model.models = models;
model.length = length;
model.code_matrix = code_matrix;
model.matrix_H = matrix_H;
end


function model = clfer_generate(num_class,datas,labels,setTHR)
maxIter = 2000;
num_ = size(datas,1);
for iter=1:maxIter
    code = (rand(num_class,1)>=0.5);
    code = code.';
    %disp(code);
    yp=[];
    yn=[];
    for i=1:num_
        %disp(labels(i,:));
        if((labels(i,:) & code)==labels(i,:))
            item = [datas(i,:),1];
            yp = [yp;item];
        else
            if((labels(i,:) & ~code)==labels(i,:))
                item = [datas(i,:),0];
                yn = [yn;item];
            end
        end
    end
    num_p = size(yp,1);
    %disp(num_p);
    num_n = size(yn,1);
    if((num_p+num_n)>=setTHR&&(num_p>=setTHR/2)&&(num_n>=setTHR))
        train_set = [yp;yn];
        [train_num,len] = size(train_set);
        train_X = train_set(:,1:len-1);
        %disp(train_X);
        train_Y = train_set(:,len);
        %model.svmmodel = fitcsvm(train_X,train_Y,'KernelFunction','rbf','OptimizeHyperparameters',{'BoxConstraint','KernelScale'},  'HyperparameterOptimizationOptions',struct('ShowPlots',false));
        %model.code = code;
        break;
    end
end
model.svmmodel = fitcsvm(train_X,train_Y,'ClassNames',[false true],'Standardize',true,...
        'KernelFunction','rbf','BoxConstraint',1);
% model.svmmodel = fitcsvm(train_X,train_Y,'KernelFunction','rbf','OptimizeHyperparameters',{'BoxConstraint','KernelScale'},  'HyperparameterOptimizationOptions',struct('ShowPlots',false));
model.code = code;
end

function matrix_H = performance_matrix(models,num_class,length,labels,datas,code_matrix)
H = zeros(num_class,length);
for j=1:length
    model_j = models{j,1};
    %disp(class(model_j));
    predict_label = predict(model_j,datas);
    for i=1:num_class
        label_i = predict_label(labels(:,i)==1,:);
        H(i,j) = sum(label_i == code_matrix(i,j))/size(label_i,i);
    end
end
sum_H = sum(H);
matrix_H = H ./ sum_H;
end




