function model = Propagation(partialData, partialTarget, unlabeledData , k, alpha, beta)

partialData = normr(partialData);
unlabeledData = normr(unlabeledData);
[a, label_num] = size(partialTarget);
b = size(unlabeledData, 1);

Fp = partialTarget;
Fp = Fp ./ repmat(sum(Fp, 2), 1, label_num);
Fu = ones(b, label_num);
Fu = Fu ./ repmat(sum(Fu, 2), 1, label_num);

kdPartialData = KDTreeSearcher(partialData);
kdUnlabelData = KDTreeSearcher(unlabeledData);

[neighbor, ~] = knnsearch(kdPartialData, partialData, 'k', k);
neighbor = neighbor(:, 2:k);
Waa = sparse(a, a);
for i = 1:a
    neighborIns = partialData(neighbor(i,:),:)';
    w = lsqnonneg(neighborIns,partialData(i,:)');
    Waa(i,neighbor(i,:)) = w';
end

[neighbor, ~] = knnsearch(kdUnlabelData, partialData, 'k', k);
Wab = sparse(a, b);
for i = 1:a
    neighborIns = unlabeledData(neighbor(i,:),:)';
    w = lsqnonneg(neighborIns,partialData(i,:)');
    Wab(i,neighbor(i,:)) = w';
end

[neighbor, ~] = knnsearch(kdPartialData, unlabeledData, 'k', k);
Wba = sparse(b, a);
for i = 1:b
    neighborIns = partialData(neighbor(i,:),:)';
    w = lsqnonneg(neighborIns,unlabeledData(i,:)');
    Wba(i,neighbor(i,:)) = w';
end

[neighbor, ~] = knnsearch(kdUnlabelData, unlabeledData, 'k', k);
neighbor = neighbor(:, 2:k);
Wbb = sparse(b, b);
for i = 1:b
    neighborIns = unlabeledData(neighbor(i,:),:)';
    w = lsqnonneg(neighborIns,unlabeledData(i,:)');
    Wbb(i,neighbor(i,:)) = w';
end

sumWaa = sum(Waa, 2);
sumWaa(sumWaa == 0) = 1;
sumWab = sum(Wab, 2);
sumWab(sumWab == 0) = 1;
sumWba = sum(Wba, 2);
sumWba(sumWba == 0) = 1;
sumWbb = sum(Wbb, 2);
sumWbb(sumWbb == 0) = 1;
H = Wba ./ repmat(sumWba, 1, a);
H = sparse(H);
J = Waa ./ repmat(sumWaa, 1, a);
J = sparse(J);
K = Wbb ./ repmat(sumWbb, 1, b);
K = sparse(K);
L = Wab ./ repmat(sumWab, 1, b);
L = sparse(L);

maxIter = 100;
Fp_ = Fp;
for iter = 1:maxIter
    tmp = Fp;
    if iter <= maxIter / 2
        beta_ = (iter / maxIter) * beta;
    else
        beta_ = beta;
    end
    Fu = alpha * H * Fp + (1 - alpha) * Fu;
    Fp = alpha * J * Fp + (1 - alpha) * Fp_;
    Fp = Fp .* partialTarget;
    Fp = Fp ./ repmat(sum(Fp,2), 1, label_num);
    Fu = beta_ * K * Fu + (1 - beta_) * Fu;
    Fp = beta_ * L * Fu + (1 - beta_) * Fp;
    Fp = Fp .* partialTarget;
    Fp = Fp ./ repmat(sum(Fp,2), 1, label_num);
    diff = norm(full(Fp) - full(tmp));
    if abs(diff) < 0.0001
        break
    end
end
fprintf('label propagation iteration: %d\n', iter);

labelSum = sum(Fp_);
predSum = sum(Fp);
poster = labelSum ./ predSum;
Fp = Fp .* repmat(poster, a, 1);
Fu = Fu .* repmat(poster, b, 1);

disambiguatedLabel = zeros(a, label_num);
pseudoLabel = zeros(b, label_num);

for i = 1:a
    [~, idx] = max(Fp(i,:));
    disambiguatedLabel(i,idx) = 1;
end

for i = 1:b
    [~, idx] = max(Fu(i,:));
    pseudoLabel(i,idx) = 1;
end


model.PartialData = partialData;
model.UnlabelData = unlabeledData;
model.Fp = Fp;
model.Fu = Fu;
model.disambiguatedLabel = disambiguatedLabel;
model.pseudoLabel = pseudoLabel;