function E = getErrorLRClass(theta, D)

X_C = D(1,:);
X_E = D(3,:);
n = size(D,2);
X = [ones(1,n); X_C; X_E];
pX = 1./(1+exp(-theta*X)); 
fitted = (pX > 0.5);
Y = D(2,:);
correct = (fitted == Y);
E = 1 - mean(correct);

end

