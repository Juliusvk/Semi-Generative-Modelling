function [ E ] = lucas_error(theta, S, G, C, F, L)
%compute error from parameter estimate and test data
a = sigm(theta(1));
b = sigm(theta(2));
c = sigm(theta(3));
d = sigm(theta(4));
e = sigm(theta(5));
f = sigm(theta(6));
g = sigm(theta(7));
h = sigm(theta(8));
k = sigm(theta(9));
m = sigm(theta(10));

X = [S, G, C, F];
Y = zeros(size(L));

for i=1:size(S,1)
    if X(i,:) == [0 0 0 0]
        Y(i,1) = a*(1-f)*(1-k)/( a*(1-f)*(1-k) + (1-a)*(1-e)*(1-g) );
    elseif X(i,:) == [0 0 0 1]
        Y(i,1) = a*(1-f)*k / ( a*(1-f)*k + (1-a)*(1-e)*g );
    elseif X(i,:) == [0 0 1 0]
        Y(i,1) =  a*f*(1-m) / ( a*f*(1-m) + (1-a)*e*(1-h) );
    elseif X(i,:) == [0 0 1 1]
        Y(i,1) =  a*f*m / ( a*f*m + (1-a)*e*h );
    elseif X(i,:) == [0 1 0 0]
        Y(i,1) =  b*(1-f)*(1-k) / ( b*(1-f)*(1-k) + (1-b)*(1-e)*(1-g) );
    elseif X(i,:) == [0 1 0 1]
        Y(i,1) =  b*(1-f)*k / ( b*(1-f)*k + (1-b)*(1-e)*g );
    elseif X(i,:) == [0 1 1 0]
        Y(i,1) =  b*f*(1-m) / ( b*f*(1-m) + (1-b)*e*(1-h) );
    elseif X(i,:) == [0 1 1 1]
        Y(i,1) =  b*f*m / ( b*f*m + (1-b)*e*h );
    elseif X(i,:) == [1 0 0 0]
        Y(i,1) = c*(1-f)*(1-k)/( c*(1-f)*(1-k) + (1-c)*(1-e)*(1-g) );
    elseif X(i,:) == [1 0 0 1]
        Y(i,1) = c*(1-f)*k / ( c*(1-f)*k + (1-c)*(1-e)*g );
    elseif X(i,:) == [1 0 1 0]
        Y(i,1) =  c*f*(1-m) / ( c*f*(1-m) + (1-c)*e*(1-h) );
    elseif X(i,:) == [1 0 1 1]
        Y(i,1) =  c*f*m / ( c*f*m + (1-c)*e*h );
    elseif X(i,:) == [1 1 0 0]
        Y(i,1) =  d*(1-f)*(1-k) / ( d*(1-f)*(1-k) + (1-d)*(1-e)*(1-g) );
    elseif X(i,:) == [1 1 0 1]
        Y(i,1) =  d*(1-f)*k / ( d*(1-f)*k + (1-d)*(1-e)*g );
    elseif X(i,:) == [1 1 1 0]
        Y(i,1) =  d*f*(1-m) / ( d*f*(1-m) + (1-d)*e*(1-h) );
    elseif X(i,:) == [1 1 1 1]
        Y(i,1) =  d*f*m / ( d*f*m + (1-d)*e*h );
    end
end

Pred = Y>0.5;
E = 1 - mean(Pred == L);

end

