function [ Y ] = lucasNLL_S( theta, nL00, nL00c, nL01, nL01c,...
    nL10, nL10c, nL11, nL11c, nC0, nC0c, nC1, nC1c, nF00, nF00c, nF01,...
    nF01c, nF10, nF10c, nF11, nF11c )
%computes the log likelihood over labelled sample
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

N = nC0+nC0c+nC1+nC1c;

Y = -( nL00*log(a) + nL00c*log(1-a) + nL01*log(b) + nL01c*log(1-b) + ...
    nL10*log(c) + nL10c*log(1-c) + nL11*log(d) + nL11c*log(1-d) + ...
    nC0*log(e) + nC0c*log(1-e) + nC1*log(f) + nC1c*log(1-f) + ...
    nF00*log(g) + nF00c*log(1-g) + nF01*log(h) + nF01c*log(1-h) + ...
    nF10*log(k) + nF10c*log(1-k) + nF11*log(m) + nF11c*log(1-m) )/N;

end

