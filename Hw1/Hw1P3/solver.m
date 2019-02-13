n=10;
h=1/(n+1);
A=zeros(n,n);
for i=1:n
    for j=1:n
        if abs(i-j)==1
            A(i,j)=-1;
        elseif i==j
            A(i,j)=2;
        end
    end
end
A=A/h/h;
b=ones(n,1);
x=A\b;
