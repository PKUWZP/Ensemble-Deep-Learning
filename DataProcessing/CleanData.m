te = load ('test_data.csv');
te=te';
llt=net(te);
TL=zeros(3468,2);
for i = 1: 3468
    for j = 1:6
        if llt(j,i)==max(llt(:,i))
            llt(:,i)=zeros(6,1);
            llt(j,i)=1;
            break
        end
    end
end

for k=1:6
    for m=1:3468
        if llt(k,m)==1
            TL(m,2)=k;
        end
    end
end

for n=1:3468
    TL(n,1)=n;
end
