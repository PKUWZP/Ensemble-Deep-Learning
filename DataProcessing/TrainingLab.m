ll=net(T);
for i = 1: 6831
    for j = 1:6
        if ll(j,i)==max(ll(:,i))
            ll(:,i)=zeros(6,1);
            ll(j,i)=1;
            break
        end
    end
end

