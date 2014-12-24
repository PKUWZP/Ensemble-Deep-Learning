load training_data.csv
load training_labels.csv
T=training_data;
T=T';
L=training_labels;
G=zeros(6,size(T,2));
for i = 1: length(L)
    if L(i)==1
        G(1,i)=1;
    end
    if L(i)==2
        G(2,i)=1;
    end
    if L(i)==3
        G(3,i)=1;
    end
    if L(i)==4
        G(4,i)=1;
    end
    if L(i)==5
        G(5,i)=1;
    end
    if L(i)==6
        G(6,i)=1;
    end
end

        