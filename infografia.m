
filename = "train.csv";
data = readtable(filename,"TextType","string");
a='Û';
b='20';
c='New';
d='s';
data1=data.text;
data1 = erase(data1,a);
data2=data.keyword;
data2 = erase(data2,a);
data2 = erase(data2,b);
data3=data.location;
data3 = erase(data3,a);
data3 = erase(data3,c);
data3 = erase(data3,d);


% subplot(1,4,1)
% 
% histogram(data.target);
% title('Distribución de las clases')
% legend('No desastre','Desastre')
% 
% subplot(1,4,2)
% wordcloud(data1)
% 
% subplot(1,4,3)
% wordcloud(data2)
% 
% subplot(1,4,4)
wordcloud(data3)
title('Recurrencia en la localización')


