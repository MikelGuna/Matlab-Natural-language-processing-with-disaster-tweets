
filename = "sample_submission.csv";
data = readtable(filename,"TextType","string");
% 
 filename2 = "BrutalForce.csv";
 datos = readtable(filename2,"TextType","string");

% filename3 = "BrutalForce0565.csv";
% datos = readtable(filename3,"TextType","string");

 datos=datos.elecion;
id=data.id;
 target=datos.elecion;
 sub=table(id,target);
 writetable(sub,'masivo.csv');



target=datos.elecion;
sub2=table(id,target);
writetable(sub2,'masivohumbral0565.csv');

