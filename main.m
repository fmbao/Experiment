clc,clear
nclas=5;%ѡȡ40����Ƭ
global imgrow;
global imgcol;
global edit2
imgrow=87;     %%%%%%ѡ�����Ƭ�Ĵ�С��Ӧ�ö���Ϊ87��424
imgcol=118;
%%%%%%%%%%%%%%%�������̾��ǽ��������ֵ�һ��Ȼ�����ϱ�ǩ��ͬ���Ļ��в��Լ�������
feature1_7=zeros(300,7);
feature8_9=zeros(300,2);
feature10_14=zeros(300,5);
feature15_73=zeros(300,59);
feature74_75=zeros(300,2);
feature76_78=zeros(300,3);
feature79_294=zeros(300,216);
feature3679_7278=zeros(300,3600);
feature7279_7284=zeros(300,6);
feature7285_7308=zeros(300,24);
gabor0=[];
gabor1=[];
gabor2=[];
gabor3=[];
I=cell(5,60); 
I_1=I;
% I_1=cell(5,40); 
p=cell(60,5);
count=0;
% for i=1:5                  %%%%%%%%%%%%%%%��ͬ���������
%     picpath='C:\Users\Administrator\Desktop\ʵ��\������ȡ+������ȡ+SVM����\text10_temprature_substraction';   %%%%��Ƭ��·��
%     picpath=strcat(picpath,num2str(i)); 
%     cd (picpath);
% % I{1,1}=imread('1.jpg');
%     Im=dir('*.jpg');
% %     N=length(Im);
% %     I=cell(1,N); 
% %     I_1=cell(1,N); 
% %     p=cell(N,1);
%     for k=1:40 
%         I{i,k}=imread(Im(k,1).name);
%     end
%     cd ..
% end
%  I_1=I;
 
%% ѵ������
for i=1:5                 
    picpath='G:\desktop\ʵ��\������ȡ+������ȡ+SVM����\text10_temprature_substraction';   %%%%��Ƭ��·��
    picpath=strcat(picpath,num2str(i)); 
    filename1=strcat(picpath,'\');
% I{1,1}=imread('1.jpg');
    for j=(1+(i-1)*90):(60+(i-1)*90)
        filename=strcat(filename1,num2str(j),'.jpg');
        I{i,(j-(i-1)*90)}=imread(filename);
        filename=filename1;
    end
   
end

T=cell(5,30);
T_1=T;
%% ��������
for i=1:5                 
    picpath='G:\desktop\ʵ��\������ȡ+������ȡ+SVM����\text10_temprature_substraction';   %%%%��Ƭ��·��
    picpath=strcat(picpath,num2str(i)); 
    filename1=strcat(picpath,'\');
% I{1,1}=imread('1.jpg');
    for j=(90*i-29):i*90
        filename=strcat(filename1,num2str(j),'.jpg');
        T{i,(j-90*i+30)}=imread(filename);
        filename=filename1;
    end
   
end

 %% ��ȡѵ����������
 %%%%%%���ڵ������ǿ���������ѡ��Ĳ��ã����µķ���������ǿ�������еļ�������������1-7������ء�����8-9�������ŷ����
 %%%%%%%����10-14��tamura��������������15-73��LBP���������� ����74_75����������ĻҶ�ֵ�ͱ�׼��  
 %%%%%%%����76_78:��ͨ���������һ���ڶ��������ӣ����жϱ�׼������������֮��ı�׼�������֮��ķ���
 %%%%%%%����79_3678��HOG������������3679_ ����Ϣ�ص�����
%%%%%%% ����7279_7282 �ǣ̣£�ͼ������ĻҶȹ���������ĸ�����
for j=1:5
    for k=1:60
     count=count+1;
    imag=I{j,k};
    imag_1=I{j,k};
    SM=sum(imag_1(:));
    %%%lbp�󵽵ĻҶȹ������������
    GLCM_lbp=lbp_change(imag);
    GLCM2_lbp=graycomatrix(GLCM_lbp,'NumLevels',256);
    GLCM2n_lbp=GLCM2_lbp/sum(GLCM2_lbp(:));
    stats=graycoprops(GLCM2_lbp);
    maxProbability2=max(GLCM2n_lbp(:));
    contrast2=stats.Contrast;
    coor2=stats.Correlation;
    energy2=stats.Energy;
    hom2=stats.Homogeneity;
    for I_cor=1:size(GLCM2n_lbp,1)
        sumcols(I_cor)=sum(-GLCM2n_lbp(I_cor,1:end).*log2(GLCM2n_lbp(I_cor,1:end)+eps));
    end
    entropy2=sum(sumcols);
    feature7279_7284(count,:)=[ maxProbability2,contrast2,coor2,energy2,hom2,entropy2];
    
    
    %%%gabor�õ����ĸ������ͼ��Ҷȹ�����������    [gabor0,gabor1,gabor2,gabor3]=Gabor2(imag);
    %0
    gabor0=graycomatrix(gabor0,'NumLevels',256);
    gabor0n=gabor0/sum(gabor0(:));
    stats=graycoprops(gabor0);
    maxProbability0=max(gabor0n(:));
    contrast0=stats.Contrast;
    coor0=stats.Correlation;
    energy0=stats.Energy;
    hom0=stats.Homogeneity;
    for I_cor=1:size(gabor0n,1)
        sumcols(I_cor)=sum(-gabor0n(I_cor,1:end).*log2(gabor0n(I_cor,1:end)+eps));
    end
    entropy0=sum(sumcols);
    %1
    gabor1=graycomatrix(gabor1,'NumLevels',256);
    gabor1n=gabor1/sum(gabor1(:));
    stats=graycoprops(gabor1);
    maxProbability1=max(gabor1n(:));
    contrast1=stats.Contrast;
    coor1=stats.Correlation;
    energy1=stats.Energy;
    hom1=stats.Homogeneity;
    for I_cor=1:size(gabor1n,1)
        sumcols(I_cor)=sum(-gabor1n(I_cor,1:end).*log2(gabor1n(I_cor,1:end)+eps));
    end
    entropy1=sum(sumcols);
    %2
    gabor2=graycomatrix(gabor2,'NumLevels',256);
    gabor2n=gabor2/sum(gabor2(:));
    stats=graycoprops(gabor2);
    maxProbability2=max(gabor2n(:));
    contrast2=stats.Contrast;
    coor2=stats.Correlation;
    energy2=stats.Energy;
    hom2=stats.Homogeneity;
    for I_cor=1:size(gabor2n,1)
        sumcols(I_cor)=sum(-gabor2n(I_cor,1:end).*log2(gabor2n(I_cor,1:end)+eps));
    end
    entropy2=sum(sumcols);
    %3
    gabor3=graycomatrix(gabor3,'NumLevels',256);
    gabor3n=gabor3/sum(gabor3(:));
    stats=graycoprops(gabor3);
    maxProbability3=max(gabor3n(:));
    contrast3=stats.Contrast;
    coor3=stats.Correlation;
    energy3=stats.Energy;
    hom3=stats.Homogeneity;
    for I_cor=1:size(gabor3n,1)
        sumcols(I_cor)=sum(-gabor3n(I_cor,1:end).*log2(gabor3n(I_cor,1:end)+eps));
    end
    entropy3=sum(sumcols);
    %%% ���ĸ������ͼ����Ҷȹ�����������
    feature7285_7308(count,:)=[maxProbability0,contrast0,coor0,energy0,hom0,entropy0 ,...
                                                     maxProbability1,contrast1,coor1,energy1,hom1,entropy1,...
                                                     maxProbability2,contrast2,coor2,energy2,hom2,entropy2,...
                                                      maxProbability3,contrast3,coor3,energy3,hom3,entropy3,...
                                                     ];
                                                 
   %% feature79_294(count,:)=cell2mat(HOG(imag));
    feature3679_7278(count,:)=entropy(imag);
    %[sum_area,block_num] =region_num(imag_1);
    %sum_block=sum_area*block_num/10;
    %feature76_78(count,:)=[sum_area,block_num,sum_block];
    feature74_75(count,:)=[mean2(imag_1),SM];
    imag=contourextraction(imag);
    I_1{j,k}=contourextraction(I{j,k});
%      I{j,k}=contourextraction(I{j,k});
    th = graythresh(imag);           %ȷ����ֵ
    I_1{j,k}=im2bw(imag,th);      %��ֵ�� 
    imag= I_1{j,k};
    I_1{j,k}=imopen(I_1{j,k},strel('disk',3));    %������
    I_1{j,k}=imfill(I_1{j,k}, 'holes');    %���
    I_1{j,k}=~I_1{j,k};                        %ȡ��
    Area=bwarea(I_1{j,k})/(imgrow*imgcol);    %�����
    Euler=bweuler(I_1{j,k});             %��ŷ����
% I_1{1,k}=double(I_1{1,k});
    p{k,j}=imagejui(imag);               %�����
    feature8_9(count,:)=[Area,Euler];
     feature1_7(count,:)=[p{k,j}]';
     
     Fcrs{k,j}=coarseness(I{j,k},4);
     Fcon{k,j}=contrast(I{j,k});
     [Fdir{k,j},sita{k,j}]=directionality(I{j,k});
     Flin{k,j}=linelikeness(I{j,k},sita{k,j},4);
%  Freg{k,1}=regularity(I{1,k},32);
     Frgh{k,j}=Fcrs{k,j}+Fcon{k,j};
     feature10_14(count,:)=[ Fcrs{k,j}, Fcon{k,j},Fdir{k,j},Flin{k,j},Frgh{k,j}];
     mapping=getmapping(8,'u2');   
     feature15_73(count,:)=lbp(I{j,k},1,8,mapping,'h'); %LBP histogram in (8,1) neighborhood  using uniform patterns
                                   
    end
end
% ,feature79_3678,,feature3679_7278,feature74_75,feature8_9,feature10_14,feature15_73,feature1_7,,feature79_3678
%  feature1_7,feature8_9,feature10_14,,feature3679_7278,feature79_3678,feature15_73,feature8_9,
feature_sum=cat(2,feature1_7, feature8_9,feature10_14,feature74_75,feature3679_7278,feature7279_7284);

%% ��ȡ�������ݵ�����
%%%%%%%%%%%%%%%%%%%%��ȡ�������ݲ��֣�����������ѵ�����ݱ�����1:4
feature_c_1_7=zeros(150,7);
feature_c_8_9=zeros(150,2);
feature_c_10_14=zeros(150,5);
feature_c_15_73=zeros(150,59);
feature_c_74_75=zeros(150,2);
feature_c_76_78=zeros(150,3);
feature_c_79_294=zeros(150,216);
feature_c_3679_7278=zeros(150,3600);
feature_c_7279_7284=zeros(150,6);
feature_c_7285_7308=zeros(150,24);
gabor_c_0=[];
gabor_c_1=[];
gabor_c_2=[];
gabor_c_3=[];
realclass=zeros(150,1);

count=0;
T_p=cell(30,5);
    %%%%%%%%%%%%%%%%�����150�����Լ���73������
for i=1:5
     for j=1:30
     count=count+1;
     T_1=T;
    imag=T{i,j};
    imag_1=T{i,j};
    SM_c_=sum(imag_1(:));
    GLCM_c=lbp_change(imag);
    GLCM2_c=graycomatrix(GLCM_c,'NumLevels',256);
    GLCM2n_c=GLCM2_c/sum(GLCM2_c(:));
    stats_c=graycoprops(GLCM2_c);
    maxProbability2_c=max(GLCM2n_c(:));
    contrast2_c=stats_c.Contrast;
    coor2_c=stats_c.Correlation;
    energy2_c=stats_c.Energy;
    hom2_c=stats_c.Homogeneity;
    for I_cor=1:size(GLCM2n_c,1)
        sumcols_c(I_cor)=sum(-GLCM2n_c(I_cor,1:end).*log2(GLCM2n_c(I_cor,1:end)+eps));
    end
    entropy2_c=sum(sumcols_c);
    feature_c_7279_7284(count,:)=[ maxProbability2_c,contrast2_c,coor2_c,energy2_c,hom2_c,entropy2_c];
    
    
    
    %%%��gabor ����ĸ������ͼ��Ȼ�������Ҷȹ�������
    [gabor_c_0,gabor_c_1,gabor_c_2,gabor_c_3]=Gabor2(imag);
    %0
    gabor_c_0=graycomatrix(gabor_c_0,'NumLevels',256);
    gabor_c_0n=gabor_c_0/sum(gabor_c_0(:));
    stats=graycoprops(gabor_c_0);
    maxProbability_c_0=max(gabor_c_0n(:));
    contrast_c_0=stats.Contrast;
    coor_c_0=stats.Correlation;
    energy_c_0=stats.Energy;
    hom_c_0=stats.Homogeneity;
    for I_cor=1:size(gabor_c_0n,1)
        sumcols(I_cor)=sum(-gabor_c_0n(I_cor,1:end).*log2(gabor_c_0n(I_cor,1:end)+eps));
    end
    entropy_c_0=sum(sumcols);
    %1
    gabor_c_1=graycomatrix(gabor_c_1,'NumLevels',256);
    gabor_c_1n=gabor_c_1/sum(gabor_c_1(:));
    stats=graycoprops(gabor_c_1);
    maxProbability_c_1=max(gabor_c_1n(:));
    contrast_c_1=stats.Contrast;
    coor_c_1=stats.Correlation;
    energy_c_1=stats.Energy;
    hom_c_1=stats.Homogeneity;
    for I_cor=1:size(gabor_c_1n,1)
        sumcols(I_cor)=sum(-gabor_c_1n(I_cor,1:end).*log2(gabor_c_1n(I_cor,1:end)+eps));
    end
    entropy_c_1=sum(sumcols);
    %2
    gabor_c_2=graycomatrix(gabor_c_2,'NumLevels',256);
    gabor_c_2n=gabor_c_2/sum(gabor_c_2(:));
    stats=graycoprops(gabor_c_2);
    maxProbability_c_2=max(gabor_c_2n(:));
    contrast_c_2=stats.Contrast;
    coor_c_2=stats.Correlation;
    energy_c_2=stats.Energy;
    hom_c_2=stats.Homogeneity;
    for I_cor=1:size(gabor_c_2n,1)
        sumcols(I_cor)=sum(-gabor_c_2n(I_cor,1:end).*log2(gabor_c_2n(I_cor,1:end)+eps));
    end
    entropy_c_2=sum(sumcols);
    %3
    gabor_c_3=graycomatrix(gabor_c_3,'NumLevels',256);
    gabor_c_3n=gabor_c_3/sum(gabor_c_3(:));
    stats=graycoprops(gabor_c_3);
    maxProbability_c_3=max(gabor_c_3n(:));
    contrast_c_3=stats.Contrast;
    coor_c_3=stats.Correlation;
    energy_c_3=stats.Energy;
    hom_c_3=stats.Homogeneity;
    for I_cor=1:size(gabor_c_3n,1)
        sumcols(I_cor)=sum(-gabor_c_3n(I_cor,1:end).*log2(gabor_c_3n(I_cor,1:end)+eps));
    end
    entropy_c_3=sum(sumcols);
    %%% ���ĸ������ͼ����Ҷȹ�����������
    feature_c_7285_7308(count,:)=[maxProbability_c_0,contrast_c_0,coor_c_0,energy_c_0,hom_c_0,entropy_c_0 ,...
                                                     maxProbability_c_1,contrast_c_1,coor_c_1,energy_c_1,hom_c_1,entropy_c_1,...
                                                     maxProbability_c_2,contrast_c_2,coor_c_2,energy_c_2,hom_c_2,entropy_c_2,...
                                                      maxProbability_c_3,contrast_c_3,coor_c_3,energy_c_3,hom_c_3,entropy_c_3,...
                                                     ];
    
     %[sum_area,block_num] =region_num(imag_1);
     %sum_block=sum_area*block_num/10;
    %% feature_c_79_294(count,:)=cell2mat(HOG(imag));
     feature_c_3679_7278(count,:)=entropy(imag);
    %feature_c_76_78(count,:)=[sum_area,block_num,sum_block];
    feature_c_74_75(count,:)=[mean2(imag_1),SM_c_];
     imag=contourextraction(imag);
     T_1{i,j}=contourextraction( T_1{i,j});
%     testpics{1,k}=contourextraction(testpics{1,k});
    
    th = graythresh(imag);    %ȷ����ֵ
    T_1{i,j}=im2bw(imag,th);      %��ֵ�� 
    imag= T_1{i,j};
    T_1{i,j}=imopen(T_1{i,j},strel('disk',3));    %������
    T_1{i,j}=imfill(T_1{i,j}, 'holes');    %���
    T_1{i,j}=~T_1{i,j};                    %ȡ��
    Area=bwarea(T_1{i,j})/(imgrow*imgcol);    %�����
    Euler=bweuler(T_1{i,j});             %��ŷ����
    T_p{j,i}=imagejui(imag);               %�����
    feature_c_8_9(count,:)=[Area,Euler];
     feature_c_1_7(count,:)=[T_p{j,i}]';
    
     Fcrs_c_{j,i}=coarseness(T{i,j},4);
     Fcon_c_{j,i}=contrast(T{i,j});
     [Fdir_c_{j,i},sita_c_{j,i}]=directionality(T{i,j});
     Flin_c_{j,i}=linelikeness(T{i,j},sita_c_{j,i},4);
%  Freg{k,1}=regularity(I{1,k},32);
     Frgh_c_{j,i}=Fcrs_c_{j,i}+Fcon_c_{j,i};
     feature_c_10_14(count,:)=[ Fcrs_c_{j,i}, Fcon_c_{j,i},Fdir_c_{j,i},Flin_c_{j,i},Frgh_c_{j,i}];
     mapping=getmapping(8,'u2');   
     feature_c_15_73(count,:)=lbp(T{i,j},1,8,mapping,'h'); %LBP histogram in (8,1) neighborhood  using uniform patterns
     end                           
end

% ,feature_c_79_3678,feature_c_3679_7278 ,feature_c_74_75,feature_c_8_9,feature_c_10_14,feature_c_15_73,feature_c_1_7,,feature_c_79_3678
% feature_c_1_7,feature_c_8_9,feature_c_10_14,,feature_c_3679_7278,feature_c_79_3678feature_c_15_73,,feature_c_74_75
% ,feature_c_8_9
feature_c_sum=cat(2, feature_c_1_7,feature_c_8_9,feature_c_10_14,feature_c_74_75,feature_c_3679_7278,feature_c_7279_7284);

%% ��ѵ�����ݺͲ����������ϱ�ǩ
%%%%%�Ȱ�ѵ���ı�ǩ������
practise_label=zeros(300,1);
for i=1:300
    if rem(i,60)~=0
    practise_label(i)=floor(i/60)+1;
    else
    practise_label(i)=floor(i/60);
    end 
end
%%%%%���Եı�ǩ
test_label=zeros(150,1);
for i=1:150
    if rem(i,30)~=0
    test_label(i)=floor(i/30)+1;
    else
    test_label(i)=floor(i/30);
    end 
end

%% ���ݹ�һ��
% ����Ԥ����,��ѵ�����Ͳ��Լ���һ����[0,1]����

[mtrain,ntrain] = size(feature_sum);
[mtest,ntest] = size(feature_c_sum);

dataset = [feature_sum;feature_c_sum];
% mapminmaxΪMATLAB�Դ��Ĺ�һ������
[dataset_scale,ps] = mapminmax(dataset',0,1);
dataset_scale = dataset_scale';

feature_sum = dataset_scale(1:mtrain,:);
feature_c_sum = dataset_scale( (mtrain+1):(mtrain+mtest),: );


%% PCA
set(edit2,'string','ѵ������PCA������ȡ......')
drawnow
[coef,score,latent,t2]=princomp(feature_sum);
% score=score';
% latent=latent';
% coef=coef';
pcapic=score(:,1:400);
disp('���ɷ�����');coef;
disp('������');score;
disp('����ֵ');latent;
figure;
percent_explained = 100*latent/sum(latent);
pareto(percent_explained);
xlabel('Principal Component');
ylabel('Variance Explained (%)');
figure;
biplot(coef(:,1:3), 'scores',score(:,1:3));
% % mA=mean(f_matrix);
% mA=mean(feature_sum);
% k=50;%��ά��10ά
% [pcapic,V]=fastPCA(feature_sum,k,mA);%���ɷַ�����������ȡ



%%%%%%%%%%%%%%��һ�����ǲ��Լ��Ľ�ά����
%%%%%%%%%%�������ݽ�ά
set(edit2,'string','��������PCA������ȡ......')
drawnow
[coef_c,score_c,latent_c,t2_c]=princomp(feature_c_sum);
% score_c=score_c';
% latent_c=latent_c';
% coef_c=coef_c';
pcatestpics=score_c(:,1:400);
disp('���ɷ�����');coef_c;
disp('������');score_c;
disp('����ֵ');latent_c;
figure;
percent_explained_c = 100*latent_c/sum(latent_c);
pareto(percent_explained_c);
xlabel('Principal Component');
ylabel('Variance Explained (%)');
figure;
biplot(coef_c(:,1:3), 'scores',score_c(:,1:3));

%% ��
%% ѡ����ѵ�SVM����c&g

% ���Ƚ��д���ѡ��: c&g �ı仯��Χ�� 2^(-10),2^(-9),...,2^(10)
[bestacc,bestc,bestg] = SVMcgForClass( practise_label, pcapic,-5,5,-5,5);

% ��ӡ����ѡ����
disp('��ӡ����ѡ����');
str = sprintf( 'Best Cross Validation Accuracy = %g%% Best c = %g Best g = %g',bestacc,bestc,bestg);
disp(str);

% ���ݴ���ѡ��Ľ��ͼ�ٽ��о�ϸѡ��: c �ı仯��Χ�� 2^(-2),2^(-1.5),...,2^(4), g �ı仯��Χ�� 2^(-4),2^(-3.5),...,2^(4),
% [bestacc,bestc,bestg] = SVMcgForClass(practise_label,pcapic,-2,4,-2,4,3,0.5,0.5,0.8);
% % ��ӡ��ϸѡ����
% disp('��ӡ��ϸѡ����');
% str = sprintf( 'Best Cross Validation Accuracy = %g%% Best c = %g Best g = %g',bestacc,bestc,bestg);
% disp(str);

%% ������ѵĲ�������SVM����ѵ��
cmd = ['-c ',num2str(bestc),' -g ',num2str(bestg)];
model = svmtrain(practise_label, pcapic,cmd);

%% SVM����Ԥ��
[ptrain_label, train_accuracy,train_values] = svmpredict( practise_label,  pcapic, model);
[predict_label,accuracy,values] = svmpredict(test_label,pcatestpics,model);

% ��ӡѵ����������׼ȷ��
total_x = length(practise_label);
right_x = sum(ptrain_label == practise_label);
disp('��ӡ���Լ�����׼ȷ��');
str = sprintf( 'Accuracy = %g%% (%d/%d)',train_accuracy(1),right_x,total_x);
disp(str);

% ��ӡ���Լ�����׼ȷ��
total = length(test_label);
right = sum(predict_label == test_label);
disp('��ӡ���Լ�����׼ȷ��');
str = sprintf( 'Accuracy = %g%% (%d/%d)',accuracy(1),right,total);
disp(str);

%% �������

% ���Լ���ʵ�ʷ����Ԥ�����ͼ
% ͨ��ͼ���Կ���ֻ���������������Ǳ���ֵ�
figure;
hold on;
plot(test_label,'o');
plot(predict_label,'r*');
xlabel('���Լ�����','FontSize',12);
ylabel('����ǩ','FontSize',12);
legend('ʵ�ʲ��Լ�����','Ԥ����Լ�����');
title('���Լ���ʵ�ʷ����Ԥ�����ͼ','FontSize',12);
grid on;
snapnow;




% %% ѡ����ѵ�SVM����c&g
% 
% 
% %%%%%%%%%���ãӣ֣ͽ��з���
% 
% [bestacc,bestc,bestg]=SVMcgForClass( practise_label, pcapic,-3,3,-3,3);
% cmd = ['-c ',num2str(bestc),' -g ',num2str(bestg)];
% % cmd = ['-c ',num2str(bestc),'-t',0];
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %����Ԥ��
% model = libsvmtrain( practise_label,  pcapic,cmd);
% [ptrain_label, train_accuracy,train_values] = libsvmpredict( practise_label,  pcapic, model);
% train_accuracy;
% [ptest_label, test_accuracy,dec_values] = libsvmpredict(test_label, pcatestpics, model);
% test_accuracy;
% % toc;


 




