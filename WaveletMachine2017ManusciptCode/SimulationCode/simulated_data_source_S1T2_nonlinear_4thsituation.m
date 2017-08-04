%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Reference
%N. Correa, Y.-O. Li, T. Adali, and V. Calhoun, "Comparison of blind source
%separation algorithms for fMRI using a new Matlab toolbox: GIFT," in 
% Proc. IEEE Int. Conf. Acoust., Speech, Signal Processing (ICASSP),
%Philadelphia, PA, vol. 5, pp. 401-404, March 2005.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clearvars;
close all;
xdim=60;
ydim=60;
ALLsubs=zeros(40,3600);
CURRENTsub = 0;
xy_dim=xdim*ydim;
timeset=100;
num_sources=8;
figure;
%*  *   *   *   *   *   *   *   *   *   *   *   *   *   *
%Source 1
%*  *   *   *   *   *   *   *   *   *   *   *   *   *   *
for subj= 1:40
pd1=2;
pd2=3;
weight = [1, 1, 4, 9, 16, -400, -361, -324, -289, -0.5, -225, -196, -169, -0.01, -121, -100, -81, -64, -49, -36, -25, -16, -9, -4, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225, 256, 289, 324, 361, -0.4, -225, -196, -169];
w=[0:2*pi/(xdim-1):2*pi];
S1=sin(pd1.*w')*sin(pd2.*w);
image(100*S1); title('Sources');
S1n=zeros(60,60);
for j=41:50
    for i=31:45
        S1n(i,j)=S1(i,j)*weight(subj);
    end;
    for i=1:15
        S1n(i,j)=S1(i,j)*(1/weight(subj));
    end;
end;
S1=S1n;
subplot(num_sources,3,1);
imagesc(S1); title('Sources'); axis square, axis off;
colormap gray;

%*  *   *   *   *   *   *   *   *   *   *   *   *   *   *
%Source 2
%*  *   *   *   *   *   *   *   *   *   *   *   *   *   *

pd1=3;
pd2=4;
w=[0:2*pi/(xdim-1):2*pi];
S2=sin(pd1.*w')*sin(pd2.*w);
S2n=zeros(60,60);
for i=11:20
    for j=9:15
        S2n(i,j)=S2(i,j);
    end;
end;
for i=31:40
    for j=9:15
        S2n(i,j)=S2(i,j);
    end;
end;
for i=21:30
    for j=31:37
        S2n(i,j)=S2(i,j);
    end;
end;
S2n=S2n+0.5;
%%deactivation area
S2=S2n;
A=S2(11:20, 9:15);
A=.5+-1*A;
S2(31:40, 39:45)=A;
subplot(num_sources,3,4);
imagesc(S2); axis square, axis off;
colormap gray;


%*  *   *   *   *   *   *   *   *   *   *   *   *   *   *
%Source 3
%*  *   *   *   *   *   *   *   *   *   *   *   *   *   *

S3=zeros(60,60);
for j=2:60
    for i=1:60
        S3(i,j)=S3(i,j-1)+rand(1)/40;
    end;
end;

subplot(num_sources,3,7);
imagesc(S3); axis square, axis off;
colormap gray;

%*  *   *   *   *   *   *   *   *   *   *   *   *   *   *
%Source 4
%*  *   *   *   *   *   *   *   *   *   *   *   *   *   *
%newshifted gaussian
S4=0+ 0.3 *randn(60,60);
cnt=1;
for i=20-cnt:-1:16
    for j=20+cnt:40-cnt
        S4(i,j)=0.25;
    end
    cnt=cnt+2;
end
cnt=1;
for j=20:40
    for i=25:-1:20
       S4(i,j)=0.25;
    end
end 
for i=22+cnt:1:25
    for j=30-cnt:30+cnt
        S4(i,j)=S4(i,j)-0.20;
    end
    cnt=cnt+3;
end
cnt=1;
 for i=40+cnt:1:44
    for j=20+cnt:40-cnt
        S4(i,j)=-0.25;
    end
    cnt=cnt+2;
end       
 cnt=1;
for j=20:40
    for i=40:-1:34
       S4(i,j)=-0.25;
    end
end 
for i=38-cnt:-1:34
    for j=30-cnt:30+cnt
        S4(i,j)=S4(i,j)+0.20;
    end
    cnt=cnt+3;
end       

S_4=zeros(60,60);
for i=2:59
    for j=2:59
        for s=1:3
            for t=1:3
                S_4(i,j)=S_4(i,j)+S4(i-2+s,j-2+t);
                t=t+1;
            end
            s=s+1;
        end
    end
end

S4=S_4;
subplot(num_sources,3,10);
imagesc(S4); axis square, axis off;
colormap gray;
% 

%*  *   *   *   *   *   *   *   *   *   *   *   *   *   *
%Source 5
%*  *   *   *   *   *   *   *   *   *   *   *   *   *   *
pd1=.5;
pd2=.9;
w=[0:4*pi/(xdim-1):4*pi];
t3=sin(pd1.*w')*sin(pd2.*w);
test=t3(33:57,35:50);
B=imresize(test,[21 11], 'bilinear');
S5=zeros(60,60);
S5(1:21,3:13)=B;
S5(40:60,50:60)=-5*B;
subplot(num_sources,3,13);
imagesc(S5); axis square, axis off;
colormap gray;

%*  *   *   *   *   *   *   *   *   *   *   *   *   *   *
%Source 6
%*  *   *   *   *   *   *   *   *   *   *   *   *   *   *
pd1=.5;
pd2=.9;
w=[0:4*pi/(xdim-1):4*pi];
S6=sin(pd1.*w')*sin(pd2.*w);
S6(:,1:33)=0;
S6(:,51:end)=0;
S6=rot90(rot90(S6));
s6=zeros(60,60);
s6(15:44,1:20)=S6(31:60,11:30);
s6(15:44,41:60)=S6(31:60,11:30);
s6=rot90(s6);
S6=s6;
subplot(num_sources,3,16),imagesc(S6);axis square, axis off;
colormap gray;

%*  *   *   *   *   *   *   *   *   *   *   *   *   *   *
%Source 7
%*  *   *   *   *   *   *   *   *   *   *   *   *   *   *
pd1=.1;
pd2=.2;
w=[0:2*pi/(xdim-1):2*pi];
S7=sin(pd1.*w')*sin(pd2.*w);
S7(:,1:30)=fliplr(S7(:,31:60));
% pd1=2;
% pd2=1;
% w=[0:3*pi/(xdim-1):3*pi];
% S7=sin(pd1.*w')*cos(pd2.*w);
% S7(:,1:11)=0;
% S7(:,30:end)=0;
% S7(1:50,:)=0;
% S7(60:end,:)=0;
% temp=rot90(S7);
% temp=rot90(temp);
% S7=S7+temp;
% S7=rot90(S7);
% T=zeros(60,60);
% T(10:40,50:60)=S7(25:55,50:60);
% S7=T;
% S7=fliplr(S7);
subplot(num_sources,3,19),imagesc(S7); axis square, axis off;

%*  *   *   *   *   *   *   *   *   *   *   *   *   *   *
%Source 8
%*  *   *   *   *   *   *   *   *   *   *   *   *   *   *
pd1=0.5;
pd2=0.4;
w=[0:3*pi/(xdim-1):3*pi];
S8=sin(pd1.*w')*cos(pd2.*w);
temp=zeros(60,60);
temp(1:38,1:26)=S8(1:38,1:26);
S8=fliplr(temp);
subplot(num_sources,3,22),imagesc(S8); axis square, axis off;


s=zeros(xy_dim,num_sources);
S_temp=randn(1,3600);
[f_temp,x_temp] = ksdensity((S_temp/max(abs(S_temp))),'width',0.1);


%reshaping the sources and plotting the histograms
S(1,:)=reshape(S1,1,xy_dim);
s(:,1)=(S(1,:)');
tp = (s(:,1)'-mean(s(:,1)))/(max(abs(s(:,1))));
%/(max(abs(s(:,1))));
%figure,subplot(5,1,1),hist(tp,61);
kurtosis(tp')-3
[f,x,u] = ksdensity(tp,'width',0.08);
subplot(num_sources,3,2);
hold on;
plot(x,f);
plot(x_temp,f_temp,':');title('Source pdf');
axis([-2 2 0 5]); 
%box on;
colormap gray;
S(2,:)=reshape(S2,1,xy_dim);
s(:,2)=(S(2,:)');
tp = (s(:,2)'-mean(s(:,2)))/(max(abs(s(:,2))));
%subplot(5,1,2),hist(tp,61);
kurtosis(tp')-3
[f,x,u] = ksdensity(tp,'width',0.08);
subplot(num_sources,3,5);
hold on;
plot(x_temp,f_temp,':');
plot(x,f);
axis([-2 2 0 5]);
colormap gray;
S(3,:)=reshape(S3,1,xy_dim);
s(:,3)=(S(3,:)');
tp = (s(:,3)'-mean(s(:,3)))/(max(abs(s(:,3))));
%subplot(5,1,3),hist(tp,61);
kurtosis(tp')-3
[f,x] = ksdensity(tp,'width',0.08);
subplot(num_sources,3,8);
hold on;
plot(x_temp,f_temp,':');
plot(x,f);
axis([-2 2 0 5]);
colormap gray;
S(4,:)=reshape(S4,1,xy_dim);
s(:,4)=(S(4,:)');
tp = (s(:,4)'-mean(s(:,4)))/(max(abs(s(:,4))));
%subplot(5,1,4),hist(tp,61);
kurtosis(tp')-3
[f,x] = ksdensity(tp,'width',0.08);
subplot(num_sources,3,11);
hold on;
plot(x_temp,f_temp,':');
plot(x,f);
axis([-2 2 0 5]);
colormap gray;
S(5,:)=reshape(S5,1,xy_dim);
s(:,5)=(S(5,:)');
tp = (s(:,5)'-mean(s(:,5)))/(max(abs(s(:,5))));
%subplot(5,1,5),hist(tp,61);
kurtosis(tp')-3
[f,x] = ksdensity(tp,'width',0.08);
subplot(num_sources,3,14);
hold on;
plot(x_temp,f_temp,':');
plot(x,f);
axis([-2 2 0 5]);
colormap gray;
S(6,:)=reshape(S6,1,xy_dim);
s(:,6)=(S(6,:)');
tp = (s(:,6)'-mean(s(:,6)))/(max(abs(s(:,6))));
%subplot(5,1,5),hist(tp,61);
kurtosis(tp')-3
[f,x] = ksdensity(tp,'width',0.08);
subplot(num_sources,3,17);
hold on;
plot(x_temp,f_temp,':');
plot(x,f);
axis([-2 2 0 5]);
colormap gray;
S(7,:)=reshape(S7,1,xy_dim);
s(:,7)=(S(7,:)');
tp = (s(:,7)'-mean(s(:,7)))/(max(abs(s(:,7))));
%subplot(5,1,5),hist(tp,61);
kurtosis(tp')-3
[f,x] = ksdensity(tp,'width',0.08);
subplot(num_sources,3,20);
hold on;
plot(x_temp,f_temp,':');
plot(x,f);
axis([-2 2 0 5]);
colormap gray;
S(8,:)=reshape(S8,1,xy_dim);
s(:,8)=(S(8,:)');
tp =(s(:,8)'-mean(s(:,8)))/(max(abs(s(:,8))));
%subplot(5,1,5),hist(tp,61);
kurtosis(tp')-3
[f,x] = ksdensity(tp,'width',0.08);
subplot(num_sources,3,23);
hold on;
plot(x_temp,f_temp,':');
plot(x,f);
axis([-2 2 0 5]);
colormap gray;
%*  *   *   *   *   *   *   *   *   *   *   *   *   *   *
%*  *   *                                       *   *   *
%*  *   *                                       *   *   *
%               create timecourses:
%*  *   *                                       *   *   *
%*  *   *                                       *   *   *
%*  *   *   *   *   *   *   *   *   *   *   *   *   *   *
%spike_col               = zeros([timeset 1]);
rand_col                = zeros([timeset 1]);  %Time course 5
rand_plus_constant_col  = zeros([timeset 1]);  %Time course 4
sin_col                 = zeros([timeset 1]);  %Time course 3
a_col                   = ones([timeset+20 1]);%Time course 1
derivative_col          = zeros([1 timeset+1]);%Time course 2
constant                =0.25;

%Time course 1
a_col_n = ones([timeset+20 1]);
for i=1:2:10
a_col_n(12*i+1:12*i+12)=-1;
%a_col=a_col';
end
%figure; plot(1:120,a_col);
x = (1:1:12);
y = normpdf(x,6,0.8);
y=y';
%figure,plot(1:12,y);
z=conv(a_col_n,y);
%figure,plot(1:131,z);
for i=1:100
alternating_col_new(i)=z(i+7);
end

for i=1:2:12
a_col(10*i+1:10*i+10)=-1;
%a_col=a_col';
end
%figure; plot(1:120,a_col);
x = (1:1:12);
y = normpdf(x,6,0.8);
y=y';
%figure,plot(1:12,y);
z=conv(a_col,y);
%figure,plot(1:131,z);
for i=1:timeset
alternating_col(i)=z(i+7);
end


mod_d=0.65;
%for mod = 1:40
    %mod_d = mod_d+0.05;
    %mod_d = mod_d^rand(1);
    %mod_d = mod_d^(1/rand(1));
    weight = [1, 0.01, 4, 9, 0.16, -400, -3.61, -324, -289, -0.5, -225, -196, -169, -0.01, -121, -100, -81, 64, -49, -36, -25, -16, -9, -4, 2.5, 36, 49, -64, 81, 100, 1.21, 144, 169, 196, 225, 256, 28.9, 324, 361, -0.4, -225, -196, -16.9];
    %dummy_weight = [1, 1.1, 0.5, 0.33, 0.25, 0.77, 0.57, 0.36, 0.8, 0.44, 0.8, 0.9, 1, 1.5, 2, 1.3, 3, 1.75, 2.75, 1.25];  
    %mod_d = mod_d^(1/sqrt(mod_d))/dummy_weight(mod);
    %mod_d = mod_d^(1/(mod_d));
    %mod_d = 2^(1/mod_d);
    %Time course 2
    alternating_col=alternating_col';
    for i=1:timeset
    d(i)=alternating_col(i);
    end
    d(1,101)=z(108,1);
    %dif=diff(d*real(log(weight(subj))));
    dif=diff(d*(weight(subj)));
    derivative_col=dif';

%Time course 3
w_col= [0:2*pi/(timeset-1):2*pi];
sin_col= sin(w_col');
tmp=zeros(size(sin_col));
tmp(1:50,:)=sin_col(1:50,1);
tmp(51:100,:)=sin_col(51:100,1)-1;
sin_col=tmp;

% Time course 4
rand_plus_constant_col  = 0.5 + 0.05* randn([timeset 1]);
rand_plus_constant_col(timeset-30:timeset)=rand_plus_constant_col(timeset-30:timeset)+constant; 

%Time course 5
rand_col  = 0.75 + 0.07* randn([timeset 1]);

%Time Course 6
% triangle_col=zeros(100,1);
% count=0;
% for i=1:100
%     value=count/10+ 0.05* randn;
%     %value=value+1*count;
%     triangle_col(i,1)=value;
%     count=count+1;
%     count=mod(count,10);
% end
% triangle_col=(-1)*triangle_col+1;


t=1:1:20;
x = (0.01:0.1:10);
p  = fpdf(x,5,20);
x1=zeros(1,20);
j=1;
for i=1:20
x1(1,i)=p(1,j);
j=j+2;
end;
t=1:1:100;
x2=zeros(1,100);
c=0;
for i=1:5
    for j=1:20
        x2(1,j+c)=x1(1,j);
    end;
    c=c+20;
end;
t=zeros(1,100);
t(1,1:20)=x2(1,1:20);
t(1,9:18)=t(1,9:18)+x2(1,21:30);
t(1,21:40)=t(1,1:20);
t(1,41:60)=t(1,1:20);
t(1,61:80)=t(1,1:20);
t(1,81:100)=t(1,1:20);


%Time Course 7
exponential_col=zeros(100,1);
for i=1:100
    exponential_col(i,1)=i^2/10000+.1*randn;
end

%Time Course 8
spike_col=zeros(1,100);
spike_col= 0.2 + 0.1* randn([timeset 1]);
A=23;
spike_col(35,1)=spike_col(35,1)-2;

%populate W with columns we have created
W(:,1)=alternating_col_new';
W(:,2)=derivative_col;
%W(:,2)=spike_col;
W(:,3)=sin_col;
W(:,4)=rand_plus_constant_col;
W(:,5)=rand_col;
W(:,6)=t';
W(:,7)=exponential_col;
W(:,8)=spike_col;

subplot(num_sources,3,3);   plot(1:timeset, W(:,1)); title('Timescales');
subplot(num_sources,3,6);   plot(1:timeset, W(:,2));
subplot(num_sources,3,9);   plot(1:timeset,W(:,3));
subplot(num_sources,3,12);  plot(1:timeset,W(:,4));
subplot(num_sources,3,15);  plot(1:timeset, W(:,5));
subplot(num_sources,3,18);   plot(1:timeset,W(:,6));
subplot(num_sources,3,21);  plot(1:timeset,W(:,7));
subplot(num_sources,3,24);  plot(1:timeset, W(:,8));

%generate data
%data = sources * mixing matrix
%%data=mixing matrix *sources
%X=S*W;
X=W*S;
CURRENTsub = CURRENTsub + 1;
ALLsubs(CURRENTsub,:) = (mean(X))';
%csvwrite(sprintf('Mean_X_%d',mod),mean(X));
%csvwrite(sprintf('PCA_X_%d',mod),pca(X));
%PCA_X=(pca(X))';
figure;
for i=1:timeset
subplot(10,10,i);
image(100*reshape(X(i,:),xdim, ydim));
colormap gray;
axis off;
end
end
save ALLsubs.mat ALLsubs