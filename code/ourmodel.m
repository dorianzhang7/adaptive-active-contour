%% model

close all;

%% initial input

%-----------------------------medical image-------------------------------

    Img = imread('inputs/1.png');
    u0 = double(Img(:,:,1));    
    u0(u0 < 5) = 5;
    [r, c] = size(u0);
    phi = ones(r, c) .* 2;
    phi(35:55,45:65)= -2;
    maxIter = 10;   

%     Img = imread('inputs/2.png');
%     u0 = double(Img(:,:,1));    
%     u0(u0 < 5) = 5;
%     [r, c] = size(u0);
%     phi = ones(r, c) .* 2;
%     phi(15:35,45:55)= -2;
%     maxIter = 13; 
    
%     Img = imread('inputs/3.png');
%     u0 = double(Img(:,:,1));    
%     u0(u0 < 5) = 5;
%     [r, c] = size(u0);
%     phi = ones(r, c) .* 2;
%     phi(30:60,40:100) = -2;
%     maxIter = 20;

%     Img = imread('inputs/4.png');
%     u0 = double(Img(:,:,1));    
%     u0(u0 < 5) = 5;
%     [r, c] = size(u0);
%     phi = ones(r, c) .* 2;
%     phi(60:90,100:130)= -2;
%     maxIter =8;
    
%     Img = imread('inputs/5.png');
%     u0 = double(Img(:,:,1));    
%     u0(u0 < 5) = 5;
%     [r, c] = size(u0);
%     phi = ones(r, c) .* 2;
%     phi(65:105,55:95)= -2;
%     maxIter = 12;
     
%     Img = imread('inputs/6.png');
%     u0 = double(Img(:,:,1));    
%     u0(u0 < 5) = 5;
%     [r, c] = size(u0);
%     phi = ones(r, c) .* 2;
%     phi(40:50,40:50)= -2;
%     maxIter = 20;

% ---------------------------nature image-------------------------------
%
%     Img = imread('inputs/7.png');
%     u0 = double(Img(:,:,1));    
%     u0(u0 < 5) = 5;
%     [r, c] = size(u0);
%     phi = ones(r, c) .* 2;
%     phi(70:110,80:120)=-2;
%     maxIter =5;
    
%     Img = imread('inputs/8.png');
%     u0 = double(Img(:,:,1));    
%     u0(u0 < 5) = 5;
%     [r, c] = size(u0);
%     phi = ones(r, c) .* 2;
%     phi(90:170,35:115)=-2;
%     maxIter =1;
      
%     Img = imread('inputs/9.png');
%     u0 = double(Img(:,:,1));    
%     u0(u0 < 5) = 5;
%     [r, c] = size(u0);
%     phi = ones(r, c) .* 2;
%     phi(50:80,40:80)=-2;
%     maxIter =6;
    
%     Img = imread('inputs/10.png');
%     u0 = double(Img(:,:,1));    
%     u0(u0 < 5) = 5;
%     [r, c] = size(u0);
%     phi = ones(r, c) .* 2;
%     phi(100:150,50:90)=-2;
%     maxIter =15;
    
%     Img = imread('inputs/11.png');
%     u0 = double(Img(:,:,1));    
%     u0(u0 < 5) = 5;
%     [r, c] = size(u0);
%     phi = ones(r, c) .* 2;
%     phi(100:140,30:80)=-2;
%     maxIter =3;
      
%     show the given image with initial contour;

    subplot(2, 2, 1);
    imshow(Img, [0, 255]), hold on;
    contour(phi, [.5 .5], 'r', 'linewidth', 3);
    phi_init = phi;
  
    lambda = 1;      % coefficient of cv term;

    nu = .5;        % coefficient of smooth term, increase it can obtain more smooth segmentation result;
    mu = 1;         % coefficient of penalty term with respect to level set function \phi,

    epsilon = 1;    % parameter of regularized Heaviside function and its derivative;
    timeStep = .1;  % time step.

    nu1 = 0.001*255*255; 
    
    G1=zeros(r,c);
     rid = 5;
     for i=1:r
        for j=1:c
            temp5=0;
            for x=i-rid:i+rid
                for y=j-rid:j+rid
                    if x>0 && x<r && y>0 && y<c                
                        d=sqrt((i-x)^2+(j-y)^2); 
                        temp5 = temp5 + 1/(d+1) * abs((u0(x,y) - u0(i,j)) * log(u0(x,y)/u0(i,j)) + 10e-6);
                    end
                end
            end
            G1(i,j) = temp5;
         end
     end
    G2 =  ( G1 - min(min(G1)) )./ ( max(max(G1)) -  min(min(G1)) );
     
    similar = 1 - G2;
    sigma_local = 10*similar;

    sigma_local(sigma_local < 2) = 2;
    [r,c] = size(u0);

    s_G = fspecial('gaussian');
    u0_smooth=conv2( u0,s_G,'same');

    mean = sum(u0(:))/(r*c);
    u0_mean = ones(r,c)*mean;
    s0 = abs(double(u0_mean) - double(u0_smooth));
    s0 = s0./max(max(abs(s0)));

%% evolution process
    tic     % record the starting time of evolution process;
    for k = 1:maxIter

        if mod(k, 1) == 0 || k == 1
        pause(.1);
        subplot(2, 2, 2), 
        imshow(Img, [0, 255]), hold on;
        contour(phi, [0 0], 'r', 'linewidth', 1);
        % show the number of iteration;
        iterNum=[num2str(k), ' iterations'];
        title(iterNum);

        hold off;
        end 
        
        phi = NeumannBoundCond(phi);    
        K = curvature_central(phi);     % compute div(D\phi/(|D\phi|));
        Ks = curvature_centrals(phi,s0);

        % compute dirac function and H function 
        D = (epsilon/pi)./(epsilon^2.+phi.^2);
        H = (1 + (2/pi) * atan(phi./epsilon)) / 2;

        out = find(phi <= 0);
        in = find(phi > 0);
        % compute average intensity from inside and outside
        cv_c1 = sum(u0(in))/(length(in)+eps);
        cv_c2 = sum(u0(out))/(length(out)+eps);

        lambda0 = 10;

        F = lambda0*2*(u0 + cv_c1 - 2*sqrt(u0*cv_c1)) - lambda0*2*(u0 + cv_c2 - 2*sqrt(u0*cv_c2));

        [r,c] = size(u0);
        I_local = u0.*H;
        KI_local = zeros(r,c);
        KONE_local = zeros(r,c);
        local_c1 = zeros(r,c);
        local_c2 = zeros(r,c);

        for i = 1:r
            for j = 1:c
                Ksigma_local = fspecial('gaussian',round(2*sigma_local(i,j))*2+1, sigma_local(i,j));

                KI_local(i,j) = improved_convolution(u0,Ksigma_local,i,j);
                KONE_local(i,j) = improved_convolution(ones(size(u0)),Ksigma_local,i,j);
                local_c1(i,j) = improved_convolution(H, Ksigma_local,i,j);
                local_c2(i,j) = improved_convolution(I_local, Ksigma_local,i,j);
            end
        end
        local_f1 = local_c2./(local_c1+10e-6);
        local_f2 = (KI_local - local_c2)./(KONE_local - local_c1+10e-6);


        lambda1 = 100; 

        dataForce_local2 = lambda1*2*(u0 + local_f1 - 2*sqrt(u0.*local_f1)) - lambda1*2*(u0 + local_f2 - 2*sqrt(u0.*local_f2));

        region_term = -D.*(dataForce_local2); 
        length_term = 2* nu1.*D.*Ks;             
        fitting_term = -D.*F;                % fitting term;
        penal_term = mu.*(4*del2(phi)-K);    % penalizing term, proposed in Chunming Li's paper: 

    % medical image
        s = exp(-0.2.*(maxIter/k));  

    % nature image
%         s = exp(-2.*(maxIter/k)); 
        
        phi = phi + timeStep.* (s.*region_term + (1-s).*fitting_term +length_term +penal_term);
        
        subplot(2,2,3);
        imshow(mat2gray(s.*region_term + (1-s).*fitting_term));

        subplot(2,2,4);
        mesh(-phi); 
        colormap Jet;
    end
     
    toc     % display evolution time;
    
    pause(.1);
    subplot(2, 2, 2), 
    imshow(Img, [0, 255]), hold on;
    contour(phi, [0 0], 'r', 'linewidth', 3);
    % show the number of iteration;
    iterNum=[num2str(k), ' iterations'];
    title(iterNum);

    hold off;
    
 %% º∆À„diceœ‡À∆∂»
% groundtruth = mat2gray(groundtruth);
% 
% similarity = dice(double(im2bw(-phi)),double(im2bw(groundtruth)));
% subplot(2, 2, 3),
% imshowpair(phi, groundtruth);
% title(['Dice Index = ' num2str(similarity)]);
    

%% other function definitions:
function g = NeumannBoundCond(f)
% Neumann boundary condition
[nrow,ncol] = size(f);
g = f;
g([1 nrow],[1 ncol]) = g([3 nrow-2],[3 ncol-2]);  
g([1 nrow],2:end-1) = g([3 nrow-2],2:end-1);          
g(2:end-1,[1 ncol]) = g(2:end-1,[3 ncol-2]);  
end 

function k = curvature_central(u)                       
% compute curvature
[ux,uy] = gradient(u);                                  
normDu = sqrt(ux.^2+uy.^2+1e-10);                       % the norm of the gradient plus a small possitive number 
                                                        % to avoid division by zero in the following computation.
Nx = ux./normDu;                                       
Ny = uy./normDu;
[nxx,~] = gradient(Nx);                              
[~,nyy] = gradient(Ny);                              
k = nxx+nyy;                                            % compute divergence
end

function ks = curvature_centrals(u,s0)                       
[ux,uy] = gradient(u);                                  
normDu = sqrt(ux.^2+uy.^2+1e-10);                       % the norm of the gradient plus a small possitive number 
                                                        % to avoid division by zero in the following computation.
Nx = ux./normDu;                                       
Ny = uy./normDu;
Nxs = s0.*Nx;
Nys = s0.*Ny;
[nxx,~] = gradient(Nxs);                              
[~,nyy] = gradient(Nys);                              
ks = nxx+nyy;                                            % compute divergence
end

