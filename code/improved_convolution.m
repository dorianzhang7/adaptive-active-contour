% Customized multi-core convolution

function [ Kout ] = improved_convolution( I,Ksigma,Ki,Kj)
%   此处显示详细说明
% 扩充边界,rad表示卷积核的长度
rad = size(Ksigma,1);
r = (rad-1)/2;

if (Ki-r <1 || Kj-r <1 || Ki+r>size(I,1) || Kj+r>size(I,2))
    myF = zeros(r, 2*r + size(I, 2));
    myF(r+1:(size(I, 1) + r), 1:r) = 0;
    myF(r+1:(size(I, 1) + r),r+1:(size(I, 2) + r)) = I;
    myF(r+1:(size(I, 1) + r),(size(I, 2) + r+1):(size(I, 2) + 2*r)) = 0;
    myF((size(I, 1) + r+1):(size(I, 1) + 2*r), 1:(size(I, 2) + 2*r)) = 0;
    % 扩充完边界后，更新下标
    Ki = Ki + r;
    Kj = Kj + r;
    Kout = 0;
    for ii = (Ki-r):(Ki+r)
        for jj = (Kj-r):(Kj+r)
            si = (ii -(Ki-r))+1;
            sj = (jj -(Kj-r))+1;
            Kout = Kout + Ksigma(si, sj)*myF(ii,jj);
        end
    end
else
    Kout = 0;
    for ii = (Ki-r):(Ki+r)
        for jj = (Kj-r):(Kj+r)
            si = (ii -(Ki-r))+1;
            sj = (jj -(Kj-r))+1;
            Kout = Kout + Ksigma(si, sj)*I(ii,jj);
        end
    end
end

end

