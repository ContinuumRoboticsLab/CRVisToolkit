function g = robotindependentmapping(kappa, phi, ell, ptsperseg)
% ROBOTINDEPENDENTMAPPING creates a framed curve for given configuration parameters
%
%   EXAMPLE
%       g = robotindependentmapping([1/40e-3;1/10e-3],[0,pi],[25e-3,20e-3],10)
%       creates a 2-segment curve with radius of curvatures 1/40 and 1/10
%       and segment lengths 25 and 20, where the second segment is rotated by pi rad.
%
%   INPUT: configuration parameters
%       kappa (nx1): segment curvatures
%       phi (nx1): segment bending plane angles
%       l (nx1): segment lengths
%       ptsperseg (nx1): number of points per segment
%                        if n=1 all segments with equal number of points
%   OUTPUT: backbone curve
%       g (n,16): backbone curve with n 4x4 transformation matrices reshaped into 1x16 vector (columnwise)
%
%   Author: Jessica Burgner-Kahrs <jbk@cs.toronto.edu>
%   Date: 2022/02/16
%   Version: 0.2
%
%   Copyright: 2023 Continuum Robotics Laboratory, University of Toronto

    arguments
        kappa (1,:) double %segment curvatures
        phi (1,:) double %segment bending plane angles
        ell (1,:) double %segment lengths
        ptsperseg (1,:) uint8 %number of points per segment
    end

    if length(kappa) ~= length(phi) || length(kappa) ~= length(ell)
        error("Dimension mismatch.")
    end
    numseg = length(kappa);
    if size(ptsperseg,2) == 1 && numseg > 1 %same number of points per segment
        ptsperseg = double(ptsperseg)*ones(1,numseg);
    end

    g = zeros(sum(ptsperseg),16);
    T_base = eye(4);
    for i=1:numseg
        T = zeros(ptsperseg(i),16);
        c_p=cos(phi(i));
        s_p=sin(phi(i));

        for j=1:ptsperseg(i)
            c_ks=cos(kappa(i)*(j-1)*(ell(i)/ptsperseg(i)));
            s_ks=sin(kappa(i)*(j-1)*(ell(i)/ptsperseg(i)));
            if kappa(i)~=0
                    T_temp = [c_p*c_p*(c_ks-1)+1 s_p*c_p*(c_ks-1) -c_p*s_ks 0 ...
                    s_p*c_p*(c_ks-1) c_p*c_p*(1-c_ks)+c_ks -s_p*s_ks 0 ...
                    c_p*s_ks s_p*s_ks c_ks 0 ...
                    (c_p*(1-c_ks))/kappa(i) (s_p*(1-c_ks))/kappa(i) s_ks/kappa(i) 1];

            else %kappa=0 ->otherwise division by zero
                    T_temp =[c_p*c_p*(c_ks-1)+1 s_p*c_p*(c_ks-1) -c_p*s_ks 0 ...
                    s_p*c_p*(c_ks-1) c_p*c_p*(1-c_ks)+c_ks -s_p*s_ks 0 ...
                    c_p*s_ks s_p*s_ks c_ks 0 ...
                    0 0 (j-1)*(ell(i)/(ptsperseg(i))) 1];
            end
            T(j,:)=reshape(T_base*reshape(T_temp,4,4),1,16);
        end
        if i==1
            g(1:ptsperseg(i),:) = T;
        else
            g(sum(ptsperseg(1:i-1))+1:sum(ptsperseg(1:i-1))+ptsperseg(i),:) = T;
        end

        T_base = reshape(T(ptsperseg(i),:),4,4);
    end

end
