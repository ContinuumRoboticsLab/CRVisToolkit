function [fig] = draw_tdcr(g,seg_end,r_disk,r_height,options)
% DRAW_TDCR Creates a figure of a tendon-driven continuum robot (tdcr)
%
%   Takes a matrix with nx16 entries, where n is the number
%   of points on the backbone curve. For each point on the curve, the 4x4
%   transformation matrix is stored columnwise (16 entries). The x- and
%   y-axis span the material orientation and the z-axis is tangent to the
%   curve.
%
%   INPUT
%   g(n,16): backbone curve with n 4x4 transformation matrices reshaped into 1x16 vector (columnwise) 
%   seg_end(1,m): indices of g where tdcr segments terminate
%   r_disk: radius of spacer disks
%   r_height: height of space disks
%   options: 
%       tipframe (shows tip frame, default true/1)
%       segframe (shows segment end frames, default false/0)
%       baseframe (shows robot base frame, default false/0)
%       projections (shows projections of backbone curve onto 
%                    coordinate axes, default false/0)
%       baseplate (shows robot base plate, default false/0)
%
%
%   Author: Jessica Burgner-Kahrs <jbk@cs.toronto.edu>
%   Date: 2023/01/04
%   Version: 0.1
%
%   Copyright: 2023 Continuum Robotics Laboratory, University of Toronto

    arguments
        g (:,16) double %backbone curve (transformation matrices stored columnwise) 
        seg_end (1,:) uint8 %segment end indices
        r_disk double = 2.5*1e-3 %radius of spacer disk
        r_height double =1.5*1e-3 %height of spacer disk
        options.tipframe (1,1) {mustBeNumericOrLogical} = 1
        options.segframe (1,1) {mustBeNumericOrLogical} = 0
        options.baseframe (1,1) {mustBeNumericOrLogical} = 0
        options.projections (1,1) {mustBeNumericOrLogical} = 0
        options.baseplate (1,1) {mustBeNumericOrLogical} = 1
    end
    
    if size(g,1)<length(seg_end) || max(seg_end)>size(g,1)
        error("Dimension mismatch")
    end

    numseg = length(seg_end);
    curvelength = sum(vecnorm(g(2:end,13:15)'-g(1:end-1,13:15)'));
    
    %% Setup figure
    fig=figure;
    hold on
    set(fig,'Position',[0 0 1280 1024]);

    % Axes, Labels
    clearance = 0.03;
    axis([-(max(abs(g(:,13)))+clearance) max(abs(g(:,13)))+clearance ...
          -(max(abs(g(:,14)))+clearance) max(abs(g(:,14)))+clearance ...
          0 curvelength+clearance])
    ax=gca;
    ax.GridAlpha=0.3;

    xlabel('x (m)')
    ylabel('y (m)')
    zlabel('z (m)')
    grid on
    view([0.5 0.5 0.5])
    daspect([1 1 1])

    col = linspace(0.2,0.8,numseg);

    %% backbone
    plot3(g(1:seg_end(1),13), g(1:seg_end(1),14),g(1:seg_end(1),15),...
        'LineWidth',5,'Color',col(1)*[1 1 1]);
    for i=2:numseg
        plot3(g(seg_end(i-1):seg_end(i),13), g(seg_end(i-1):seg_end(i),14),g(seg_end(i-1):seg_end(i),15),...
            'LineWidth',5,'Color',col(i)*[1 1 1]);
    end
    
    %% Projections
    if options.projections
        plot3(g(:,13), ax.YLim(1)*ones(size(g, 1)), g(:,15), 'LineWidth', 2, 'Color',[0 1 0]); % project in x-z axis
        plot3(ax.XLim(1)*ones(size(g, 1)), g(:,14), g(:,15), 'LineWidth', 2, 'Color',[1 0 0]); % project in y-z axis
        plot3(g(:,13), g(:,14), zeros(size(g, 1)), 'LineWidth', 2, 'Color',[0 0 1]); % project in x-y axis
    end

    %% tendons
    tendon1=zeros(seg_end(numseg),3);
    tendon2=tendon1;
    tendon3=tendon1;
    
    % tendon locations on disk
    r1=[0 r_disk 0]';
    r2=[cos(30*pi/180)*r_disk -sin(30*pi/180)*r_disk 0]';
    r3=[-cos(30*pi/180)*r_disk -sin(30*pi/180)*r_disk 0]';
    
    for i=1:seg_end(numseg)
        RotMat=reshape([g(i,1:3) g(i,5:7) g(i,9:11)],3,3);
        tendon1(i,1:3)=RotMat*r1+g(i,13:15)';
        tendon2(i,1:3)=RotMat*r2+g(i,13:15)';
        tendon3(i,1:3)=RotMat*r3+g(i,13:15)';
    end
    
    plot3(tendon1(1:end,1),tendon1(1:end,2),tendon1(1:end,3),'Color',[0 0 0]);
    plot3(tendon2(1:end,1),tendon2(1:end,2),tendon2(1:end,3),'Color',[0 0 0]);
    plot3(tendon3(1:end,1),tendon3(1:end,2),tendon3(1:end,3),'Color',[0 0 0]);
    
    % draw spheres to represent tendon location at end disks
    [x, y, z]=sphere;
    radius = 0.75e-3;
    for i=1:numseg
        surf(x*radius+tendon1(seg_end(i),1),y*radius+tendon1(seg_end(i),2),z*radius+tendon1(seg_end(i),3), 'FaceColor',[0 0 0]);
        surf(x*radius+tendon2(seg_end(i),1),y*radius+tendon2(seg_end(i),2),z*radius+tendon2(seg_end(i),3), 'FaceColor',[0 0 0]);
        surf(x*radius+tendon3(seg_end(i),1),y*radius+tendon3(seg_end(i),2),z*radius+tendon3(seg_end(i),3), 'FaceColor',[0 0 0]);
    end
    
    %% spacer disks
    for i=1:size(g,1)
        seg = find(seg_end >= i,1);
        if isempty(seg)
            color = col(1)*[1 1 1];
        else
            color = col(seg)*[1 1 1];
        end

        RotMat=reshape([g(i,1:3) g(i,5:7) g(i,9:11)],3,3);
        normal=RotMat(1:3,3)';
        pos=g(i,13:15)'-RotMat*[0 0 r_height/2]';
         
        theta=0:0.05:2*pi;
        v=null(normal);
        lowercirc=repmat(pos,1,size(theta,2))+r_disk*(v(:,1)*cos(theta)+v(:,2)*sin(theta));
        fill3(lowercirc(1,:),lowercirc(2,:),lowercirc(3,:),color);

        pos=g(i,13:15)'+RotMat*[0 0 r_height/2]';
        uppercirc=repmat(pos,1,size(theta,2))+r_disk*(v(:,1)*cos(theta)+v(:,2)*sin(theta));
        fill3(uppercirc(1,:),uppercirc(2,:),uppercirc(3,:),color);
    
        x=[lowercirc(1,1:end); uppercirc(1,1:end)];
        y=[lowercirc(2,1:end); uppercirc(2,1:end)];
        z=[lowercirc(3,1:end); uppercirc(3,1:end)];
    
        surf(x,y,z,'FaceColor',color,'MeshStyle','row');
    end
    
    %% base plate
    if options.baseplate
        color = [1 1 1]*0.9;
        squaresize = 0.02;
        thickness = 0.001;
        patch([-1 1 1 -1]*squaresize,[-1 -1 1 1]*squaresize,[-1 -1 -1 -1]*thickness,color)
        patch([-1 1 1 -1]*squaresize,[-1 -1 1 1]*squaresize,[0 0 0 0],color)
        patch([1 1 1 1]*squaresize,[-1 -1 1 1]*squaresize,[-1 0 0 -1]*thickness,color)
        patch([-1 -1 -1 -1]*squaresize,[-1 -1 1 1]*squaresize,[-1 0 0 -1]*thickness,color)
        patch([-1 1 1 -1]*squaresize,[-1 -1 -1 -1]*squaresize,[-1 -1 0 0]*thickness,color)
        patch([-1 1 1 -1]*squaresize,[1 1 1 1]*squaresize,[-1 -1 0 0]*thickness,color)
    end

    %% Coordinate Frames
    if options.tipframe && ~options.segframe
        quiver3(g(end,13),g(end,14),g(end,15),g(end,1),g(end,2),g(end,3),0.01,'LineWidth',3,'Color',[1 0 0]);
        quiver3(g(end,13),g(end,14),g(end,15),g(end,5),g(end,6),g(end,7),0.01,'LineWidth',3,'Color',[0 1 0])
        quiver3(g(end,13),g(end,14),g(end,15),g(end,9),g(end,10),g(end,11),0.01,'LineWidth',3,'Color',[0 0 1]);
    end  
    
    if options.segframe
        for i=1:numseg
            quiver3(g(seg_end(i),13),g(seg_end(i),14),g(seg_end(i),15),g(seg_end(i),1),g(seg_end(i),2),g(seg_end(i),3),0.01,'LineWidth',3,'Color',[1 0 0]);
            quiver3(g(seg_end(i),13),g(seg_end(i),14),g(seg_end(i),15),g(seg_end(i),5),g(seg_end(i),6),g(seg_end(i),7),0.01,'LineWidth',3,'Color',[0 1 0])
            quiver3(g(seg_end(i),13),g(seg_end(i),14),g(seg_end(i),15),g(seg_end(i),9),g(seg_end(i),10),g(seg_end(i),11),0.01,'LineWidth',3,'Color',[0 0 1]);
        end
    end

    if options.baseframe
        quiver3(0,0,0,1,0,0,0.01,'LineWidth',3,'Color',[1 0 0]);
        quiver3(0,0,0,0,1,0,0.01,'LineWidth',3,'Color',[0 1 0]);
        quiver3(0,0,0,0,0,1,0.01,'LineWidth',3,'Color',[0 0 1]);
    end
end

