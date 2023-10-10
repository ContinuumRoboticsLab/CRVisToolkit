function [fig] = draw_ctcr(g,tube_end,r_tube,options)
% DRAW_CTCR Creates a figure of a concentric tube continuum robot (ctcr)
%
%   Takes a matrix with nx16 entries, where n is the number
%   of points on the backbone curve. For each point on the curve, the 4x4
%   transformation matrix is stored columnwise (16 entries). The x- and
%   y-axis span the material orientation and the z-axis is tangent to the
%   curve.
%
%   INPUT
%   g(n,16): backbone curve with n 4x4 transformation matrices reshaped into 1x16 vector (columnwise)
%   tube_end(1,m): indices of g where ctcr tubes terminate
%   r_tube(1,m): radii of tubes
%   options:
%       tipframe (shows tip frame, default true/1)
%       baseframe (shows robot base frame, default false/0)
%       projections (shows projections of backbone curve onto
%                    coordinate axes, default false/0)
%       baseplate (shows robot base plate, default false/0)
%
%
%   Author: Jessica Burgner-Kahrs <jbk@cs.toronto.edu>
%   Date: 2023/02/16
%   Version: 0.2
%
%   Copyright: 2023 Continuum Robotics Laboratory, University of Toronto

    arguments
        g (:,16) double %backbone curve (transformation matrices stored columnwise)
        tube_end (1,:) uint8 %tube end indices
        r_tube (1,:) double % tube radius
        options.tipframe (1,1) {mustBeNumericOrLogical} = 1
        options.segframe (1,1) {mustBeNumericOrLogical} = 0
        options.baseframe (1,1) {mustBeNumericOrLogical} = 0
        options.projections (1,1) {mustBeNumericOrLogical} = 0
        options.baseplate (1,1) {mustBeNumericOrLogical} = 1
    end

    if max(tube_end)>size(g,1) || size(tube_end,2) ~= size(r_tube,2)
        error("Dimension mismatch")
    end

    curvelength = sum(vecnorm(g(2:end,13:15)'-g(1:end-1,13:15)'));
    numtubes = size(tube_end,2);

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
    hold on

    radial_pts = 16; %resolution (num point on circular cross section, increase for high detail level
    tcirc = linspace(0,2*pi,radial_pts);
    col = linspace(0.2,0.8,numtubes);
    alpha = 1;%0 = transparent

    %% draw tubes
    for j=1:numtubes
        %set tube start and end index for current tube
        if j == 1
            starttube = 1;
        else
            starttube=tube_end(j-1);
        end
        endtube = tube_end(j);
        color = col(j)*[1,1,1];

        % points on a circle in the local x-y plane
        basecirc = [r_tube(j)*sin(tcirc);r_tube(j)*cos(tcirc);zeros(1,length(tcirc));ones(1,length(tcirc))];

        % transform circle points into the tube's base frame
        basecirc_trans = reshape(g(starttube,:),4,4)*basecirc;

        %draw patches to fill in the circle
        patch(basecirc_trans(1,:),basecirc_trans(2,:),basecirc_trans(3,:),color,'EdgeAlpha',0,'FaceAlpha',alpha)
        material shiny

        %Loop to draw each cylindrical segment for tube
        for i=starttube:endtube-1
            basecirc_trans = reshape(g(i,:),4,4)*basecirc; %current frame circle points
            basecirc_trans_ahead = reshape(g(i+1,:),4,4)*basecirc; %next frame circle points

            %loop to draw each square patch for this segment
            for k=1:radial_pts-1
                xedge = [basecirc_trans(1,k),basecirc_trans(1,k+1),basecirc_trans_ahead(1,k+1),basecirc_trans_ahead(1,k)];
                yedge = [basecirc_trans(2,k),basecirc_trans(2,k+1),basecirc_trans_ahead(2,k+1),basecirc_trans_ahead(2,k)];
                zedge = [basecirc_trans(3,k),basecirc_trans(3,k+1),basecirc_trans_ahead(3,k+1),basecirc_trans_ahead(3,k)];
                patch(xedge,yedge,zedge,color,'EdgeAlpha',0,'FaceAlpha',alpha)
            end
        end

        %circle points at tube end
        basecirc_trans = reshape(g(endtube,:),4,4)*basecirc;
        %draw patches to fill in the circle
        patch(basecirc_trans(1,:),basecirc_trans(2,:),basecirc_trans(3,:),color,'EdgeAlpha',.3,'FaceAlpha',alpha)
        material shiny
    end

    %% Projections
    if options.projections
        plot3(g(:,13), ax.YLim(1)*ones(size(g, 1)), g(:,15), 'LineWidth', 2, 'Color',[0 1 0]); % project in x-z axis
        plot3(ax.XLim(1)*ones(size(g, 1)), g(:,14), g(:,15), 'LineWidth', 2, 'Color',[1 0 0]); % project in y-z axis
        plot3(g(:,13), g(:,14), zeros(size(g, 1)), 'LineWidth', 2, 'Color',[0 0 1]); % project in x-y axis
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
        for i=1:numtubes
            quiver3(g(tube_end(i),13),g(tube_end(i),14),g(tube_end(i),15),g(tube_end(i),1),g(tube_end(i),2),g(tube_end(i),3),0.01,'LineWidth',3,'Color',[1 0 0]);
            quiver3(g(tube_end(i),13),g(tube_end(i),14),g(tube_end(i),15),g(tube_end(i),5),g(tube_end(i),6),g(tube_end(i),7),0.01,'LineWidth',3,'Color',[0 1 0])
            quiver3(g(tube_end(i),13),g(tube_end(i),14),g(tube_end(i),15),g(tube_end(i),9),g(tube_end(i),10),g(tube_end(i),11),0.01,'LineWidth',3,'Color',[0 0 1]);
        end
    end

    if options.baseframe
        quiver3(0,0,0,1,0,0,0.01,'LineWidth',3,'Color',[1 0 0]);
        quiver3(0,0,0,0,1,0,0.01,'LineWidth',3,'Color',[0 1 0]);
        quiver3(0,0,0,0,0,1,0.01,'LineWidth',3,'Color',[0 0 1]);
    end


