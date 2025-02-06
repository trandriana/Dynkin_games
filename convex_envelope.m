function [Vp] = convex_envelope(U, p)
    % This function returns the convex envelope of a continuous function u

    V  = squeeze(U);
    nb = size(V);
    Vp = zeros(nb);
    
    parfor j =1:nb(1)
        PP = [p, V(j, :)'; 0.1, 1.e3]; % I add an artificial point
        % to infinity to avoid degeneracy
    
        KK = convhull(PP);
        low = [true;  diff(PP(KK,1))>=0];
        PP(KK(low),:);
        Q = unique(PP(KK(low),:), 'rows');
 
        Vp(j,:) = interpn(Q(:,1), Q(:,2), p);
    end
end