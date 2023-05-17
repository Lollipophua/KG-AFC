function new = maxmin(old) 
maxold = max(old);
minold = min(old);
m = size(old,1);
maxnew = repmat(maxold,m,1);
minnew = repmat(minold,m,1);
new = (old-minnew)./(maxnew-minnew);
% m = size(old,1);
% for r = 1:22
%     maxold = max(old(:,r));
%     minold = min(old(:,r));
%     maxnew = repmat(maxold,m,1);
%     minnew = repmat(minold,m,1);
%     new(:,r) = (old(:,r)-minnew)./(maxnew-minnew);
% end